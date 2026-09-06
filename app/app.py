"""
Local demo web app to check the LightGBM HbA1c-trajectory model.

Run:   python app.py       (then open http://127.0.0.1:5000 )

You enter a patient's clinical values; the app builds the 180-feature vector the model expects
(numeric features + token-count features) and shows the predicted probability of the next HbA1c
being improved / stable / worsened.

NOTE: this is a single-snapshot approximation for checking the model. Training-time feature vectors
aggregate a patient's whole pre-index history across visits; here we build a one-visit snapshot with
sensible defaults (missing tokens) for anything not entered. It is a research prototype, not a
validated clinical tool.
"""
import os, numpy as np, lightgbm as lgb
from flask import Flask, request, render_template_string

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = lgb.Booster(model_file=os.path.join(HERE, "lgbm_temporal.txt"))
FEATURES = MODEL.feature_name()            # exact order the model expects
CLASSES = ["improved", "stable", "worsened"]
COLORS = {"improved": "#2e7d32", "stable": "#f9a825", "worsened": "#c62828"}

# ---------- binning (identical thresholds to the training pipeline) ----------
def bin_hba1c(v):
    for hi, t in [(6, "LE_6_0"), (7, "6_0_7_0"), (8, "7_0_8_0"), (9, "8_0_9_0"),
                  (10, "9_0_10_0"), (12, "10_0_12_0")]:
        if v <= hi: return t
    return "GT_12_0"
def bin_fbs(v):
    for hi, t in [(110, "LE_110"), (130, "110_130"), (160, "130_160"), (200, "160_200"), (400, "200_400")]:
        if v <= hi: return t
    return "GT_400"
def bin_creat(v):
    for hi, t in [(0.6, "LE_0_6"), (0.9, "0_6_0_9"), (1.2, "0_9_1_2"), (2.0, "1_2_2_0")]:
        if v <= hi: return t
    return "GT_2_0"
def bin_bun(v):
    for hi, t in [(7, "LE_7"), (20, "7_20"), (30, "20_30"), (50, "30_50")]:
        if v <= hi: return t
    return "GT_50"
def bin_ldl(v):
    for hi, t in [(70, "LE_70"), (100, "70_100"), (130, "100_130"), (160, "130_160"), (190, "160_190")]:
        if v <= hi: return t
    return "GT_190"
def bin_trig(v):
    for hi, t in [(150, "LE_150"), (200, "150_200"), (300, "200_300"), (500, "300_500")]:
        if v <= hi: return t
    return "GT_500"
def bin_chol(v):
    for hi, t in [(150, "LE_150"), (200, "150_200"), (240, "200_240"), (300, "240_300")]:
        if v <= hi: return t
    return "GT_300"
def age_band(a):
    for hi, t in [(30, "LE_30"), (40, "30_40"), (50, "40_50"), (60, "50_60"),
                  (70, "60_70"), (80, "70_80"), (90, "80_90")]:
        if a <= hi: return t
    return "GT_90"

LAB_SPECS = [  # (form field, token prefix, binner, unit)
    ("fbs", "FBS", bin_fbs, "mg/dL"),
    ("creatinine", "CREATININE", bin_creat, "mg/dL"),
    ("bun", "BUN", bin_bun, "mg/dL"),
    ("ldl", "LDL", bin_ldl, "mg/dL"),
    ("triglycerides", "TRIGLYCERIDES", bin_trig, "mg/dL"),
    ("chol", "TOTAL_CHOLESTEROL", bin_chol, "mg/dL"),
]
MED_CLASSES = ["METFORMIN", "SULFONYLUREA", "DPP4", "SGLT2", "GLP1", "INSULIN_BASAL", "INSULIN_BOLUS"]
COMORBIDITIES = ["HTN", "DYSLIPIDEMIA", "CKD", "ASCVD", "HF", "MASLD"]

def build_vector(form):
    tok = {}                                  # token -> count
    def add(name, n=1): tok[name] = tok.get(name, 0) + n

    # --- index HbA1c (required) ---
    hba1c = float(form["baseline_hba1c"])
    add(f"NUM_HBA1C_{bin_hba1c(hba1c)}"); add("LABAGE_HBA1C_0_30D")

    # --- other labs (optional) ---
    for field, prefix, binner, _ in LAB_SPECS:
        val = form.get(field, "").strip()
        if val:
            add(f"NUM_{prefix}_{binner(float(val))}"); add(f"LABAGE_{prefix}_0_30D")
        else:
            add(f"NUM_{prefix}_MISSING"); add(f"LABAGE_{prefix}_MISSING")

    # --- medications ---
    active = [m for m in MED_CLASSES if form.get(f"med_{m}")]
    for m in active: add(f"MED_ON_{m}")
    if active:
        add(f"THERAPY_NAGENTS_{min(len(active), 6)}")
        add("THERAPY_MONOCOMBO_1" if len(active) >= 2 else "THERAPY_MONOCOMBO_0")

    # --- diagnoses / comorbidities ---
    dxn = int(form.get("dxcount", "0") or 0)
    add("DXCOUNT_0" if dxn == 0 else "DXCOUNT_1_2" if dxn <= 2 else "DXCOUNT_3_5" if dxn <= 5 else "DXCOUNT_GT5")
    for c in COMORBIDITIES:
        if form.get(f"com_{c}"): add(f"COM_HAS_{c}")

    # --- demographics ---
    age = float(form["age"])
    add(f"DEM_AGE_{age_band(age)}")
    sex = form.get("sex", "MISSING")
    add(f"DEM_SEX_{sex}" if sex in ("FEMALE", "MALE") else "DEM_SEX_MISSING")
    sex_code = {"FEMALE": 0, "MALE": 1}.get(sex, -1)

    # --- vitals: no timestamp in the data -> always missing (as in training) ---
    for v in ["SBP", "DBP", "BMI", "WEIGHT", "HEIGHT"]: add(f"VITAL_{v}_MISSING")

    # --- structural tokens for one visit ---
    n_visits = max(1, int(form.get("n_visits", "1") or 1))
    add("SPECIAL_CLS"); add("SPECIAL_CTX"); add("SPECIAL_SEP"); add("SPECIAL_EV", n_visits)

    seq_len = sum(tok.values())
    numeric = {"baseline_hba1c": hba1c, "age": age, "sex_code": sex_code,
               "n_visits": n_visits, "seq_len": seq_len}

    # assemble in the model's feature order
    vec = np.zeros((1, len(FEATURES)), dtype=np.float32)
    unknown = []
    for name, val in {**tok, **numeric}.items():
        if name in FEAT_INDEX:
            vec[0, FEAT_INDEX[name]] = val
        else:
            unknown.append(name)
    return vec, unknown

FEAT_INDEX = {n: i for i, n in enumerate(FEATURES)}

PAGE = """
<!doctype html><html><head><meta charset="utf-8"><title>HbA1c trajectory - LightGBM</title>
<style>
 body{font-family:system-ui,Segoe UI,Arial;margin:0;background:#0f1419;color:#e6edf3}
 .wrap{max-width:900px;margin:0 auto;padding:24px}
 h1{font-size:22px} h2{font-size:15px;color:#9db3c8;margin:18px 0 6px;text-transform:uppercase;letter-spacing:.5px}
 .card{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:18px;margin-bottom:16px}
 label{display:block;font-size:13px;margin:8px 0 3px;color:#c9d6e2}
 input[type=number],select{width:100%;padding:8px;border:1px solid #30363d;border-radius:6px;background:#0d1117;color:#e6edf3}
 .grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}
 .chk{display:inline-flex;align-items:center;gap:6px;margin:4px 12px 4px 0;font-size:13px}
 button{background:#238636;color:#fff;border:0;padding:12px 22px;border-radius:8px;font-size:15px;cursor:pointer;margin-top:12px}
 .bar{height:26px;border-radius:5px;color:#fff;font-size:13px;line-height:26px;padding-left:8px;margin:4px 0;white-space:nowrap}
 .pred{font-size:20px;font-weight:700;margin-bottom:10px}
 .note{color:#8b97a3;font-size:12px;margin-top:10px}
</style></head><body><div class="wrap">
<h1>HbA1c trajectory - LightGBM checker</h1>
<p class="note">Predicts the direction of the next HbA1c (90-450 days ahead). Research prototype, not for clinical use.</p>
<div class="card" style="border-color:#3d5a80">
  <h2 style="margin-top:0">How to read this</h2>
  <p class="note" style="font-size:13px;line-height:1.5">
  This model predicts the <b>direction the HbA1c will move</b>, not an overall risk score.
  <b style="color:#2e7d32">Improved</b> = the next HbA1c is likely to <b>fall</b> by at least 0.5 points;
  <b style="color:#c62828">worsened</b> = likely to <b>rise</b> by at least 0.5;
  <b style="color:#f9a825">stable</b> = little change.
  Because very high values tend to come down and near-target values have little room to fall, a
  <b>high current HbA1c usually predicts "improved" (regression to the mean)</b>. "Improved" does not
  mean the patient reaches target: someone can improve from 12% to 10% and still be poorly controlled.
  To see the effect of risk factors, compare patients at the <b>same baseline HbA1c</b>; adding risk
  features then increases "worsened".
  </p>
</div>
{% if result %}
<div class="card">
  <div class="pred">Prediction: <span style="color:{{result.color}}">{{result.label|upper}}</span></div>
  {% for c,p in result.probs %}
    <div class="bar" style="background:{{result.colors[c]}};width:{{ (p*100)|round(1) }}%">{{c}} {{ (p*100)|round(1) }}%</div>
  {% endfor %}
  <p style="margin-top:12px;font-size:14px;color:#c9d6e2">{{result.message}}</p>
  {% if result.unknown %}<p class="note">Ignored unknown tokens: {{result.unknown|join(', ')}}</p>{% endif %}
</div>
{% endif %}
<form method="post"><div class="card">
  <h2>Core (required)</h2>
  <div class="grid">
    <div><label>Baseline HbA1c (%)</label><input type="number" step="0.1" name="baseline_hba1c" value="{{form.get('baseline_hba1c','8.0')}}" required></div>
    <div><label>Age (years)</label><input type="number" step="1" name="age" value="{{form.get('age','58')}}" required></div>
    <div><label>Sex</label><select name="sex">
       <option value="FEMALE" {{'selected' if form.get('sex')=='FEMALE'}}>Female</option>
       <option value="MALE" {{'selected' if form.get('sex')=='MALE'}}>Male</option>
       <option value="MISSING" {{'selected' if form.get('sex')=='MISSING'}}>Unknown</option></select></div>
  </div>
  <h2>Recent labs (optional; leave blank if unknown)</h2>
  <div class="grid">
    <div><label>Fasting glucose (mg/dL)</label><input type="number" step="0.1" name="fbs" value="{{form.get('fbs','')}}"></div>
    <div><label>Creatinine (mg/dL)</label><input type="number" step="0.01" name="creatinine" value="{{form.get('creatinine','')}}"></div>
    <div><label>BUN (mg/dL)</label><input type="number" step="0.1" name="bun" value="{{form.get('bun','')}}"></div>
    <div><label>LDL (mg/dL)</label><input type="number" step="1" name="ldl" value="{{form.get('ldl','')}}"></div>
    <div><label>Triglycerides (mg/dL)</label><input type="number" step="1" name="triglycerides" value="{{form.get('triglycerides','')}}"></div>
    <div><label>Total cholesterol (mg/dL)</label><input type="number" step="1" name="chol" value="{{form.get('chol','')}}"></div>
  </div>
  <h2>Medications on</h2>
  <div>{% for m in meds %}<label class="chk"><input type="checkbox" name="med_{{m}}" {{'checked' if form.get('med_'+m)}}> {{m.replace('_',' ').title()}}</label>{% endfor %}</div>
  <h2>Diagnoses</h2>
  <div class="grid"><div><label>Number of recorded diagnoses</label><input type="number" step="1" name="dxcount" value="{{form.get('dxcount','1')}}"></div>
    <div><label>Prior visits (history depth)</label><input type="number" step="1" name="n_visits" value="{{form.get('n_visits','3')}}"></div></div>
  <div style="margin-top:8px">{% for c in coms %}<label class="chk"><input type="checkbox" name="com_{{c}}" {{'checked' if form.get('com_'+c)}}> {{c}}</label>{% endfor %}</div>
  <button type="submit">Predict</button>
</div></form>
<p class="note">Model: LightGBM (180 features), trained on the temporal, patient-disjoint split. Top driver is the baseline HbA1c value.</p>
</div></body></html>
"""

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    form = request.form if request.method == "POST" else {}
    if request.method == "POST":
        vec, unknown = build_vector(request.form)
        proba = MODEL.predict(vec)[0]
        order = sorted(zip(CLASSES, proba), key=lambda x: -x[1])
        label = order[0][0]
        hb = float(request.form["baseline_hba1c"])
        direction = {"improved": "to DECREASE (improve)", "worsened": "to INCREASE (worsen)",
                     "stable": "to stay about the same"}[label]
        msg = f"From a baseline of {hb:g}%, the model expects the next HbA1c {direction}."
        if label == "improved" and hb >= 9:
            msg += " Note: a high baseline usually predicts improvement (regression to the mean); the patient may still remain above target."
        result = {"label": label, "color": COLORS[label],
                  "probs": order, "colors": COLORS, "unknown": unknown, "message": msg}
    return render_template_string(PAGE, result=result, form=form,
                                  meds=MED_CLASSES, coms=COMORBIDITIES)

if __name__ == "__main__":
    print("Serving on http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
