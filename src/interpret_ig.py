"""
Interpretability: Layer Integrated Gradients (captum) on the trained transformer (seed-0 model
saved by train_transformer.py). Produces:
  - global token importance for the 'worsened' class (mean |attribution| per vocab token)
  - a few local example attributions
  - a faithfulness check: deletion curve (remove top-k attributed tokens, watch prob drop)
Outputs to Create_results/ (tables) and Create_figures/ (plots).
"""
import os, sys, json, numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from transformers import BertForSequenceClassification
from captum.attr import LayerIntegratedGradients
sys.path.insert(0, os.path.dirname(__file__))
from common import load_meta, RES, FIG, PREP

torch.set_num_threads(min(24, os.cpu_count()))
DEVICE = "cpu"
MODEL_DIR = os.path.join(RES, "transformer_model")
VOCAB = json.load(open(os.path.join(r"D:\Hakeem_compitition\new_code","Example_transformer_model","vocab.json")))
ID2TOK = {v:k for k,v in VOCAB.items()}
PAD, CLS = 0, 2
WORS = 2  # 'worsened' class index

model = BertForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE).eval()
emb_layer = model.bert.embeddings.word_embeddings

def fwd(input_ids, attention_mask):
    return model(input_ids=input_ids, attention_mask=attention_mask).logits

lig = LayerIntegratedGradients(fwd, emb_layer)

# test rows (seq-96 arrays via same retruncation as training)
ids = np.load(os.path.join(PREP,"input_ids.npy")); mask = np.load(os.path.join(PREP,"attn_mask.npy"))
meta = load_meta()
def retrunc(ids, mask, ML=96):
    reallen = mask.sum(1); out_i=np.zeros((len(ids),ML),ids.dtype); out_m=np.zeros((len(ids),ML),mask.dtype)
    short=reallen<=ML; out_i[short]=ids[short,:ML]; out_m[short]=mask[short,:ML]
    for i in np.where(~short)[0]:
        rl=reallen[i]; out_i[i,0]=ids[i,0]; out_i[i,1:ML]=ids[i,rl-(ML-1):rl]; out_m[i,:]=1
    return out_i,out_m
ids, mask = retrunc(ids, mask)
ids = ids.astype(np.int64); mask = mask.astype(np.int64)   # embedding needs int64/long
test_mask = (meta["split_temporal"]=="test").to_numpy()
ids_t, mask_t, lab_t = ids[test_mask], mask[test_mask], meta.loc[test_mask,"label_id"].to_numpy()

rng = np.random.default_rng(0)
sample = rng.choice(len(ids_t), size=min(1500, len(ids_t)), replace=False)  # sample for tractable IG

def attribute(x_ids, x_mask, target):
    x = torch.tensor(x_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
    a = torch.tensor(x_mask, dtype=torch.long, device=DEVICE).unsqueeze(0)
    ref = torch.full_like(x, PAD); ref[0,0] = CLS   # baseline: all PAD except [CLS]
    att = lig.attribute(inputs=x, baselines=ref, additional_forward_args=(a,),
                        target=int(target), n_steps=32, internal_batch_size=8)
    att = att.sum(dim=-1).squeeze(0).detach().numpy()  # per-token attribution
    return att

# ---- global token importance for 'worsened' ----
agg_abs = {}; agg_signed = {}; counts = {}
for si in sample:
    att = attribute(ids_t[si], mask_t[si], WORS)
    L = int(mask_t[si].sum())
    for pos in range(L):
        tid = int(ids_t[si][pos])
        if tid in (PAD, CLS): continue
        tok = ID2TOK.get(tid, str(tid))
        agg_abs[tok] = agg_abs.get(tok,0.0) + abs(att[pos])
        agg_signed[tok] = agg_signed.get(tok,0.0) + att[pos]
        counts[tok] = counts.get(tok,0) + 1

glob = pd.DataFrame({"token": list(agg_abs),
                     "mean_abs_attr": [agg_abs[t]/counts[t] for t in agg_abs],
                     "mean_signed_attr": [agg_signed[t]/counts[t] for t in agg_abs],
                     "n_occ": [counts[t] for t in agg_abs]})
glob = glob[glob["n_occ"]>=20].sort_values("mean_abs_attr", ascending=False)
glob.to_csv(os.path.join(RES,"ig_global_token_importance.csv"), index=False)
print("=== Top 20 tokens by mean |IG attribution| for 'worsened' ===")
print(glob.head(20).to_string(index=False))

# plot top-20 signed
top = glob.head(20).iloc[::-1]
plt.figure(figsize=(9,7))
colors = ["#a5573b" if v>0 else "#3b6ea5" for v in top["mean_signed_attr"]]
plt.barh(top["token"], top["mean_signed_attr"], color=colors)
plt.xlabel("mean signed IG attribution (toward 'worsened')")
plt.title("Global token importance (Layer Integrated Gradients) — 'worsened'")
plt.tight_layout(); plt.savefig(os.path.join(FIG,"ig_global_importance.png"), dpi=140); plt.close()

# ---- local examples (one per outcome, correctly-and-confidently predicted) ----
with torch.no_grad():
    probs=[]
    for i in range(0,len(ids_t),256):
        x=torch.tensor(ids_t[i:i+256],dtype=torch.long); a=torch.tensor(mask_t[i:i+256],dtype=torch.long)
        probs.append(torch.softmax(model(input_ids=x,attention_mask=a).logits,1).numpy())
    probs=np.concatenate(probs,0)
local_rows=[]
for cls in [0,1,2]:
    cand=np.where((lab_t==cls)&(probs.argmax(1)==cls))[0]
    if len(cand)==0: continue
    si=cand[np.argmax(probs[cand,cls])]
    att=attribute(ids_t[si],mask_t[si],cls); L=int(mask_t[si].sum())
    toks=[ID2TOK.get(int(ids_t[si][p]),"?") for p in range(L)]
    order=np.argsort(-np.abs(att[:L]))
    top_toks=[(toks[p], round(float(att[p]),3)) for p in order[:8] if toks[p] not in ("[CLS]",)]
    local_rows.append({"true_class": ["improved","stable","worsened"][cls],
                       "pred_prob": round(float(probs[si,cls]),3),
                       "top_tokens": "; ".join(f"{t}({v:+.2f})" for t,v in top_toks)})
pd.DataFrame(local_rows).to_csv(os.path.join(RES,"ig_local_examples.csv"), index=False)
print("\n=== Local examples ===")
print(pd.DataFrame(local_rows).to_string(index=False))

# ---- faithfulness: deletion curve (remove top-k IG tokens -> worsened prob) ----
del_sample = rng.choice(len(ids_t), size=300, replace=False)
ks=[0,1,2,3,5,8,12]
curve=np.zeros(len(ks))
for si in del_sample:
    att=attribute(ids_t[si],mask_t[si],WORS); L=int(mask_t[si].sum())
    order=[p for p in np.argsort(-att[:L]) if ids_t[si][p] not in (PAD,CLS)]
    x=ids_t[si].copy(); m=mask_t[si].copy()
    for ki,k in enumerate(ks):
        xx=ids_t[si].copy(); mm=mask_t[si].copy()
        for p in order[:k]:
            mm[p]=0; xx[p]=PAD  # mask out top-k attributed tokens
        with torch.no_grad():
            pr=torch.softmax(model(input_ids=torch.tensor(xx,dtype=torch.long).unsqueeze(0),
                                   attention_mask=torch.tensor(mm,dtype=torch.long).unsqueeze(0)).logits,1).numpy()[0,WORS]
        curve[ki]+=pr
curve/=len(del_sample)
pd.DataFrame({"k_tokens_removed":ks,"mean_worsened_prob":curve.round(4)}).to_csv(
    os.path.join(RES,"ig_deletion_faithfulness.csv"), index=False)
plt.figure(figsize=(7,5))
plt.plot(ks,curve,"o-"); plt.xlabel("top-k IG tokens removed"); plt.ylabel("mean P(worsened)")
plt.title("Faithfulness: deletion of top-attributed tokens (should drop)")
plt.tight_layout(); plt.savefig(os.path.join(FIG,"ig_deletion_faithfulness.png"),dpi=140); plt.close()
print(f"\nDeletion faithfulness: P(worsened) {curve[0]:.3f} -> {curve[-1]:.3f} after removing top-{ks[-1]} tokens")
print("interpret_ig.py DONE")
