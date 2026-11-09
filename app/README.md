# HbA1c-trajectory LightGBM demo (run it on your own computer)

An interactive app for the best model (LightGBM). You enter a patient's clinical values and it shows
the predicted probability that the next HbA1c (90-450 days ahead) will be **improved / stable /
worsened**. It runs entirely on your machine. No account, server, or internet is needed after you
download it.

---

## 1. What to download

The app is the **`app/` folder** of this repository. You need these files (all inside `app/`):

| File | What it is | Needed? |
|---|---|---|
| `app.py` | the application | required |
| `lgbm_temporal.txt` | the trained LightGBM model (~14 MB) | required |
| `requirements.txt` | the list of packages | helpful |
| `run.bat` | one-click launcher (Windows) | optional |
| `README.md` | this guide | optional |

**Easiest way - download everything as a ZIP:**
1. Go to the repository's main page: https://github.com/Saeedmomo/t2dm-treatment-response-Jordan
2. Click the green **`< > Code`** button, then **Download ZIP**.
3. Unzip the downloaded file, then open the **`app`** folder inside it.

(Advanced: you can instead download just `app/app.py` and `app/lgbm_temporal.txt` by opening each file
on GitHub and clicking the download icon.)

---

## 2. Install Python and the packages (one time)

1. Install **Python 3.9 or newer** from https://www.python.org/downloads/
   - On Windows, tick **"Add Python to PATH"** in the installer.
2. Open a terminal (**Command Prompt** or **PowerShell** on Windows, **Terminal** on Mac/Linux) in the
   `app` folder and run:
   ```
   pip install flask lightgbm numpy
   ```

---

## 3. Run it

In the same terminal, from the `app` folder:
```
python app.py
```
You will see a line like `Running on http://127.0.0.1:5000`. Open that address in your web browser.

**Windows shortcut:** just **double-click `run.bat`** - it installs the packages and starts the app for
you. Then open http://127.0.0.1:5000

To stop the app: press **Ctrl + C** in the terminal, or close the terminal window.

> Tip: the terminal window must stay open while you use the app. If you close it, the page will show
> "site can't be reached" - just start it again.

---

## 4. How to read the output

This model predicts the **direction the HbA1c will move**, not an overall risk score.
- **improved** = the next HbA1c is likely to **fall** by at least 0.5 points
- **worsened** = likely to **rise** by at least 0.5 points
- **stable** = little change

Because very high values tend to come down and near-target values have little room to fall, a **high
current HbA1c usually predicts "improved" (regression to the mean)**. "Improved" does not mean the
patient reaches target. To see the effect of risk factors, compare patients at the **same baseline
HbA1c**: with the baseline fixed, adding risk features (higher fasting glucose, insulin, comorbidities)
increases "worsened".

---

## Notes

- **Research prototype, not for clinical use.**
- The app contains **no patient data**; it only turns the values you type into a prediction.
- Vitals are treated as missing, exactly as in training (the source data had no vital-sign timestamps).
- It is a single-visit snapshot approximation of the full pipeline; the baseline HbA1c is the dominant
  input, so predictions track the trained model closely.
