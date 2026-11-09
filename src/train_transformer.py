"""
Model 1: BERT/BEHRT-style transformer, reusing vocab.json (vocab_size=175).
Architecture mirrors the reference phaseA_cpu_model config (hidden 192, 4 layers, 4 heads,
intermediate 768) -> ~2-3M params, sized for CPU / an 8GB GPU.

Pipeline:
  (a) optional MLM pretraining on the training-split sequences
  (b) supervised fine-tuning on the 3-class label with class weights
  3 seeds for the main (pretrained) model; a no-pretrain ablation; a single-seed run on the
  secondary random split for the sensitivity check.

Runs on CPU here (no CUDA present). AMP is a no-op on CPU so we train fp32; batch size is kept
modest and the model is small, which is the CPU analogue of the "fit within 8GB" instruction.
Saves 3-class probabilities (val+test) to Create_results/preds/ and the seed-0 pretrained model
(for Integrated Gradients) to Create_results/transformer_model/.
"""
import os, sys, time, json, argparse, numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertConfig, BertForMaskedLM, BertForSequenceClassification
sys.path.insert(0, os.path.dirname(__file__))
from common import load_meta, load_transformer_arrays, save_preds, RES

torch.set_num_threads(min(24, os.cpu_count()))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VOCAB = json.load(open(os.path.join(r"D:\Hakeem_compitition\new_code",
                                    "Example_transformer_model", "vocab.json")))
SPECIAL_NO_MASK = {0,2,3,5,6}   # [PAD],[CLS],[SEP],[EV],[CTX] are structural -> not MLM targets
MASK_ID = 4                     # [MASK]

def make_config():
    return BertConfig(vocab_size=len(VOCAB), hidden_size=192, num_hidden_layers=4,
                      num_attention_heads=4, intermediate_size=768, max_position_embeddings=512,
                      type_vocab_size=2, pad_token_id=0, problem_type="single_label_classification",
                      num_labels=3)

def set_seed(s):
    np.random.seed(s); torch.manual_seed(s)

def get_arrays():
    ids, mask, lab = load_transformer_arrays()
    meta = load_meta()
    return ids, mask, lab, meta

def retruncate(ids, mask, ML):
    """Shrink canonical MAX_LEN arrays to ML, keeping [CLS] + most-recent tail (recency matters)."""
    if ids.shape[1] <= ML:
        return ids, mask
    reallen = mask.sum(1)
    out_i = np.zeros((len(ids), ML), dtype=ids.dtype)
    out_m = np.zeros((len(ids), ML), dtype=mask.dtype)
    short = reallen <= ML
    out_i[short] = ids[short, :ML]; out_m[short] = mask[short, :ML]
    for i in np.where(~short)[0]:
        rl = reallen[i]
        out_i[i, 0] = ids[i, 0]
        out_i[i, 1:ML] = ids[i, rl-(ML-1):rl]
        out_m[i, :] = 1
    return out_i, out_m

def loaders_for(meta, ids, mask, lab, split_col, batch=128):
    out = {}
    for name in ["train","validation","test"]:
        m = (meta[split_col]==name).to_numpy()
        ds = TensorDataset(torch.tensor(ids[m], dtype=torch.long),
                           torch.tensor(mask[m], dtype=torch.long),
                           torch.tensor(lab[m], dtype=torch.long))
        out[name] = (DataLoader(ds, batch_size=batch, shuffle=(name=="train")), m)
    return out

# ---------------- MLM pretraining ----------------
def mlm_pretrain(train_loader, epochs=2, lr=5e-4, log_every=400):
    cfg = make_config()
    model = BertForMaskedLM(cfg).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()
    for ep in range(epochs):
        t0=time.time(); tot=0.0; nb=0
        for bi,(x, attn, _lab) in enumerate(train_loader):
            x=x.to(DEVICE); attn=attn.to(DEVICE)
            labels = x.clone()
            # candidate positions: attended AND not structural specials
            cand = attn.bool()
            for sid in SPECIAL_NO_MASK: cand &= (x!=sid)
            prob = torch.full(x.shape, 0.15, device=DEVICE)
            masked = torch.bernoulli(prob).bool() & cand
            labels[~masked] = -100
            # 80% MASK, 10% random, 10% keep
            r = torch.rand(x.shape, device=DEVICE)
            x[masked & (r<0.8)] = MASK_ID
            rand_tok = torch.randint(7, len(VOCAB), x.shape, device=DEVICE)
            take_rand = masked & (r>=0.8) & (r<0.9)
            x[take_rand] = rand_tok[take_rand]
            out = model(input_ids=x, attention_mask=attn, labels=labels)
            opt.zero_grad(); out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot+=out.loss.item(); nb+=1
            if (bi+1)%log_every==0:
                print(f"    MLM ep{ep+1} batch {bi+1} loss={tot/nb:.4f} ({(time.time()-t0)/(bi+1):.3f}s/b)", flush=True)
        print(f"  MLM epoch {ep+1}/{epochs} loss={tot/nb:.4f} time={time.time()-t0:.0f}s", flush=True)
    return model.bert.state_dict()

# ---------------- supervised fine-tuning ----------------
def finetune(loaders, class_weights, seed, pretrained_bert=None, epochs=2, lr=3e-4, log_every=400):
    set_seed(seed)
    cfg = make_config()
    model = BertForSequenceClassification(cfg).to(DEVICE)
    if pretrained_bert is not None:
        # MLM's bert has no pooler; strict=False loads encoder+embeddings, pooler trains during FT
        model.bert.load_state_dict(pretrained_bert, strict=False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    cw = torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)
    lossfn = nn.CrossEntropyLoss(weight=cw)
    train_loader = loaders["train"][0]
    for ep in range(epochs):
        model.train(); t0=time.time(); tot=0.0; nb=0
        for bi,(x, attn, y) in enumerate(train_loader):
            x=x.to(DEVICE); attn=attn.to(DEVICE); y=y.to(DEVICE)
            logits = model(input_ids=x, attention_mask=attn).logits
            loss = lossfn(logits, y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot+=loss.item(); nb+=1
            if (bi+1)%log_every==0:
                print(f"    FT seed{seed} ep{ep+1} batch {bi+1} loss={tot/nb:.4f} ({(time.time()-t0)/(bi+1):.3f}s/b)", flush=True)
        print(f"  FT seed{seed} epoch {ep+1}/{epochs} loss={tot/nb:.4f} time={time.time()-t0:.0f}s", flush=True)
    return model

@torch.no_grad()
def predict(model, loader):
    model.eval(); ps=[]
    for x, attn, y in loader:
        logits = model(input_ids=x.to(DEVICE), attention_mask=attn.to(DEVICE)).logits
        ps.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(ps, 0)

def class_weights_from(lab, mask):
    y = lab[mask]; counts = np.bincount(y, minlength=3).astype(float)
    w = counts.sum()/(3*counts)      # inverse-frequency, mean ~1
    return w

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs_ft", type=int, default=2)
    ap.add_argument("--epochs_mlm", type=int, default=2)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0,1,2])
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--max_len", type=int, default=96)
    ap.add_argument("--ablation_seeds", type=int, nargs="+", default=[0])
    args = ap.parse_args()

    ids, mask, lab, meta = get_arrays()
    ids, mask = retruncate(ids, mask, args.max_len)
    print(f"seq len={ids.shape[1]}, batch={args.batch}, device={DEVICE}", flush=True)

    # ---------- PRIMARY temporal split ----------
    L = loaders_for(meta, ids, mask, lab, "split_temporal", args.batch)
    cw = class_weights_from(lab, L["train"][1])
    print(f"class weights (temporal train): {cw.round(3)}", flush=True)

    pre_path = os.path.join(RES, "transformer_pretrained_bert.pt")
    if os.path.exists(pre_path):
        print("=== loading existing MLM-pretrained bert (temporal) ===", flush=True)
        pre = torch.load(pre_path)
    else:
        print("=== MLM pretraining (temporal train) ===", flush=True)
        t0=time.time()
        pre = mlm_pretrain(L["train"][0], epochs=args.epochs_mlm)
        print(f"MLM pretrain total {time.time()-t0:.0f}s", flush=True)
        torch.save(pre, pre_path)

    # main pretrained finetune, 3 seeds
    for s in args.seeds:
        print(f"=== finetune (pretrained) seed {s} ===", flush=True)
        model = finetune(L, cw, s, pretrained_bert=pre, epochs=args.epochs_ft)
        save_preds("transformer", "temporal_validation", meta[L["validation"][1]], predict(model, L["validation"][0]), seed=s)
        save_preds("transformer", "temporal_test",       meta[L["test"][1]],       predict(model, L["test"][0]),       seed=s)
        if s == args.seeds[0]:
            os.makedirs(os.path.join(RES, "transformer_model"), exist_ok=True)
            model.save_pretrained(os.path.join(RES, "transformer_model"))

    # ablation: no-pretrain finetune
    for s in args.ablation_seeds:
        print(f"=== finetune (NO pretrain) seed {s} ===", flush=True)
        model = finetune(L, cw, s, pretrained_bert=None, epochs=args.epochs_ft)
        save_preds("transformer_nopretrain", "temporal_validation", meta[L["validation"][1]], predict(model, L["validation"][0]), seed=s)
        save_preds("transformer_nopretrain", "temporal_test",       meta[L["test"][1]],       predict(model, L["test"][0]),       seed=s)

    # ---------- SECONDARY random split (sensitivity, single seed) ----------
    Lr = loaders_for(meta, ids, mask, lab, "split_random", args.batch)
    cwr = class_weights_from(lab, Lr["train"][1])
    prer_path = os.path.join(RES, "transformer_pretrained_bert_random.pt")
    if os.path.exists(prer_path):
        print("=== loading existing MLM-pretrained bert (random) ===", flush=True)
        prer = torch.load(prer_path)
    else:
        print("=== MLM pretraining (random train) ===", flush=True)
        prer = mlm_pretrain(Lr["train"][0], epochs=args.epochs_mlm)
        torch.save(prer, prer_path)
    print("=== finetune (pretrained) random seed 0 ===", flush=True)
    modelr = finetune(Lr, cwr, 0, pretrained_bert=prer, epochs=args.epochs_ft)
    save_preds("transformer", "random_validation", meta[Lr["validation"][1]], predict(modelr, Lr["validation"][0]), seed=0)
    save_preds("transformer", "random_test",       meta[Lr["test"][1]],       predict(modelr, Lr["test"][0]),       seed=0)

    print("DONE transformer.", flush=True)

if __name__ == "__main__":
    main()
