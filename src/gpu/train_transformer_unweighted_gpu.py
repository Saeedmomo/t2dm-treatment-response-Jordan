"""V7 revision, item 4: the primary BEHRT-style transformer fine-tuned WITHOUT class weights.

Everything else is identical to train_transformer_gpu.py: same prepped arrays, same 96-token window,
same MLM-pretrained encoder checkpoint, same optimiser, epochs, batch size and seeds 0/1/2. Only the
loss changes from inverse-frequency weighted cross-entropy to unweighted cross-entropy.

Run from GPU_trans/code on the RTX 5060 (about 10 minutes):
    python train_transformer_unweighted_gpu.py
Writes GPU_trans/preds/transformer_unweighted__temporal_{validation,test}__seed{0,1,2}.parquet,
which revision/evaluate_runs.py then scores under the common protocol.
"""
import os, sys, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_transformer_gpu as T
from common_gpu import save_preds, CPU_RES, RES

def main(seeds=(0, 1, 2), max_len=96, batch=256, epochs=2):
    assert torch.cuda.is_available(), "CUDA GPU required"
    ids, mask, lab, meta = T.get_arrays()
    ids, mask = T.retruncate(ids, mask, max_len)
    L = T.loaders_for(meta, ids, mask, lab, "split_temporal", batch)
    pre_path = os.path.join(CPU_RES, "transformer_pretrained_bert.pt")
    if not os.path.exists(pre_path):
        pre_path = os.path.join(RES, "transformer_pretrained_bert.pt")
    pre = torch.load(pre_path, map_location="cpu")
    ones = np.ones(3)                      # unweighted cross-entropy
    for s in seeds:
        t0 = time.time()
        model = T.finetune(L, ones, s, pretrained_bert=pre, epochs=epochs, tag="ft_unweighted_temporal")
        for name in ("validation", "test"):
            save_preds("transformer_unweighted", f"temporal_{name}", meta[L[name][1]],
                       T.predict(model, L[name][0]), seed=s)
        print(f"seed {s} done in {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
