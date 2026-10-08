# Activation Patching & Logit Lens on GPT-2 Small

A small mechanistic-interpretability notebook built with [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens). It walks through caching activations, patching a layer's residual stream, reading the residual stream with the logit lens, and attributing a token's logit to individual attention heads.

## What's in the notebook

`notebooks/Activation_Patching_Layer_Specific.ipynb` uses the prompt **"My dog is a good"** on `gpt2-small` and covers:

1. **Clean / corrupted caches** – `run_with_cache` to record activations and the logit of a target token.
2. **Layer-specific patching** – overwrite `blocks.3.hook_resid_pre` in a run with the cached activation from another run.
3. **Logit lens** – apply `ln_final` and `W_U` to intermediate residual streams to see what the model "would predict" at each layer (top-10 at layer 3, plus per-layer trajectories for chosen tokens).
4. **Direct head attribution** – project each layer-10 head's output (`hook_z @ W_O`) onto the `' dog'` unembedding direction.

### Results recorded in the notebook

| Quantity | Value |
|---|---|
| Clean `' dog'` logit | 14.10 |
| Patched `' dog'` logit (layer 3) | 14.10 |
| Layer-10 head 10.0 → `' dog'` | +2.70 |
| Layer-10 head 10.7 → `' dog'` | −0.83 |
| Layer-10 other heads | roughly −0.04 to +0.29 |

At layer 3 the top logit-lens tokens are things like `'enough'`, `' luck'` and `' friend'`; `' dog'` only becomes a top candidate in later layers.

## Known limitations

- **The patching demo is a no-op.** The clean and corrupted prompts are both `"My dog is a good"`, so patching a layer from one run into the other changes nothing (the patched logit equals the clean logit, 14.10). The "corrupted" number in the notebook differs only because it reads the `' friend'` logit instead of `' dog'`. For a real patching experiment the two prompts must differ (e.g. `"My dog is a good"` vs `"My cat is a good"`).
- `src/patching_sweep.py` is a draft that does this properly, with a layer × position sweep. It has **not been run**; verify its output before drawing conclusions.
- Head attribution is a direct-effect approximation; it ignores indirect effects through later layers and the layer-norm mean-centering.
- Single prompt, single model: treat the numbers as an illustration, not a finding.

## Getting started

```bash
git clone https://github.com/<your-username>/activation-patching-gpt2.git
cd activation-patching-gpt2
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/Activation_Patching_Layer_Specific.ipynb
```

A GPU helps but isn't required; GPT-2 small runs fine on CPU.

To try the sweep script:

```bash
python src/patching_sweep.py --clean "My dog is a good" --corrupted "My cat is a good" \
    --answer " dog" --wrong " cat"
```

## Repo layout

```
.
├── notebooks/Activation_Patching_Layer_Specific.ipynb
├── src/patching_sweep.py
├── requirements.txt
├── LICENSE
└── README.md
```

## References

- Neel Nanda et al., [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens)
- nostalgebraist, [interpreting GPT: the logit lens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens)
- Meng et al., [Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262) (causal tracing / activation patching)

## License

MIT – see [LICENSE](LICENSE). Replace `<YOUR NAME>` with your name.
