"""Layer x position activation-patching sweep on GPT-2 small.

Patches the clean run's residual stream (blocks.{L}.hook_resid_pre) into a
corrupted run, one (layer, position) at a time, and reports how much of the
clean-vs-corrupted logit difference is restored.

NOTE: untested draft - run it and sanity-check the output before relying on it.
"""
import argparse
import os

import matplotlib.pyplot as plt
import torch
from transformer_lens import HookedTransformer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clean", default="My dog is a good")
    p.add_argument("--corrupted", default="My cat is a good")
    p.add_argument("--answer", default=" dog", help="token the clean run should favour")
    p.add_argument("--wrong", default=" cat", help="token the corrupted run should favour")
    p.add_argument("--out", default="results/patching_sweep.png")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    model.eval()

    clean_tokens = model.to_tokens(args.clean)
    corrupted_tokens = model.to_tokens(args.corrupted)
    assert clean_tokens.shape == corrupted_tokens.shape, "Prompts must tokenize to the same length"

    ans = model.to_single_token(args.answer)
    wrong = model.to_single_token(args.wrong)

    def logit_diff(logits):
        return (logits[0, -1, ans] - logits[0, -1, wrong]).item()

    with torch.no_grad():
        clean_logits, clean_cache = model.run_with_cache(clean_tokens)
        corrupted_logits = model(corrupted_tokens)
    clean_ld, corrupted_ld = logit_diff(clean_logits), logit_diff(corrupted_logits)
    print(f"Clean logit diff: {clean_ld:.3f} | Corrupted logit diff: {corrupted_ld:.3f}")
    if abs(clean_ld - corrupted_ld) < 1e-3:
        raise SystemExit("Clean and corrupted runs give the same metric - patching can't show anything.")

    n_layers, n_pos = model.cfg.n_layers, clean_tokens.shape[1]
    scores = torch.zeros(n_layers, n_pos)

    for layer in range(n_layers):
        name = f"blocks.{layer}.hook_resid_pre"
        for pos in range(n_pos):
            def hook(resid, hook, pos=pos):
                resid[:, pos, :] = clean_cache[hook.name][:, pos, :]
                return resid

            with torch.no_grad():
                patched = model.run_with_hooks(corrupted_tokens, fwd_hooks=[(name, hook)])
            # 0 = same as corrupted run, 1 = fully restores the clean run
            scores[layer, pos] = (logit_diff(patched) - corrupted_ld) / (clean_ld - corrupted_ld)

    labels = model.to_str_tokens(clean_tokens)
    plt.figure(figsize=(7, 6))
    plt.imshow(scores.numpy(), aspect="auto", cmap="RdBu", vmin=-1, vmax=1)
    plt.colorbar(label="Fraction of clean logit diff restored")
    plt.xticks(range(n_pos), labels, rotation=45)
    plt.yticks(range(n_layers))
    plt.xlabel("Patched position")
    plt.ylabel("Layer (resid_pre)")
    plt.title("Activation patching: clean -> corrupted")
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
