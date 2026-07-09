"""Plot train/validation loss curves from a train_lora.py run.

Reads <run_dir>/log_history.json (written by train_lora.py after training);
falls back to the newest checkpoint-*/trainer_state.json if it's missing
(e.g. the run crashed before finishing). Saves <run_dir>/loss_curve.png.

  python src/plot_training.py --run runs/joint_fold0_llama
In a Colab notebook cell, display it afterwards with:
  from IPython.display import Image; Image("runs/joint_fold0_llama/loss_curve.png")
"""
import argparse
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TRAIN_COLOR = "#2a78d6"   # blue
VAL_COLOR = "#1baf7a"     # aqua
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e5e4e0"


def load_history(run_dir):
    p = run_dir / "log_history.json"
    if p.exists():
        return json.load(open(p, encoding="utf-8")), str(p)
    # fallback: newest checkpoint's trainer_state.json
    states = sorted(glob.glob(str(run_dir / "checkpoint-*" / "trainer_state.json")),
                    key=lambda s: int(Path(s).parent.name.split("-")[-1]))
    if states:
        return json.load(open(states[-1], encoding="utf-8"))["log_history"], states[-1]
    raise SystemExit(f"no log_history.json or checkpoint-*/trainer_state.json "
                     f"under {run_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="output_dir of a train_lora.py run")
    ap.add_argument("--out", default=None, help="png path (default <run>/loss_curve.png)")
    args = ap.parse_args()
    run_dir = Path(args.run)
    history, src = load_history(run_dir)

    train = [(h["step"], h["loss"]) for h in history if "loss" in h]
    val = [(h["step"], h["eval_loss"]) for h in history if "eval_loss" in h]
    if not train:
        raise SystemExit(f"no training loss entries found in {src}")

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    ax.plot(*zip(*train), color=TRAIN_COLOR, linewidth=2, label="train loss")
    if val:
        ax.plot(*zip(*val), color=VAL_COLOR, linewidth=2, marker="o",
                markersize=4, label="validation loss")
        gap = val[-1][1] - min(v for _, v in val)
        note = (f"final val {val[-1][1]:.3f} (min {min(v for _, v in val):.3f})"
                + ("  — rising: consider fewer epochs" if gap > 0.02 else ""))
    else:
        note = ("no validation entries — retrain with *_train_full.jsonl files "
                "and --eval_fold K to get a val curve")
    ax.set_xlabel("optimizer step", color=MUTED)
    ax.set_ylabel("loss", color=MUTED)
    ax.set_title(f"SFT convergence — {run_dir.name}", color=INK, loc="left")
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.tick_params(colors=MUTED)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.legend(frameon=False)
    ax.text(0.0, -0.16, note, transform=ax.transAxes, color=MUTED, fontsize=8)
    fig.tight_layout()

    out = Path(args.out or (run_dir / "loss_curve.png"))
    fig.savefig(out, bbox_inches="tight", facecolor="#fcfcfb")
    print(f"plot -> {out}")
    if val:
        print(f"train loss: {train[0][1]:.3f} -> {train[-1][1]:.3f}   "
              f"val loss: {val[0][1]:.3f} -> {val[-1][1]:.3f} (min {min(v for _, v in val):.3f})")


if __name__ == "__main__":
    main()
