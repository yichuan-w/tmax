"""Teaser figure: Terminal-Bench 2.0 performance vs (rough) model size.

The figure is the paper's opener: a scatter where the *y*-axis is TB2.0 score
and the *x*-axis is model size — but the size axis is deliberately **not** a
plain log scale.  We have many models bunched at 4B / 8B / 32B and then a big
empty gap up to the 200–350B frontier models (MiniMax, GLM) and the
closed-source models whose size is unknown (Claude, GPT-5).  A literal log
axis wastes the middle and crushes the small-model region, so instead we use a
*piecewise-linear-in-log* transform with hand-placed knots (see ``X_KNOTS``)
that gives each "tier" roughly equal horizontal real-estate, and parks the
unknown-size models at a dedicated ``?`` position on the far right.

Individual data points are drawn either as a brand **logo** (PNG in
``scripts/plot/media``) or, for research baselines, as a simple coloured dot.
Every logo is normalised to (roughly) the same on-figure size, with a per-model
scale knob for fine-tuning, plus a small text annotation naming the model.

Run::

    uv run python scripts/plot/teaser_terminal_bench.py
    uv run python scripts/plot/teaser_terminal_bench.py --csv scripts/plot/terminal_bench_results.csv

Styling intentionally mirrors ``rl_data/comparison/styles.py`` (DejaVu Serif,
no top/right spines, light-grey axes, Anthropic-warm palette) so the teaser
reads as part of the same figure family as the rest of the paper.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import importlib.util

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image

logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
MEDIA = HERE / "media"


def _load_styles_module():
    """Load ``rl_data/comparison/styles.py`` *directly by file path*.

    We deliberately avoid ``from rl_data.comparison.styles import ...`` because
    importing it as a package runs ``rl_data/__init__.py``, which imports
    ``litellm`` (≈4s cold).  ``styles.py`` itself only uses the stdlib, so
    loading the file in isolation keeps it as the single source of truth for
    the paper's palette while making this script start near-instantly.
    """
    styles_path = HERE.parent.parent / "rl_data" / "comparison" / "styles.py"
    spec = importlib.util.spec_from_file_location("_tmax_paper_styles", styles_path)
    module = importlib.util.module_from_spec(spec)
    # Register before exec so dataclasses in styles.py can resolve their own
    # module via sys.modules[cls.__module__] during class creation.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# Reuse the paper's colour helpers so this figure matches the others.
darken_hex = _load_styles_module().darken_hex
DEFAULT_CSV = HERE / "terminal_bench_results.csv"
DEFAULT_OUT = HERE / "output" / "teaser_terminal_bench"


# ===========================================================================
# 1. X-AXIS TRANSFORM  (the "untraditional" part)
# ===========================================================================
#
# We map a model's size (in billions of params) to a horizontal plot position
# via piecewise-linear interpolation **in log10(size)** through these knots.
# Each consecutive knot pair is given one full unit of horizontal space, so the
# 4B->8B, 8B->32B and 32B->(200-350B) jumps all *look* equally wide even though
# they span very different multiplicative ranges.  Tweak these to re-balance.
#
#                       size(B)  ->  x-position
X_KNOTS: List[Tuple[float, float]] = [
    (4.0, 0.0),
    (8.0, 1.0),
    (32.0, 2.0),
    (256.0, 3.0),    # representative anchor for the 200-350B "very large" tier
    (1000.0, 4.0),   # ~1T total-param tier (e.g. MAI-Thinking, 35B active)
]
# Models with no public size (Claude, GPT-5, ...) are parked here, on the right.
UNKNOWN_X: float = 5.0

# Where to draw x ticks and what to call them.
X_TICKS: List[Tuple[float, str]] = [
    (0.0, "4B"),
    (1.0, "8B"),
    (2.0, "32B"),
    (3.0, "200–350B"),
    (4.0, "~1T"),
    (UNKNOWN_X, "?"),
]
X_LIM: Tuple[float, float] = (-0.45, 5.45)


def size_to_x(size: Optional[float]) -> float:
    """Map a size in billions of params to an x position (see ``X_KNOTS``).

    ``None`` / NaN (size undisclosed) maps to :data:`UNKNOWN_X`.  Sizes outside
    the knot range are linearly extrapolated (in log space) from the nearest
    segment, so e.g. GLM-4.7 (357B) lands just to the right of the 256B knot.
    """
    if size is None or (isinstance(size, float) and np.isnan(size)):
        return UNKNOWN_X
    xs = np.log10([k[0] for k in X_KNOTS])
    ys = [k[1] for k in X_KNOTS]
    lx = float(np.log10(size))
    if lx <= xs[0]:
        return ys[0] + (lx - xs[0]) * (ys[1] - ys[0]) / (xs[1] - xs[0])
    if lx >= xs[-1]:
        return ys[-1] + (lx - xs[-1]) * (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
    return float(np.interp(lx, xs, ys))


# ===========================================================================
# 2. PER-MODEL DISPLAY CONFIG
# ===========================================================================
#
# Keyed by the exact ``model_name`` in the CSV.  Every field is optional:
#
#   logo        : filename in scripts/plot/media (PNG). None -> coloured dot.
#   logo_scale  : per-model multiplier on the (otherwise uniform) logo size.
#   label       : annotation text (defaults to the model name).
#   dx, dy      : annotation offset from the point, in *points* (typographic).
#   ha, va      : annotation horizontal / vertical alignment.
#   x_offset    : nudge the point horizontally, in x *data* units.  Handy for
#                 fanning out models that share a size (and so an x position),
#                 or for separating the undisclosed-size models in the "?" zone.
#   ours        : True -> emphasised (bold, dark label, optional halo).
#   color       : dot colour override (only used when logo is None).
#   show_label  : set False to hide the text label for this point.
#
# Anything not listed falls back to DEFAULTS below.

OURS_LOGO = "Tmax-Logo.png"

MODEL_STYLE: Dict[str, dict] = {
    # ---- ours (TMax) ------------------------------------------------------
    "TMax-9b": dict(logo=OURS_LOGO, logo_scale=1.5, ours=True, label="TMax-9B (Ours)",
                    dx=0, dy=25, ha="center", va="bottom"),
    "Qwen3.6-27B-TMax": dict(logo=OURS_LOGO, ours=True, label="TMax-27B (Ours)",
                             dx=0, dy=15, ha="center", va="bottom"),
    "TMax3.5-4b": dict(logo=OURS_LOGO, ours=True, label="TMax-4B (Ours)",
                       dx=10, dy=10, ha="left", va="bottom"),
    # ---- Qwen base models -------------------------------------------------
    "Qwen3-8b": dict(logo="Qwen_logo.png", label="Qwen3-8B", x_offset=-0.02,
                     dx=0, dy=-24, ha="center", va="top"),
    "Qwen3.5-9b": dict(logo="Qwen_logo.png", label="Qwen3.5-9B",
                       dx=-12, dy=6, ha="right", va="bottom"),
    "Qwen3.6-27B": dict(logo="Qwen_logo.png", label="Qwen3.6-27B",
                        dx=0, dy=15, ha="center", va="bottom"),
    "Qwen3.5-4b": dict(logo="Qwen_logo.png", label="Qwen3.5-4B",
                       dx=0, dy=-24, ha="center", va="top"),
    # ---- closed / frontier ------------------------------------------------
    "Claude Haiku": dict(logo="Claude_AI_logo.png", logo_scale=0.9,
                         label="Claude Haiku", x_offset=-0.27,
                         dx=0, dy=-24, ha="center", va="top"),
    "Gpt5-Mini": dict(logo="openai.png", logo_scale=0.8, label="GPT-5-Mini",
                      x_offset=0.14, dx=0, dy=20, ha="center", va="bottom"),
    "MiniMax-M2.1": dict(logo="minimax-logo.png", label="MiniMax-M2.1 (229B)",
                         dx=0, dy=-24, ha="center", va="top"),
    "MiniMax-M2.7": dict(logo="minimax-logo.png", label="MiniMax-M2.7 (229B)",
                         dx=0, dy=20, ha="center", va="bottom"),
    "GLM-4.7": dict(logo="GLM-Zai-Logo.png", logo_scale=0.8,
                    label="GLM-4.7 (357B)", dx=0, dy=16, ha="center", va="bottom"),
    "MAI-Thinking-V1": dict(logo="Microsoft_AI_Logo.png", logo_scale=1.0,
                            label="MAI-Thinking-V1\n(35B active, ~1T total)",
                            dx=0, dy=-20, ha="center", va="top"),
    # ---- research baselines (plain dots) ----------------------------------
    "Nemotron-Terminal-8B": dict(label="Nemotron-Terminal-8B", x_offset=0.06,
                                 dx=8, dy=0, ha="left", va="center"),
    "Endless-Terminal-8B (OT-SFT)": dict(label="Endless-Terminal-8B",
                                         x_offset=0.16,
                                         dx=8, dy=2, ha="left", va="center"),
    "OpenThoughts-Agent-V1-8B": dict(label="OpenThoughts-Agent-8B",
                                     x_offset=-0.14,
                                     dx=-8, dy=0, ha="right", va="center"),
    "TerminalTraj-7B": dict(label="TerminalTraj-7B",
                            dx=-8, dy=0, ha="right", va="center"),
    "TerminalTraj-14B": dict(label="TerminalTraj-14B",
                             dx=8, dy=2, ha="left", va="center"),
    "TerminalTraj-32B": dict(label="TerminalTraj-32B", x_offset=0.1,
                             dx=8, dy=-2, ha="left", va="center"),
    "TermiGen-Qwen3-32B": dict(label="TermiGen-Qwen3-32B"),
    "TermiGen-Qwen2.5-Coder-32B": dict(label="TermiGen-Qwen2.5-32B"),
}

DEFAULTS = dict(
    logo=None,
    logo_scale=1.0,
    label=None,
    dx=8.0,
    dy=0.0,
    x_offset=0.0,
    ha="left",
    va="center",
    ours=False,
    color=None,
    show_label=True,
)


def style_for(model: str) -> dict:
    """Return the merged display config for ``model`` (defaults + overrides)."""
    cfg = dict(DEFAULTS)
    cfg.update(MODEL_STYLE.get(model, {}))
    if cfg["label"] is None:
        cfg["label"] = model
    return cfg


# ===========================================================================
# 3. FIGURE STYLE
# ===========================================================================


@dataclass
class TeaserStyle:
    """All knobs for the teaser figure (mirrors the paper's style family)."""

    # Typography
    font_family: str = "serif"
    font_serif: Sequence[str] = field(default_factory=lambda: ["DejaVu Serif"])
    font_size: float = 14.0
    title_size: float = 21.0
    axes_label_size: float = 16.0
    tick_size: float = 14.0
    annotation_size: float = 11.5
    annotation_size_ours: float = 13.0

    # Layout / spines
    figsize: Tuple[float, float] = (13.5, 7.0)
    spine_color: str = "#808080"
    spine_linewidth: float = 1.25

    # Grid
    grid: bool = True
    grid_alpha: float = 0.18
    grid_linestyle: str = "--"

    # Title / axis labels
    title: str = "Terminal-Bench 2.0: performance vs. model size"
    title_pad: float = 18.0
    title_weight: str = "bold"
    xlabel: str = "Model size (billions of parameters)"
    ylabel: str = "Terminal-Bench 2.0 (%)"
    ylim: Tuple[float, float] = (0.0, 52.0)

    # Logos — normalised so each occupies ~the same bounding box.
    #   logo_target_in : target *max* dimension of every logo, in inches.
    logo_target_in: float = 0.52

    # Dots (research baselines without a logo)
    palette_name: str = "anthropic_book"
    dot_size: float = 90.0
    dot_color: str = "#8A7B6B"          # muted warm grey
    dot_edge_color: str = "white"
    dot_edge_width: float = 1.1

    # Emphasis for "ours"
    ours_color: str = "#C15F3C"         # terracotta (anthropic_book #1)
    ours_halo: bool = True
    ours_halo_scale: float = 1.55       # halo radius relative to logo box
    ours_halo_alpha: float = 0.16
    annotation_color: str = "#5A5148"   # muted text for non-ours labels

    # Right-hand "frontier / large" band
    frontier_band: bool = True
    frontier_x: float = 2.58            # left edge of the shaded band
    frontier_color: str = "#C9BBA9"
    frontier_alpha: float = 0.14
    frontier_label: str = "Very large  /  undisclosed size"

    # Output
    dpi: int = 220
    save_pdf: bool = True


def _apply_rc(style: TeaserStyle) -> None:
    plt.rcParams.update(
        {
            "font.family": style.font_family,
            "font.serif": list(style.font_serif),
            "font.size": style.font_size,
            "axes.titlesize": style.title_size,
            "axes.labelsize": style.axes_label_size,
            "xtick.labelsize": style.tick_size,
            "ytick.labelsize": style.tick_size,
            "svg.fonttype": "none",
        }
    )


# ===========================================================================
# 4. DATA LOADING
# ===========================================================================


@dataclass
class Row:
    model: str
    size_b: Optional[float]
    tb2: Optional[float]
    x: float


def load_rows(csv_path: Path) -> Tuple[List[Row], List[str]]:
    """Load plottable rows; return ``(rows, skipped)`` (skipped = no TB2 yet)."""
    rows: List[Row] = []
    skipped: List[str] = []
    with csv_path.open(newline="") as f:
        for r in csv.DictReader(f):
            model = (r.get("model_name") or "").strip()
            if not model:
                continue
            tb2_raw = (r.get("tb2") or "").strip()
            if not tb2_raw:                      # "NEED EVAL" -> not ready yet
                skipped.append(model)
                continue
            size_raw = (r.get("size_b") or "").strip()
            size = float(size_raw) if size_raw else None
            tb2 = float(tb2_raw)
            rows.append(Row(model=model, size_b=size, tb2=tb2, x=size_to_x(size)))
    return rows, skipped


# ===========================================================================
# 5. LOGO HANDLING
# ===========================================================================

_LOGO_CACHE: Dict[str, np.ndarray] = {}


def _load_logo(name: str) -> np.ndarray:
    if name not in _LOGO_CACHE:
        path = MEDIA / name
        _LOGO_CACHE[name] = np.asarray(Image.open(path).convert("RGBA"))
    return _LOGO_CACHE[name]


def _logo_zoom(img: np.ndarray, style: TeaserStyle, scale: float) -> float:
    """Zoom so the logo's *max* dimension equals ``logo_target_in`` inches.

    OffsetImage renders ``zoom * native_pixels`` points, so a constant target
    in inches is ``target_in * 72 / max(h, w)``.  This makes every logo share
    the same bounding box regardless of source resolution; ``scale`` is the
    per-model fine-tune multiplier.
    """
    h, w = img.shape[:2]
    return (style.logo_target_in * 72.0 / max(h, w)) * scale


# ===========================================================================
# 6. RENDER
# ===========================================================================


def render(rows: List[Row], style: TeaserStyle, out_base: Path) -> None:
    _apply_rc(style)
    fig, ax = plt.subplots(figsize=style.figsize)

    # --- frontier band (drawn first, behind everything) --------------------
    if style.frontier_band:
        ax.axvspan(style.frontier_x, X_LIM[1], color=style.frontier_color,
                   alpha=style.frontier_alpha, zorder=0, linewidth=0)
        ax.text(
            (style.frontier_x + X_LIM[1]) / 2.0, style.ylim[1] - 1.4,
            style.frontier_label, ha="center", va="top",
            fontsize=style.annotation_size - 0.5, color="#9A8B79",
            style="italic", zorder=1,
        )

    # --- grid / spines -----------------------------------------------------
    if style.grid:
        ax.grid(True, axis="both", alpha=style.grid_alpha,
                linestyle=style.grid_linestyle, color=style.spine_color, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(style.spine_color)
        ax.spines[side].set_linewidth(style.spine_linewidth)

    # --- points ------------------------------------------------------------
    for row in rows:
        cfg = style_for(row.model)
        x, y = row.x + cfg["x_offset"], row.tb2

        if cfg["logo"]:
            img = _load_logo(cfg["logo"])
            zoom = _logo_zoom(img, style, cfg["logo_scale"])

            if cfg["ours"] and style.ours_halo:
                # soft circular halo behind our models so they pop
                box = style.logo_target_in * cfg["logo_scale"]
                ax.scatter(
                    [x], [y], s=(box * style.ours_halo_scale * 72) ** 2 * 0.55,
                    color=style.ours_color, alpha=style.ours_halo_alpha,
                    edgecolors="none", zorder=3,
                )

            oi = OffsetImage(img, zoom=zoom, interpolation="hanning")
            ab = AnnotationBbox(oi, (x, y), frameon=False, pad=0.0, zorder=5)
            ax.add_artist(ab)
        else:
            color = cfg["color"] or style.dot_color
            ax.scatter(
                [x], [y], s=style.dot_size, color=color,
                edgecolors=style.dot_edge_color, linewidths=style.dot_edge_width,
                zorder=4,
            )

        if cfg["show_label"]:
            is_ours = cfg["ours"]
            ax.annotate(
                cfg["label"], (x, y),
                xytext=(cfg["dx"], cfg["dy"]), textcoords="offset points",
                ha=cfg["ha"], va=cfg["va"],
                fontsize=style.annotation_size_ours if is_ours else style.annotation_size,
                fontweight="bold" if is_ours else "normal",
                color=darken_hex(style.ours_color, 0.92) if is_ours else style.annotation_color,
                zorder=6,
            )

    # --- axes cosmetics ----------------------------------------------------
    ax.set_xlim(*X_LIM)
    ax.set_ylim(*style.ylim)
    ax.set_xticks([t[0] for t in X_TICKS])
    ax.set_xticklabels([t[1] for t in X_TICKS])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v)}%"))
    ax.set_xlabel(style.xlabel, labelpad=10)
    ax.set_ylabel(style.ylabel, labelpad=10)
    ax.set_title(style.title, pad=style.title_pad, fontweight=style.title_weight)
    ax.tick_params(length=0)

    fig.tight_layout()
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out_base}.png", dpi=style.dpi, bbox_inches="tight")
    if style.save_pdf:
        fig.savefig(f"{out_base}.pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s.png%s", out_base, " (+ .pdf)" if style.save_pdf else "")


# ===========================================================================
# 7. CLI
# ===========================================================================


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help="Output path base (no extension).")
    ap.add_argument("--title", default=None, help="Override the figure title.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    rows, skipped = load_rows(args.csv.resolve())
    if skipped:
        logger.info("Skipped %d model(s) without a TB2 score yet: %s",
                    len(skipped), ", ".join(skipped))
    logger.info("Plotting %d model(s).", len(rows))

    style = TeaserStyle()
    if args.title:
        style.title = args.title
    render(rows, style, args.out.resolve())


if __name__ == "__main__":
    main()
