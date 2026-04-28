#!/usr/bin/env python3
"""Draw prompt distribution charts for generated prompt JSON files."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

# Matplotlib defaults to ~/.matplotlib, which may be unavailable in sandboxed runs.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/prompt_generator_matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DIFFICULTY_ORDER = ["LOW", "MEDIUM", "HIGH"]
DEFAULT_INPUTS = [
    ("output/gemini_prompts.json", "Gemini"),
    ("output/gpt4o_prompts.json", "GPT-4o"),
    ("output/dpsk_prompts.json", "DPSK"),
]
MODEL_COLORS = {
    "Gemini": "#2E6F95",
    "GPT-4o": "#C66B3D",
    "DPSK": "#4F8A5B",
}
DIFFICULTY_COLORS = {
    "LOW": "#7FB069",
    "MEDIUM": "#E6A23C",
    "HIGH": "#C84C4C",
}


def configure_style() -> None:
    sns.set_theme(
        context="talk",
        style="whitegrid",
        rc={
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.edgecolor": "#27323A",
            "grid.color": "#D7DEE2",
            "grid.linewidth": 0.8,
            "axes.titleweight": "bold",
            "axes.labelcolor": "#27323A",
            "text.color": "#27323A",
        },
    )
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC",
        "Arial Unicode MS",
        "Heiti TC",
        "SimHei",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def load_prompts(path: Path, default_model: str) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    prompts = data.get("prompts", [])
    rows = []
    for prompt in prompts:
        llm = prompt.get("llm", {}) or {}
        provider = (llm.get("provider") or "").lower()
        model_name = llm.get("model") or ""
        display_model = model_display_name(provider, model_name, default_model)

        sampling = prompt.get("sampling", {}) or {}
        categories = sampling.get("categories_selected") or list(
            (sampling.get("concepts") or {}).keys()
        )
        challenges = sampling.get("challenge_elements") or []
        text = prompt.get("text") or ""
        difficulty = (prompt.get("difficulty", {}) or {}).get("level", "UNKNOWN")
        difficulty = str(difficulty).upper()

        rows.append(
            {
                "source_file": str(path),
                "model": display_model,
                "provider": provider,
                "prompt_id": prompt.get("prompt_id"),
                "combination_id": prompt.get("combination_id"),
                "difficulty": difficulty,
                "text_length": len(text.split()),
                "dimension_count": len(categories),
                "challenge_count": len(challenges),
            }
        )
    return rows


def model_display_name(provider: str, model_name: str, default_model: str) -> str:
    model_text = f"{provider} {model_name}".lower()
    if "gemini" in model_text:
        return "Gemini"
    if provider == "gpt" or "gpt-4o" in model_text or "4o" in model_text:
        return "GPT-4o"
    if provider == "dpsk" or "dpsk" in model_text or "deepseek" in model_text or "ds-v3" in model_text:
        return "DPSK"
    return default_model


def format_autopct(values: Iterable[int]):
    total = sum(values)

    def _formatter(pct: float) -> str:
        count = int(round(pct * total / 100.0))
        return f"{pct:.1f}%\n({count})" if count else ""

    return _formatter


def draw_difficulty_pie(df: pd.DataFrame, model: str, out_dir: Path) -> None:
    subset = df[df["model"] == model]
    counts = (
        subset["difficulty"]
        .value_counts()
        .reindex(DIFFICULTY_ORDER, fill_value=0)
    )
    nonzero_counts = counts[counts > 0]
    colors = [DIFFICULTY_COLORS[d] for d in nonzero_counts.index]

    fig, ax = plt.subplots(figsize=(8.5, 7.2))
    wedges, texts, autotexts = ax.pie(
        nonzero_counts.values,
        labels=nonzero_counts.index,
        colors=colors,
        startangle=90,
        counterclock=False,
        autopct=format_autopct(nonzero_counts.values),
        pctdistance=0.72,
        labeldistance=1.08,
        wedgeprops={"linewidth": 2, "edgecolor": "white"},
        textprops={"fontsize": 14, "weight": "bold"},
    )
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontsize(12)
        autotext.set_weight("bold")

    ax.set_title(f"{model} Difficulty Distribution", fontsize=22, pad=22)
    fig.text(
        0.5,
        0.04,
        f"Total prompts: {len(subset):,}",
        ha="center",
        fontsize=13,
        color="#52616B",
    )
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(out_dir / f"difficulty_distribution_{slug(model)}.png")
    plt.close(fig)


def draw_metric_bar(
    summary: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    out_path: Path,
    model_order: List[str],
    model_colors: Dict[str, str],
    value_fmt: str = "{:.1f}",
) -> None:
    plot_data = summary.copy()
    plot_data["difficulty"] = pd.Categorical(
        plot_data["difficulty"], categories=DIFFICULTY_ORDER, ordered=True
    )
    plot_data["model"] = pd.Categorical(
        plot_data["model"], categories=model_order, ordered=True
    )
    plot_data = plot_data.sort_values(["difficulty", "model"])

    fig, ax = plt.subplots(figsize=(10.5, 6.8))
    sns.barplot(
        data=plot_data,
        x="difficulty",
        y=metric,
        hue="model",
        palette=model_colors,
        hue_order=model_order,
        order=DIFFICULTY_ORDER,
        ax=ax,
        edgecolor="#27323A",
        linewidth=1.0,
    )

    ax.set_title(title, fontsize=21, pad=18)
    ax.set_xlabel("Difficulty")
    ax.set_ylabel(ylabel)
    ax.legend(title="", frameon=True, loc="upper left")
    ax.grid(axis="x", visible=False)
    ax.margins(y=0.14)

    for container in ax.containers:
        labels = [
            value_fmt.format(value.get_height()) if value.get_height() > 0 else ""
            for value in container
        ]
        ax.bar_label(container, labels=labels, fontsize=11, padding=4, color="#27323A")

    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def write_summary_tables(df: pd.DataFrame, out_dir: Path, model_order: List[str]) -> pd.DataFrame:
    summary = (
        df.groupby(["model", "difficulty"], observed=False)
        .agg(
            prompt_count=("prompt_id", "count"),
            avg_text_length=("text_length", "mean"),
            avg_dimension_count=("dimension_count", "mean"),
            avg_challenge_count=("challenge_count", "mean"),
        )
        .reset_index()
    )
    summary["difficulty"] = pd.Categorical(
        summary["difficulty"], categories=DIFFICULTY_ORDER, ordered=True
    )
    summary["model"] = pd.Categorical(
        summary["model"], categories=model_order, ordered=True
    )
    summary = summary.sort_values(["model", "difficulty"])
    summary.to_csv(out_dir / "summary_by_model_difficulty.csv", index=False)

    difficulty_counts = (
        df.groupby(["model", "difficulty"], observed=False)
        .size()
        .reset_index(name="count")
    )
    totals = difficulty_counts.groupby("model", observed=False)["count"].transform("sum")
    difficulty_counts["percentage"] = difficulty_counts["count"] / totals * 100
    difficulty_counts.to_csv(out_dir / "difficulty_distribution.csv", index=False)
    return summary


def slug(name: str) -> str:
    return name.lower().replace("-", "").replace(" ", "_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gemini",
        default="output/gemini_prompts.json",
        help="Gemini output JSON path.",
    )
    parser.add_argument(
        "--gpt4o",
        default="output/gpt4o_prompts.json",
        help="GPT-4o output JSON path.",
    )
    parser.add_argument(
        "--dpsk",
        default="output/dpsk_prompts.json",
        help="DPSK output JSON path.",
    )
    parser.add_argument(
        "--input",
        action="append",
        help="Additional or replacement input in path:label format. If provided, only these inputs are used.",
    )
    parser.add_argument(
        "--out-dir",
        default="reports/prompt_stats_png",
        help="Output directory for PNG charts and CSV summaries.",
    )
    return parser.parse_args()


def resolve_inputs(args: argparse.Namespace) -> List[Tuple[Path, str]]:
    if args.input:
        specs = []
        for raw in args.input:
            if ":" not in raw:
                raise ValueError(f"--input must use path:label format, got: {raw}")
            path, label = raw.rsplit(":", 1)
            specs.append((Path(path), label))
        return specs
    return [
        (Path(args.gemini), "Gemini"),
        (Path(args.gpt4o), "GPT-4o"),
        (Path(args.dpsk), "DPSK"),
    ]


def build_model_colors(model_order: List[str]) -> Dict[str, str]:
    fallback = ["#2E6F95", "#C66B3D", "#4F8A5B", "#8A6F3D", "#52616B"]
    colors = {}
    for index, model in enumerate(model_order):
        colors[model] = MODEL_COLORS.get(model, fallback[index % len(fallback)])
    return colors


def main() -> None:
    args = parse_args()
    configure_style()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_specs = resolve_inputs(args)
    model_order = [label for _, label in input_specs]
    model_colors = build_model_colors(model_order)

    rows = []
    for path, label in input_specs:
        rows.extend(load_prompts(path, label))
    df = pd.DataFrame(rows)
    df = df[df["difficulty"].isin(DIFFICULTY_ORDER)].copy()

    for model in model_order:
        draw_difficulty_pie(df, model, out_dir)

    summary = write_summary_tables(df, out_dir, model_order)

    draw_metric_bar(
        summary,
        metric="avg_text_length",
        ylabel="Average Word Count",
        title="Average Prompt Length by Difficulty",
        out_path=out_dir / "avg_text_length_by_difficulty.png",
        model_order=model_order,
        model_colors=model_colors,
    )
    draw_metric_bar(
        summary,
        metric="avg_dimension_count",
        ylabel="Average Selected Dimensions",
        title="Average Dimension Count by Difficulty",
        out_path=out_dir / "avg_dimension_count_by_difficulty.png",
        model_order=model_order,
        model_colors=model_colors,
        value_fmt="{:.2f}",
    )
    draw_metric_bar(
        summary,
        metric="avg_challenge_count",
        ylabel="Average Challenge Elements",
        title="Average Challenge Count by Difficulty",
        out_path=out_dir / "avg_challenge_count_by_difficulty.png",
        model_order=model_order,
        model_colors=model_colors,
        value_fmt="{:.2f}",
    )

    print(f"Charts written to: {out_dir}")
    print(f"Models: {', '.join(model_order)}")
    for path in sorted(out_dir.glob("*.png")):
        print(f"- {path}")


if __name__ == "__main__":
    main()
