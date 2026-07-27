"""Headless Matplotlib charts for CIFAR-10 benchmark reports."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

PlotSeries = tuple[str, Mapping[int, float], str]
PlotPanel = tuple[str, str, Sequence[PlotSeries]]


def write_line_plot(
    path: Path,
    *,
    title: str,
    y_label: str,
    series: Sequence[PlotSeries],
    provenance: Mapping[str, object],
) -> None:
    """Plot one or more real benchmark series to a headless PNG."""
    figure = Figure(figsize=(9, 4.8), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.add_subplot()
    _plot_series(axes, series)
    axes.set(title=title, xlabel="Round", ylabel=y_label)
    _save(figure, path, title=title, provenance=provenance)


def write_panel_plot(
    path: Path,
    *,
    title: str,
    panels: Sequence[PlotPanel],
    provenance: Mapping[str, object],
) -> None:
    """Plot vertically stacked benchmark panels to a headless PNG."""
    figure = Figure(figsize=(9, 6.4), layout="constrained")
    FigureCanvasAgg(figure)
    axes = [
        figure.add_subplot(len(panels), 1, index + 1)
        for index in range(len(panels))
    ]
    for axis, (panel_title, y_label, series) in zip(axes, panels, strict=True):
        _plot_series(axis, series)
        axis.set(title=panel_title, xlabel="Round", ylabel=y_label)
    figure.suptitle(title)  # pyright: ignore[reportUnknownMemberType]
    _save(figure, path, title=title, provenance=provenance)


def _plot_series(axes: Axes, series: Sequence[PlotSeries]) -> None:
    plotted = False
    for label, values, color in series:
        points = sorted(values.items())
        if not points:
            continue
        axes.plot(  # pyright: ignore[reportUnknownMemberType]
            [round_id for round_id, _ in points],
            [value for _, value in points],
            color=color,
            linewidth=2,
            marker="o",
            markersize=3,
            label=label,
        )
        plotted = True
    axes.grid(alpha=0.25)  # pyright: ignore[reportUnknownMemberType]
    if plotted:
        axes.legend()  # pyright: ignore[reportUnknownMemberType]


def _save(
    figure: Figure,
    path: Path,
    *,
    title: str,
    provenance: Mapping[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(  # pyright: ignore[reportUnknownMemberType]
        path,
        format="png",
        dpi=160,
        metadata={
            "Title": title,
            "Description": json.dumps(dict(provenance), sort_keys=True),
        },
    )
