"""Telemetry plots around a finding: one panel per topic, the finding's window shaded."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from volatus.detectors import Finding, euler
from volatus.flight import Flight, Series, clock

PAD_S = 10.0
BG, INK, MUTED, GRID = "#000000", "#f0efec", "#c3c2b7", "#2a2a28"
SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#0ca30c", "#9085e9", "#e66767"]
HIGHLIGHT = "#e66767"


def signal(f: Flight, spec: str) -> Series | None:
    """Resolve "topic/field[/instance]". vehicle_attitude/roll_deg and pitch_deg are derived from q."""
    topic, fld, *inst = spec.split("/")
    if topic == "vehicle_attitude" and fld in ("roll_deg", "pitch_deg"):
        e = euler(f)
        return None if e is None else Series(e[0], e[1] if fld == "roll_deg" else e[2])
    return f.series(topic, fld, int(inst[0]) if inst else 0)


def plot_finding(f: Flight, finding: Finding, out_dir: Path) -> Path | None:
    """Write a PNG of the finding's signals, PAD_S seconds either side. None if nothing to plot."""
    t0, t1 = max(0.0, finding.t_start - PAD_S), finding.t_end + PAD_S
    panels: dict[str, list[tuple[str, Series]]] = {}
    for spec in finding.signals:
        s = signal(f, spec)
        if s is not None and len((w := s.window(t0, t1)).t):
            panels.setdefault(spec.split("/")[0], []).append((spec.split("/")[1], w))
    if not panels:
        return None

    fig, axes = plt.subplots(len(panels), 1, figsize=(9, 1.9 * len(panels) + 0.6), sharex=True, squeeze=False)
    fig.patch.set_facecolor(BG)
    for ax, (topic, series) in zip(axes[:, 0], panels.items()):
        ax.set_facecolor(BG)
        for i, (name, s) in enumerate(series):
            ax.plot(s.t, s.v, color=SERIES[i % len(SERIES)], lw=1.4, label=name)
        ax.axvspan(finding.t_start, max(finding.t_end, finding.t_start + 0.1), color=HIGHLIGHT, alpha=0.22, lw=0)
        ax.set_title(topic if len(series) > 1 else f"{topic}.{series[0][0]}", color=MUTED, fontsize=9, loc="left")
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.grid(color=GRID, lw=0.6)
        for side in ax.spines.values():
            side.set_color(GRID)
        if len(series) > 1:
            ax.legend(fontsize=7, ncol=min(len(series), 8), facecolor=BG, edgecolor=GRID, labelcolor=INK, loc="upper left")
    ax = axes[-1, 0]
    ticks = np.linspace(t0, t1, 7)
    ax.set_xticks(ticks, [clock(t) for t in ticks])
    ax.set_xlabel("time since log start (m:ss)", color=MUTED, fontsize=8)
    fig.suptitle(f"{finding.kind}  {finding.when}  {finding.summary[:90]}", color=INK, fontsize=9, x=0.01, ha="left")
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{finding.id}.png"
    fig.savefig(path, dpi=110, facecolor=BG)
    plt.close(fig)
    return path
