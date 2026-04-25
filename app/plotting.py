from __future__ import annotations


def write_line_chart_svg(path: str, series: list[float], title: str, y_label: str) -> None:
    width = 900
    height = 420
    margin_left = 70
    margin_bottom = 50
    margin_top = 50
    margin_right = 30

    if not series:
        series = [0.0]

    min_value = min(series)
    max_value = max(series)
    if max_value == min_value:
        max_value += 1.0

    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    points = []
    for index, value in enumerate(series):
        x = margin_left + (plot_width * index / max(1, len(series) - 1))
        normalized = (value - min_value) / (max_value - min_value)
        y = margin_top + plot_height - (normalized * plot_height)
        points.append(f"{x:.1f},{y:.1f}")

    polyline = " ".join(points)
    y_ticks = []
    for idx in range(5):
        tick_value = min_value + (max_value - min_value) * (idx / 4)
        y = margin_top + plot_height - (plot_height * idx / 4)
        y_ticks.append((y, tick_value))

    x_label = "Epoch"
    y_axis = "\n".join(
        f'<text x="{margin_left - 12}" y="{y + 4:.1f}" text-anchor="end" font-size="12" fill="#475569">{value:.2f}</text>'
        for y, value in y_ticks
    )
    y_grid = "\n".join(
        f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}" stroke="#e2e8f0" stroke-width="1" />'
        for y, _ in y_ticks
    )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="{width / 2:.1f}" y="28" text-anchor="middle" font-size="20" font-family="Arial, sans-serif" fill="#0f172a">{title}</text>
  <text x="24" y="{height / 2:.1f}" transform="rotate(-90 24,{height / 2:.1f})" text-anchor="middle" font-size="14" font-family="Arial, sans-serif" fill="#334155">{y_label}</text>
  <text x="{margin_left + plot_width / 2:.1f}" y="{height - 12}" text-anchor="middle" font-size="14" font-family="Arial, sans-serif" fill="#334155">{x_label}</text>
  {y_grid}
  <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#64748b" stroke-width="1.5" />
  <line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#64748b" stroke-width="1.5" />
  {y_axis}
  <polyline fill="none" stroke="#0f766e" stroke-width="3" points="{polyline}" />
</svg>
"""

    with open(path, "w", encoding="utf-8") as handle:
        handle.write(svg)
