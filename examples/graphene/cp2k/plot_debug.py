import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm


run_dir = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).parent / "run_debug"
csv_path = run_dir / "graphene_debug_unfolded.csv"
reference_csv_path = run_dir / "graphene_debug_reference.csv"
png_path = run_dir / "graphene_debug_unfolded.png"

distance = []
energy = []
weight = []
path_points = {}
with csv_path.open(newline="") as stream:
    for row in csv.DictReader(stream):
        point_distance = float(row["distance"])
        distance.append(point_distance)
        energy.append(float(row["energy_ev"]))
        weight.append(float(row["weight"]))
        path_points.setdefault(
            point_distance,
            (float(row["k1"]), float(row["k2"]), float(row["k3"])),
        )

reference_bands = {}
with reference_csv_path.open(newline="") as stream:
    for row in csv.DictReader(stream):
        band = int(row["band"])
        reference_bands.setdefault(band, ([], []))
        reference_bands[band][0].append(float(row["distance"]))
        reference_bands[band][1].append(float(row["energy_ev"]))

fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=150)
ax.set_facecolor("#100B1E")

# Função espectral contínua: cada autovalor contribui com uma gaussiana cuja
# amplitude é o peso de unfolding. O pequeno alargamento deixa as bandas
# legíveis sem esconder separações reais entre elas.
path_distance = np.array(sorted(path_points))
distance_index = {value: index for index, value in enumerate(path_distance)}
energy_grid = np.linspace(-8.0, 8.0, 900)
spectral = np.zeros((energy_grid.size, path_distance.size))
broadening_ev = 0.055
for point_distance, point_energy, point_weight in zip(distance, energy, weight):
    if point_weight < 1.0e-7:
        continue
    column = distance_index[point_distance]
    gaussian = np.exp(-0.5 * ((energy_grid - point_energy) / broadening_ev) ** 2)
    spectral[:, column] += point_weight * gaussian
if spectral.max() > 0.0:
    spectral /= spectral.max()

# Interpolação apenas para renderização ao longo do caminho k. As energias e
# pesos originais continuam sendo os 94 pontos calculados.
dense_distance = np.linspace(path_distance[0], path_distance[-1], 8 * path_distance.size)
dense_spectral = np.vstack(
    [np.interp(dense_distance, path_distance, row) for row in spectral]
)

heatmap = ax.pcolormesh(
    dense_distance,
    energy_grid,
    dense_spectral,
    shading="auto",
    cmap="magma",
    norm=PowerNorm(gamma=0.65, vmin=0.0, vmax=1.0),
    rasterized=True,
    zorder=1,
)
for band, (band_distance, band_energy) in reference_bands.items():
    # Contorno escuro fino mantém a referência visível tanto sobre regiões
    # amarelas intensas quanto sobre o fundo roxo do mapa.
    ax.plot(
        band_distance,
        band_energy,
        color="#101010",
        linewidth=2.4,
        alpha=0.8,
        zorder=3,
    )
    ax.plot(
        band_distance,
        band_energy,
        color="#22D3EE",
        linewidth=1.05,
        alpha=1.0,
        label="Célula primitiva (CP2K)" if band == min(reference_bands) else None,
        zorder=4,
    )

targets = [(0.0, 0.0, 0.0), (2 / 3, 1 / 3, 0.0), (0.5, 0.0, 0.0), (0.0, 0.0, 0.0)]
ordered_points = list(path_points.items())
ticks = []
start = 0
for target in targets:
    index = min(
        range(start, len(ordered_points)),
        key=lambda i: sum((ordered_points[i][1][j] - target[j]) ** 2 for j in range(3)),
    )
    ticks.append(ordered_points[index][0])
    start = index + 1 if index + 1 < len(ordered_points) else index
for tick in ticks:
    ax.axvline(tick, color="white", linewidth=0.65, alpha=0.32, zorder=2)

ax.axhline(0.0, color="white", linestyle="--", linewidth=0.8, alpha=0.75, zorder=2)
ax.set_xticks(ticks, ["Γ", "K", "M", "Γ"])
ax.set_xlim(ticks[0], ticks[-1])
ax.set_ylim(-8.0, 8.0)
ax.set_xlabel("Caminho k")
ax.set_ylabel("E − referência em K (eV)")
ax.set_title("Grafeno 2×2 — debug do unfolding CP2K")
ax.legend(loc="upper right", frameon=True, facecolor="#100B1E", edgecolor="none",
          labelcolor="white")
colorbar = fig.colorbar(heatmap, ax=ax, label="Intensidade espectral normalizada")
colorbar.ax.yaxis.label.set_color("#222222")
fig.tight_layout()
fig.savefig(png_path, dpi=220)
print(f"Figure written to {png_path}")
