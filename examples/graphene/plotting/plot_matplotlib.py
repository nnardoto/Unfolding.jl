"""Plota o mesmo resultado de examples/graphene/export_unfolding.jl, mas em
Python, sem nenhuma dependência de Julia. Prova de que o HDF5 gravado por
write_unfolded_hdf5 (docs/unfolded-hdf5-schema.md) é de fato agnóstico de
ferramenta: qualquer leitor de HDF5 e qualquer biblioteca de plotagem serve.

Requer: pip install h5py matplotlib numpy
"""
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np


def read_unfolded(path):
    with h5py.File(path, "r") as f:
        return {
            "distance": f["path/distance"][:],
            "ticks": f["path/ticks"][:],
            "tick_labels": [
                s.decode() if isinstance(s, bytes) else s
                for s in f["path/tick_labels"][:]
            ],
            "energies": f["data/energies"][:],  # (nbands, nk)
            "weights": f["data/weights"][:],     # (nbands, nk)
            "reference_energies": (
                f["reference/energies"][:] if "reference" in f else None
            ),  # (nbands_pc, nk), opcional
        }


root = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(root, "..", "output")

data = read_unfolded(os.path.join(out, "graphene_unfolding.h5"))

fig, ax = plt.subplots(figsize=(9, 6), dpi=140)

if data["reference_energies"] is not None:
    for band in data["reference_energies"]:
        ax.plot(data["distance"], band, color="black", lw=1.2, ls="--")

nbands, nk = data["energies"].shape
x = np.tile(data["distance"], nbands)
y = data["energies"].reshape(-1)
w = data["weights"].reshape(-1)
visible = w > 1e-4
scatter = ax.scatter(x[visible], y[visible], c=w[visible], cmap="inferno", vmin=0, vmax=1,
                      s=4 + 40 * np.sqrt(w[visible]), linewidths=0, alpha=0.82)
fig.colorbar(scatter, ax=ax, label="Peso")

ax.set_xticks(data["ticks"])
ax.set_xticklabels(data["tick_labels"])
ax.set_xlim(data["distance"][0], data["distance"][-1])
ax.set_ylim(-8.5, 8.5)
ax.set_xlabel("Caminho k")
ax.set_ylabel("Energia (eV)")
ax.set_title("Grafeno 2×2 com defeito: bandas desdobradas (Matplotlib)")

fig.tight_layout()
path_out = os.path.join(out, "graphene_unfolding_matplotlib.png")
fig.savefig(path_out)
print("Wrote", path_out)
