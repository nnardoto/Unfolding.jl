# Schema HDF5 dos dados desdobrados (spectral)

Nome do schema: `unfolding.spectral`; versão: `1.0`.

Este arquivo é o formato binário de saída de `write_unfolded_hdf5`, usado
para exportar o resultado de um unfolding ao longo de um caminho de pontos
k para qualquer programa de plotagem — Plots.jl, Matplotlib (via `h5py`),
gnuplot, ou qualquer outra ferramenta com suporte a HDF5. Ele não tem
relação com o `unfolding.realspace` de `docs/hdf5-schema.md`, que descreve o
modelo de entrada (`H(R)`, `S(R)`); este aqui descreve apenas o resultado já
processado (energias e pesos por banda, ao longo do caminho).

```text
/schema/name             UInt8 UTF-8 bytes
/schema/version          Int[2]
/path/kpoints_frac       Float64[3, nk]
/path/distance           Float64[nk]
/path/ticks              Float64[nticks]
/path/tick_labels        UTF-8 String[nticks]
/data/energies           Float64[nbands, nk]
/data/weights            Float64[nbands, nk]
/reference/energies      Float64[nbands_pc, nk]   (opcional)
/metadata/energy_unit    UInt8 UTF-8 bytes
```

- `kpoints_frac`: ponto k fracionário desdobrado (base recíproca da célula
  de referência) associado a cada posição do caminho.
- `distance`: distância acumulada ao longo do caminho, mesma convenção de
  `interpolate_kpath`.
- `ticks`/`tick_labels`: posições e rótulos dos vértices de alta simetria
  (por exemplo `Γ`, `K`, `M`); podem ser vazios se o caminho não tiver
  vértices nomeados.
- `energies`/`weights`: uma coluna por ponto do caminho, uma linha por banda
  da supercélula. Todo ponto do caminho deve ter o mesmo número de bandas.
- `reference` (grupo opcional): `energies` são as bandas exatas da célula de
  referência no mesmo caminho (por exemplo de `solve_bands(pc, kpoints)`),
  sem peso associado -- servem só de comparação visual com o unfolding
  acima, embutidas no mesmo arquivo em vez de um segundo arquivo separado.
  Presente somente quando `write_unfolded_hdf5` recebe `reference_energies`.

## Lendo em outras ferramentas

Python (Matplotlib), sem depender de Julia:

```python
import h5py
import matplotlib.pyplot as plt

with h5py.File("graphene_unfolding.h5", "r") as f:
    distance = f["path/distance"][:]
    ticks = f["path/ticks"][:]
    labels = [s.decode() if isinstance(s, bytes) else s for s in f["path/tick_labels"][:]]
    energies = f["data/energies"][:]   # (nbands, nk)
    weights = f["data/weights"][:]     # (nbands, nk)
    reference = f["reference/energies"][:] if "reference" in f else None  # (nbands_pc, nk), opcional

if reference is not None:
    for band in reference:
        plt.plot(distance, band, color="black", ls="--", lw=1.2)
for band in range(energies.shape[0]):
    plt.scatter([distance]*1, energies[band], c=weights[band], s=2 + 6*weights[band]**0.5,
                cmap="inferno", vmin=0, vmax=1)
plt.xticks(ticks, labels)
plt.show()
```

gnuplot também lê HDF5 diretamente (`plot 'graphene_unfolding.h5' binary format='%double' ...`,
ou via o utilitário `h5dump`/`h5totxt` para conversão prévia, dependendo da
versão instalada).

Julia com Plots.jl: veja `examples/graphene/plotting/plot_unfolding.jl`, que
lê este arquivo com `read_unfolded_hdf5` e não precisa recalcular nada.
