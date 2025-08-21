#!/usr/bin/env python3
# FEL_2D_only.py — Paysage d’énergie libre 2D (ΔG), échelle 0–12 kJ/mol

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import gaussian_kde

# ── Réglages utilisateur ────────────────────────────────────────
INFILE = '2Dproj_PC1_PC2.xvg'   # fichier PC1/PC2 (2 colonnes)
TEMP   = 300.0                  # K
kB     = 0.008314               # kJ mol⁻1 K⁻1
BINS   = 100                    # maillage XY (plus grand = plus fin)
BWIDTH = 0.15                   # bande KDE (plus petit = plus détaillé)
Z_MAX  = 12.0                   # plafond ΔG (kJ/mol)

# ── Chargement ──────────────────────────────────────────────────
pc1, pc2 = np.loadtxt(INFILE, comments=('#', '@')).T

# ── Densité ρ(x,y) → ΔG = −kT ln(ρ/ρmax) ───────────────────────
kde   = gaussian_kde([pc1, pc2], bw_method=BWIDTH)
xi, yi = [np.linspace(v.min(), v.max(), BINS + 1) for v in (pc1, pc2)]
xc, yc = [0.5 * (v[:-1] + v[1:]) for v in (xi, yi)]
X, Y   = np.meshgrid(xc, yc)
rho    = kde([X.ravel(), Y.ravel()]).reshape(X.shape)

dG = -kB * TEMP * np.log(rho / rho.max())
dG = np.clip(dG, 0.0, Z_MAX)  # bornes 0–12

# ── Colormap & normalisation ───────────────────────────────────
cmap = plt.get_cmap('jet')
norm = mpl.colors.Normalize(vmin=0.0, vmax=Z_MAX)

# ── Tracé 2D ───────────────────────────────────────────────────
plt.rcParams.update({'font.size': 10})
fig, ax = plt.subplots(figsize=(6.0, 5.6), dpi=300)

pcm = ax.pcolormesh(xi, yi, dG, cmap=cmap, norm=norm, shading='auto')
# (Optionnel) isolignes toutes les 2 kJ/mol — commentez pour enlever
ax.contour(X, Y, dG, levels=np.arange(0, Z_MAX + 0.001, 2), colors='k', linewidths=0.35, alpha=0.6)

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_title('Free Energy Landscape (ΔG)', pad=10)
ax.set_aspect('equal', adjustable='box')

# Échelle unique à droite, 0–12 kJ/mol
cbar = fig.colorbar(pcm, ax=ax, pad=0.02, fraction=0.046, extend='max')
cbar.set_label('ΔG (kJ/mol)', weight='bold')
cbar.set_ticks(np.arange(0, int(Z_MAX) + 1, 1))
pcm.cmap.set_over('red')  # valeurs >12 en rouge

fig.tight_layout()
fig.savefig('FEL_2D_only_DG_0_12.png')
plt.show()
