#!/usr/bin/env python3
# FEL_3D_with_base2D.py — Surface 3D + "tapis" 2D au fond (z=0) avec échelle ΔG unique 0–12 (sans titre)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import gaussian_kde
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ── Réglages utilisateur ────────────────────────────────────────
INFILE = '2Dproj_PC1_PC2.xvg'   # fichier PC1/PC2 (2 colonnes)
TEMP   = 300.0                  # K
kB     = 0.008314               # kJ mol⁻1 K⁻1
BINS   = 100                    # maillage XY (plus grand = plus fin)
BWIDTH = 0.15                   # bande KDE (plus petit = plus détaillé)
Z_MAX  = 12.0                   # plafond ΔG (kJ/mol)

# ── Chargement des PC ───────────────────────────────────────────
pc1, pc2 = np.loadtxt(INFILE, comments=('#', '@')).T

# ── KDE ρ(x,y) → ΔG = -kT ln(ρ/ρmax) ────────────────────────────
kde   = gaussian_kde([pc1, pc2], bw_method=BWIDTH)
xi, yi = [np.linspace(v.min(), v.max(), BINS + 1) for v in (pc1, pc2)]
xc, yc = [0.5 * (v[:-1] + v[1:]) for v in (xi, yi)]
X, Y   = np.meshgrid(xc, yc)
rho    = kde([X.ravel(), Y.ravel()]).reshape(X.shape)

dG = -kB * TEMP * np.log(rho / rho.max())
dG = np.clip(dG, 0.0, Z_MAX)  # bornes 0–12

# ── Colormap et normalisation partagées ─────────────────────────
cmap = plt.get_cmap('jet')
norm = mpl.colors.Normalize(vmin=0.0, vmax=Z_MAX)

# ── Figure : un seul axe 3D ─────────────────────────────────────
plt.rcParams.update({'font.size': 10})
fig = plt.figure(figsize=(7.2, 6.2), dpi=300)
ax  = fig.add_subplot(111, projection='3d')

# Surface 3D (couleurs = ΔG)
surf = ax.plot_surface(
    X, Y, dG,
    rstride=1, cstride=1,
    cmap=cmap, norm=norm,
    linewidth=0, antialiased=False, shade=False
)

# "Tapis" 2D au fond (plan z=0)
levels = np.linspace(0.0, Z_MAX, 100)
ax.contourf(
    X, Y, dG, levels=levels,
    zdir='z', offset=0.0,
    cmap=cmap, norm=norm, antialiased=False
)

# Axes
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('ΔG (kJ/mol)')
ax.set_zlim(0.0, Z_MAX)
ax.set_box_aspect((1, 1, 0.6))
ax.view_init(elev=28, azim=-40)

# ── Colorbar unique à droite (0 → 12) ───────────────────────────
mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
mappable.set_array([])
cbar = fig.colorbar(mappable, ax=ax, pad=0.12, fraction=0.03)
cbar.set_label('ΔG (kJ/mol)', weight='bold')
cbar.set_ticks(np.arange(0, int(Z_MAX) + 1, 1))

# (Titre supprimé)  # fig.suptitle('...', y=0.98)
fig.tight_layout()   # au lieu de rect=(...) pour ne pas réserver de place au titre

fig.savefig('FEL_3D_with_base2D_DG_0_12.png')
plt.show()
