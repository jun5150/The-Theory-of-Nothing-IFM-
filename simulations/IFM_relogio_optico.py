# ifm_relogio_optico.py

import numpy as np
import matplotlib.pyplot as plt
import time

# ============================================================
# PARÂMETROS (AJUSTADOS)
# ============================================================
Nx, Ny = 20, 20
Γ0 = 0.5
α = -4.0                      # aumentado de -2.0
Γh = 2.0
t_max = 50.0                  # aumentado de 20.0
dt_amostra = 0.5
R0 = 0.1

np.random.seed(42)

# Perfil de curvatura fixo
x = np.arange(Nx) - Nx//2
y = np.arange(Ny) - Ny//2
X, Y = np.meshgrid(x, y)
r = np.sqrt(X**2 + Y**2) + 0.1
r_s = 5.0
R = R0 / (1 + (r / r_s)**3)

# Estado inicial: 50% ocupado, metade + metade -
n = np.zeros((Nx, Ny), dtype=np.int8)
for i in range(Nx):
    for j in range(Ny):
        if np.random.random() < 0.5:
            n[i, j] = 1 if np.random.random() < 0.5 else -1

print("="*60)
print("IFM 2D - RELÓGIO ÓPTICO (CORRIGIDO)")
print("="*60)
print(f"Grade: {Nx}×{Ny} = {Nx*Ny} sítios")
print(f"t_max = {t_max}")
print(f"R0 = {R0}, α = {α}, Γh = {Γh}")

# ============================================================
# FUNÇÕES
# ============================================================
def obter_eventos(n, R, Γ0, α, Γh):
    eventos = []
    for i in range(Nx):
        for j in range(Ny):
            if n[i, j] == 0:
                taxa_c = Γ0 * (1 + α * R[i, j])
                if taxa_c > 0:
                    eventos.append(('criar+', i, j, taxa_c))
                    eventos.append(('criar-', i, j, taxa_c))
            else:
                taxa_a = Γ0 * (1 - α * R[i, j])
                if taxa_a > 0:
                    eventos.append(('aniquilar', i, j, taxa_a))
                for di, dj in [(1,0), (-1,0), (0,1), (0,-1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < Nx and 0 <= nj < Ny and n[ni, nj] == 0:
                        eventos.append(('hop', i, j, Γh, ni, nj))
    return eventos

# ============================================================
# SIMULAÇÃO
# ============================================================
n_atual = n.copy()
eventos = obter_eventos(n_atual, R, Γ0, α, Γh)

t = 0.0
phi_hist = []
t_hist = []
prox_amostra = 0.0

start = time.time()

while t < t_max and eventos:
    taxa_total = sum(e[3] for e in eventos)
    if taxa_total <= 0:
        break
    
    dt = -np.log(np.random.random()) / taxa_total
    t += dt
    
    while t >= prox_amostra:
        phi = 2 * (np.abs(n_atual) > 0).astype(float) - 1
        phi_hist.append(phi.flatten())
        t_hist.append(prox_amostra)
        prox_amostra += dt_amostra
    
    r = np.random.random() * taxa_total
    acc = 0.0
    evento_escolhido = None
    for e in eventos:
        acc += e[3]
        if r <= acc:
            evento_escolhido = e
            break
    
    if evento_escolhido is None:
        break
    
    tipo = evento_escolhido[0]
    i, j = evento_escolhido[1], evento_escolhido[2]
    
    if tipo == 'criar+':
        n_atual[i, j] = 1
    elif tipo == 'criar-':
        n_atual[i, j] = -1
    elif tipo == 'aniquilar':
        n_atual[i, j] = 0
    elif tipo == 'hop':
        ni, nj = evento_escolhido[4], evento_escolhido[5]
        tipo_bit = n_atual[i, j]
        n_atual[i, j] = 0
        n_atual[ni, nj] = tipo_bit
    
    eventos = obter_eventos(n_atual, R, Γ0, α, Γh)

print(f"\nConcluído em {time.time()-start:.2f}s")
print(f"Amostras: {len(t_hist)}")

if not t_hist:
    print("Nenhuma amostra coletada.")
    exit()

# ============================================================
# ANÁLISE
# ============================================================
phi_arr = np.array(phi_hist).reshape(-1, Nx, Ny)

# Descartar transiente (primeiros 50%)
transiente = len(phi_arr) // 2
phi_med = np.mean(phi_arr[transiente:], axis=0)

centro = Nx // 2
phi_centro = phi_med[centro, centro]

# Flutuação no centro
phi_centro_series = phi_arr[transiente:, centro, centro]
flut_centro = np.std(phi_centro_series)

# Flutuação na borda (média dos cantos)
borda_vals = []
for i in [0, -1]:
    for j in [0, -1]:
        borda_vals.extend(phi_arr[transiente:, i, j])
flut_borda = np.std(borda_vals)

# Curvatura no centro e borda
R_centro = R[centro, centro]
R_borda = np.mean(R[0:2, 0:2])

print("\n" + "="*60)
print("EXTRAÇÃO DE β - RELÓGIO ÓPTICO (CORRIGIDO)")
print("="*60)
print(f"φ médio no centro: {phi_centro:.4f}")
print(f"Flutuação no centro (σ): {flut_centro:.4f}")
print(f"Flutuação na borda (σ₀): {flut_borda:.4f}")
print(f"Curvatura no centro (R): {R_centro:.4f}")
print(f"Curvatura na borda: {R_borda:.4f}")

if R_centro > 0 and flut_borda > 0:
    beta = (1 - flut_centro / flut_borda) / R_centro
    print(f"\nβ = {beta:.3f}")
    if 0.3 < beta < 0.5:
        print("β na faixa esperada (0.4 ± 0.1)")
    else:
        print("β fora da faixa esperada.")
else:
    print("\nβ não calculável.")

# ============================================================
# VISUALIZAÇÃO
# ============================================================
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
im1 = axs[0].imshow(phi_med, origin='lower', cmap='RdBu_r')
axs[0].set_title(f'φ médio (β = {beta:.3f})')
plt.colorbar(im1, ax=axs[0])
im2 = axs[1].imshow(R, origin='lower', cmap='hot')
axs[1].set_title('Curvatura fixa R(x,y)')
plt.colorbar(im2, ax=axs[1])
plt.tight_layout()
plt.savefig('ifm_relogio_optico.png', dpi=120)
plt.show()