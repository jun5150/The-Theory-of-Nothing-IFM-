# IFM_1D.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from numba import jit
import time

# ============================================================
# PARÂMETROS
# ============================================================
N = 100                     # número de sítios
Γ0 = 0.5                    # taxa base
α = -2.0                    # acoplamento curvatura
Γh = 2.0                    # taxa de hopping
t_max = 100.0               # tempo máximo
dt_amostra = 0.5            # intervalo de amostragem

# Perfil de curvatura (simulando buraco negro)
x = np.arange(N)
centro = N // 2
x_rel = np.abs(x - centro) + 1.0
R0 = 0.1
x_s = 20.0
R = R0 / (1 + (x_rel / x_s)**3)

print("="*60)
print("SIMULAÇÃO MFI - CÓDIGO CORRIGIDO")
print("="*60)
print(f"N = {N} sítios")
print(f"t_max = {t_max}")
print(f"Amplitude curvatura R0 = {R0}")

# ============================================================
# GILLESPIE (ALGORITMO DE EVENTOS)
# ============================================================

n = np.random.randint(0, 2, size=N)  # estado inicial: 0 ou 1

@jit(nopython=True)
def taxa_criacao(n, i, R, Γ0, α):
    """Taxa de criação (0 -> 1) no sítio i"""
    if n[i] == 0:
        return Γ0 * (1 + α * R[i])
    return 0.0

@jit(nopython=True)
def taxa_aniquilacao(n, i, R, Γ0, α):
    """Taxa de aniquilação (1 -> 0) no sítio i"""
    if n[i] == 1:
        return Γ0 * (1 - α * R[i])
    return 0.0

def obter_taxas(n, R, Γ0, α):
    """Retorna listas de todas as taxas e eventos possíveis"""
    eventos = []
    for i in range(N):
        # Criação
        taxa_c = Γ0 * (1 + α * R[i]) if n[i] == 0 else 0.0
        if taxa_c > 0:
            eventos.append(('criar', i, taxa_c))
        
        # Aniquilação
        taxa_a = Γ0 * (1 - α * R[i]) if n[i] == 1 else 0.0
        if taxa_a > 0:
            eventos.append(('aniquilar', i, taxa_a))
        
        # Hopping para direita
        if n[i] == 1 and i+1 < N and n[i+1] == 0:
            eventos.append(('hop_direita', i, Γh))
        
        # Hopping para esquerda
        if n[i] == 1 and i-1 >= 0 and n[i-1] == 0:
            eventos.append(('hop_esquerda', i, Γh))
    
    return eventos

print("\nRodando simulação Gillespie...")
start_time = time.time()

eventos = obter_taxas(n, R, Γ0, α)
t = 0.0
phi_history = []
t_history = []

# Amostragem
proximo_t_amostra = 0.0
ultimo_t_amostra = 0.0

while t < t_max and len(eventos) > 0:
    # Taxa total
    taxa_total = sum(e[2] for e in eventos)
    
    if taxa_total <= 0:
        break
    
    # Tempo até próximo evento
    dt = -np.log(np.random.random()) / taxa_total
    t += dt
    
    # Amostragem durante o intervalo
    while t >= proximo_t_amostra:
        t_history.append(proximo_t_amostra)
        phi_history.append(2 * n - 1)
        proximo_t_amostra += dt_amostra
        ultimo_t_amostra = proximo_t_amostra
    
    # Selecionar e executar evento
    r = np.random.random() * taxa_total
    acum = 0.0
    for evento in eventos:
        acum += evento[2]
        if r <= acum:
            tipo = evento[0]
            i = evento[1]
            
            if tipo == 'criar':
                n[i] = 1
            elif tipo == 'aniquilar':
                n[i] = 0
            elif tipo == 'hop_direita':
                n[i] = 0
                n[i+1] = 1
            elif tipo == 'hop_esquerda':
                n[i] = 0
                n[i-1] = 1
            break
    
    # Recalcular eventos
    eventos = obter_taxas(n, R, Γ0, α)

print(f"Gillespie concluído em {time.time() - start_time:.2f}s")
print(f"Eventos processados: {len(t_history)}")

# Converter listas para arrays
phi_todos = np.array(phi_history)
t_todos = np.array(t_history)

print(f"phi_todos shape: {phi_todos.shape}")

# ============================================================
# ODE (EQUAÇÃO CONTÍNUA)
# ============================================================

def sistema_phi(phi, t, D, γ, R, Γ0, α):
    """dφ/dt = D ∇²φ - γ φ + 2Γ0α R"""
    dphi = np.zeros(N)
    for i in range(N):
        lap = 0.0
        if i > 0:
            lap += phi[i-1] - phi[i]
        if i < N-1:
            lap += phi[i+1] - phi[i]
        dphi[i] = D * lap - γ * phi[i] + 2 * Γ0 * α * R[i]
    return dphi

D = 1.0
γ = 2 * Γ0

# Condição inicial: média comparável à simulação
phi0 = np.zeros(N)  # começa em 0 (⟨n⟩=0.5)

t_ode = np.linspace(0, t_max, len(t_history))
print("\nResolvendo ODE...")
sol_ode = odeint(sistema_phi, phi0, t_ode, args=(D, γ, R, Γ0, α))
print("ODE concluída.")

# ============================================================
# ANÁLISE
# ============================================================

# Estado estacionário (média dos últimos 50% do tempo)
transiente = len(t_history) // 2
phi_est_gillespie = np.mean(phi_todos[transiente:], axis=0)
phi_est_ode = sol_ode[-1]

# Flutuações
phi_centro_gillespie = phi_todos[transiente:, centro]
flut_centro = np.std(phi_centro_gillespie)
media_centro = np.mean(phi_centro_gillespie)

# RMS da diferença
dif = phi_est_gillespie - phi_est_ode
rms = np.sqrt(np.mean(dif**2))

print("\n" + "="*60)
print("RESULTADOS")
print("="*60)
print(f"RMS diferença (Gillespie vs ODE): {rms:.4f}")
print(f"φ médio no centro [Gillespie]: {media_centro:.4f}")
print(f"Flutuações (desvio padrão) no centro: {flut_centro:.4f}")
print(f"φ médio total [Gillespie]: {np.mean(phi_est_gillespie):.4f}")
print(f"φ médio total [ODE]: {np.mean(phi_est_ode):.4f}")

# ============================================================
# GRÁFICOS
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Evolução temporal (apenas um sítio representativo)
ax = axes[0, 0]
ax.plot(t_todos[transiente:], phi_centro_gillespie, 'b-', alpha=0.5, label='Gillespie')
ax.set_xlabel('tempo (t_P)')
ax.set_ylabel(f'φ (sítio {centro})')
ax.set_title(f'Evolução temporal - centro (N={N})')
ax.legend()
ax.grid(True)

# 2. Perfil estacionário
ax = axes[0, 1]
ax.plot(x, phi_est_gillespie, 'bo-', markersize=4, label='Gillespie (média)')
ax.plot(x, phi_est_ode, 'r--', linewidth=2, label='ODE')
ax.plot(x, 2*α*R, 'g:', alpha=0.5, label='fonte (2αR)')
ax.set_xlabel('posição x (l_P)')
ax.set_ylabel('φ')
ax.set_title('Perfil estacionário')
ax.legend()
ax.grid(True)

# 3. Histograma das flutuações
ax = axes[1, 0]
ax.hist(phi_centro_gillespie, bins=50, density=True, alpha=0.7, label='Gillespie')
ax.axvline(media_centro, color='b', linestyle='-', label=f'Média = {media_centro:.3f}')
ax.axvline(phi_est_ode[centro], color='r', linestyle='--', label=f'ODE = {phi_est_ode[centro]:.3f}')
ax.set_xlabel('φ')
ax.set_ylabel('densidade')
ax.set_title(f'Flutuações no centro (σ = {flut_centro:.3f})')
ax.legend()
ax.grid(True)

# 4. Perfil de curvatura
ax = axes[1, 1]
ax.plot(x, R, 'k-', linewidth=2)
ax.set_xlabel('posição x (l_P)')
ax.set_ylabel('curvatura R(x)')
ax.set_title('Perfil de curvatura (buraco negro simulado)')
ax.grid(True)

plt.tight_layout()
plt.savefig('IFM_1D.png', dpi=150)
plt.show()

# Salvar dados
np.savez('IFM_1D.npz',
         x=x, t=t_todos, phi_gillespie=phi_todos,
         phi_ode=sol_ode, R=R,
         rms=rms, flut_centro=flut_centro)

print("\nResultados salvos em 'IFM_1D.png' e 'IFM_1D.npz'")