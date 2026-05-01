# IFM_2D_hubble.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import time

# ============================================================
# PARÂMETROS DO UNIVERSO SIMULADO (AJUSTÁVEIS)
# ============================================================
Nx, Ny = 20, 20
Γ0 = 0.5
α = -4.0
Γh = 4.0
t_max = 200.0                      # tempo de simulação (mais longo para alcançar "hoje")
dt_amostra = 1.0
R0 = 0.25                          # curvatura base
Λ = 1.0                            # PRESSÃO EXPANSIVA (reduzida de 5.0)
η = 0.01                           # DILUIÇÃO (reduzida de 0.1)

np.random.seed(42)

# ============================================================
# ESTADO INICIAL HETEROGÊNEO (SEMENTE DE BUraco negro)
# ============================================================
n = np.zeros((Nx, Ny), dtype=np.int8)
centro = Nx // 2
raio_semente = 5

for i in range(Nx):
    for j in range(Ny):
        r2 = (i - centro)**2 + (j - centro)**2
        if r2 < raio_semente**2:
            # Centro: 95% ocupado (semente)
            if np.random.random() < 0.95:
                n[i, j] = 1 if np.random.random() < 0.5 else -1
        else:
            # Bordas: 30% ocupado (vácuo)
            if np.random.random() < 0.3:
                n[i, j] = 1 if np.random.random() < 0.5 else -1

print("="*70)
print("TESTE COMPLETO: TENSÃO DE HUBBLE NA IFM")
print("="*70)
print(f"Grade: {Nx}×{Ny} = {Nx*Ny} sítios")
print(f"t_max = {t_max} (ajustado para alcançar 'hoje' cósmico)")
print(f"R0 = {R0}, α = {α}, Γh = {Γh}")
print(f"Λ = {Λ} (pressão expansiva), η = {η} (diluição)")

# ============================================================
# FUNÇÕES
# ============================================================
def atualizar_curvatura(n, R0):
    phi = 2 * (np.abs(n) > 0).astype(float) - 1
    R = R0 * (1 - phi) / 2
    return np.maximum(R, 0.01)

def obter_eventos(n, R, Γ0, α, Γh, Λ, centro):
    eventos = []
    Nx, Ny = n.shape
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
                        r2 = (i - centro)**2 + (j - centro)**2
                        f_exp = 1 + Λ * np.exp(-r2 / (Nx*Ny/4))
                        eventos.append(('hop', i, j, Γh * f_exp, ni, nj))
    return eventos

def diluir_expansao(n, η, dt):
    for i in range(n.shape[0]):
        for j in range(n.shape[1]):
            if n[i, j] != 0:
                if np.random.random() < η * dt:
                    n[i, j] = 0

# ============================================================
# SIMULAÇÃO
# ============================================================
n_atual = n.copy()
R_atual = atualizar_curvatura(n_atual, R0)
eventos = obter_eventos(n_atual, R_atual, Γ0, α, Γh, Λ, centro)

t = 0.0
phi_hist = []
pressure_hist = []
t_hist = []
prox_amostra = 0.0
ultima_diluicao = 0
dt_diluicao = 0.5

start = time.time()

while t < t_max and eventos:
    R_atual = atualizar_curvatura(n_atual, R0)
    eventos = obter_eventos(n_atual, R_atual, Γ0, α, Γh, Λ, centro)
    
    taxa_total = sum(e[3] for e in eventos)
    
    if t - ultima_diluicao > dt_diluicao:
        diluir_expansao(n_atual, η, dt_diluicao)
        ultima_diluicao = t
    
    if taxa_total <= 0:
        break
    
    dt = -np.log(np.random.random()) / taxa_total
    t += dt
    
    while t >= prox_amostra:
        phi = 2 * (np.abs(n_atual) > 0).astype(float) - 1
        phi_hist.append(phi.flatten())
        
        # Pressão expansiva (energia escura)
        pressao = Λ * np.mean(R_atual) + η * np.mean(np.abs(n_atual) > 0)
        pressure_hist.append(pressao)
        
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

print(f"\nSimulação concluída em {time.time()-start:.2f}s")
print(f"Amostras coletadas: {len(t_hist)}")

if len(t_hist) == 0:
    print("Erro: nenhuma amostra coletada.")
    exit()

# ============================================================
# ANÁLISE: TENSÃO DE HUBBLE
# ============================================================
# Ajuste exponencial da pressão expansiva
tempo = np.array(t_hist)
pressao = np.array(pressure_hist)

# Usar apenas parte estável da curva (após transiente)
inicio = len(tempo) // 4
tempo_fit = tempo[inicio:]
pressao_fit = pressao[inicio:]

def exp_func(t, p0, tau):
    return p0 * np.exp(t / tau)

try:
    # Tentativa de ajuste exponencial
    popt, _ = curve_fit(exp_func, tempo_fit, pressao_fit, p0=(1.0, 50.0), maxfev=5000)
    p0_fit, tau_fit = popt
    
    # Extrapolar H_hoje / H_CMB
    # Assumindo: H² ∝ pressão expansiva
    # H_CMB corresponde ao início do ajuste (t = tempo_fit[0])
    # H_hoje corresponde ao final (t = tempo_fit[-1])
    p_cmb = exp_func(tempo_fit[0], p0_fit, tau_fit)
    p_hoje = exp_func(tempo_fit[-1], p0_fit, tau_fit)
    H_ratio = np.sqrt(p_hoje / p_cmb)
    
    print("\n" + "="*70)
    print("RESULTADO: TENSÃO DE HUBBLE NA IFM")
    print("="*70)
    print(f"Ajuste exponencial: p(t) = {p0_fit:.3f} * exp(t / {tau_fit:.1f})")
    print(f"Pressão no início (época do CMB): {p_cmb:.3f}")
    print(f"Pressão hoje (fim da simulação): {p_hoje:.3f}")
    print(f"\nRAZÃO H(hoje) / H(CMB) = {H_ratio:.4f}")
    print(f"Valor observado (Planck vs Local): 73/67 = {73/67:.4f}")
    
    # Verificar se a razão está na faixa observada
    if 1.05 < H_ratio < 1.15:
        print("\nPREDIÇÃO CONFIRMADA: a IFM reproduz a tensão de Hubble!")
    elif H_ratio < 1.05:
        print("\nPREDIÇÃO PARCIAL: a tensão é menor que a observada. Ajuste Λ e η.")
    else:
        print("\nPREDIÇÃO EXCESSIVA: a tensão é maior que a observada. Reduza Λ e η.")
        
except Exception as e:
    print(f"\nErro no ajuste exponencial: {e}")
    print("Usando razão direta entre primeiro e último ponto.")
    p_cmb = pressao[inicio]
    p_hoje = pressao[-1]
    H_ratio = np.sqrt(p_hoje / p_cmb)
    print(f"Razão H(hoje)/H(CMB) = {H_ratio:.4f} (direta)")

# ============================================================
# VISUALIZAÇÃO
# ============================================================
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

# Gráfico 1: Pressão expansiva (energia escura) vs tempo
ax[0].plot(tempo, pressao, 'purple', label='Pressão (simulação)')
ax[0].plot(tempo_fit, exp_func(tempo_fit, *popt), 'r--', label=f'Ajuste exponencial: τ={tau_fit:.1f}')
ax[0].set_xlabel('Tempo de processamento (t_P)')
ax[0].set_ylabel('Pressão expansiva (energia escura)')
ax[0].set_title('Evolução da Energia Escura na IFM')
ax[0].legend()
ax[0].grid(True)

# Gráfico 2: Comparação com a tensão de Hubble
ax[1].bar(['IFM (predição)', 'Observado (local/CMB)'], [H_ratio, 73/67], color=['purple', 'orange'])
ax[1].axhline(y=1, color='gray', linestyle='--', label='Universo sem tensão (H₀ constante)')
ax[1].set_ylabel('H(hoje) / H(CMB)')
ax[1].set_title('Tensão de Hubble: IFM vs Observação')
ax[1].legend()
ax[1].grid(True, axis='y')

plt.tight_layout()
plt.savefig('IFM_2D_hubble.py.png', dpi=120)
plt.show()

print("\nFigura salva como 'IFM_2D_hubble.py.png'")