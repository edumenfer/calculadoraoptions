import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
import math

# Configuração da calculadora
st.title("Calculadora de Opções")
st.subheader("Eduardo Mendes, Isadora Bueno, Rafael Rezende e Lisa Mandetta")

# Botão de seleção para escolher o tipo de cálculo
option_type = st.radio("Escolha o tipo de cálculo:", 
                       ("Asiática - Simulação de Monte Carlo", 
                        "Opção Binária", 
                        "Opção Europeia - Black-Scholes", 
                        "Opção Europeia/Americana - Árvore Binomial"))

# Seção de parâmetros principais (lado da página)
st.write("### Parâmetros Principais")
col1, col2 = st.columns(2)

with col1:
    S0 = st.number_input("Preço Inicial do Ativo (S0)", min_value=0.0, value=100.0)
    K = st.number_input("Preço de Exercício (K)", min_value=0.0, value=100.0)

with col2:
    r = st.number_input("Taxa de Juros Anual (r) em %", min_value=0.0, value=5.0) / 100
    sigma = st.number_input("Volatilidade Anual (%)", min_value=0.0, value=20.0) / 100
    T = st.number_input("Tempo até o Vencimento (em anos)", min_value=0.0, value=1.0)

# Seção de parâmetros específicos para cálculos
st.write("### Parâmetros de Cálculo")

# Seção para a simulação de Monte Carlo (para opções asiáticas)
if option_type == "Asiática - Simulação de Monte Carlo":
    st.write("### Parâmetros da Simulação de Monte Carlo")
    n_simulations = st.number_input("Número de Simulações", min_value=1, value=10000)
    
# Função para plotar gráficos interativos com Plotly
def plot_simulation(paths, title):
    fig = go.Figure()
    for path in paths:
        fig.add_trace(go.Scatter(x=np.arange(len(path)), y=path, mode='lines', opacity=0.5))
    fig.update_layout(
        title=title,
        xaxis_title="Passos",
        yaxis_title="Preço do Ativo",
        showlegend=False
    )
    st.plotly_chart(fig)  # Exibe o gráfico no Streamlit

# Função para opções asiáticas com Monte Carlo
def monte_carlo_asian(S, X, T, r, sigma, num_simulations=10000, option_type='call'):
    dt = T / 1000
    paths = []
    payoff_sum = 0
    
    for _ in range(num_simulations):
        path = [S]
        for _ in range(1000):
            S_t = path[-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.randn())
            path.append(S_t)
        paths.append(path)
        S_avg = np.mean(path)
        payoff = max(0, S_avg - X) if option_type == 'call' else max(0, X - S_avg)
        payoff_sum += payoff
    
    option_price = (payoff_sum / num_simulations) * np.exp(-r * T)
    return option_price, paths

# Função para opção binária
def binary_option(S0, K, r, sigma, T):
    d2 = (np.log(S0 / K) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    call_price = np.exp(-r * T) * norm.cdf(d2)
    return call_price

# Função para opções europeias com Black-Scholes
def black_scholes(S, X, T, r, sigma, option_type='call'):
    d1 = (np.log(S / X) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        price = S * norm.cdf(d1) - X * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        price = X * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")
    return price

# Função para árvore binomial (europeias e americanas)
def binomial_tree_option(S, X, T, r, sigma, N, option_type='call', exercise='european'):
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    ST = np.zeros(N + 1)
    for j in range(N + 1):
        ST[j] = S * (u ** j) * (d ** (N - j))
    if option_type == 'call':
        option_values = np.maximum(0, ST - X)
    else:
        option_values = np.maximum(0, X - ST)
    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            option_values[j] = np.exp(-r * dt) * (p * option_values[j + 1] + (1 - p) * option_values[j])
            if exercise == 'american':
                if option_type == 'call':
                    option_values[j] = max(option_values[j], S * (u ** j) * (d ** (i - j)) - X)
                else:
                    option_values[j] = max(option_values[j], X - S * (u ** j) * (d ** (i - j)))
    return option_values[0]

# Cálculo baseado na seleção
if option_type == "Asiática - Simulação de Monte Carlo":
    option_kind = st.selectbox("Tipo de Opção", ("call", "put"))
    price, paths = monte_carlo_asian(S0, K, T, r, sigma, num_simulations=n_simulations, option_type=option_kind)
    st.write(f"Preço da Opção Asiática ({option_kind.capitalize()}): ${price:.2f}")
    plot_simulation(paths[:10], "Simulações de Monte Carlo (10 Caminhos)")

elif option_type == "Opção Binária":
    price = binary_option(S0, K, r, sigma, T)
    st.write(f"Preço da Opção Binária: ${price:.2f}")
    
    # Gerando gráfico do preço da opção binária com Plotly
    fig = go.Figure()
    x = np.linspace(0, 2 * K, 100)
    y = np.exp(-r * T) * norm.cdf((np.log(x / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T)))
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Preço da Opção Binária'))
    fig.update_layout(
        title="Gráfico de Preço da Opção Binária",
        xaxis_title="Preço do Ativo Subjacente",
        yaxis_title="Preço da Opção Binária"
    )
    st.plotly_chart(fig)

elif option_type == "Opção Europeia - Black-Scholes":
    option_kind = st.selectbox("Tipo de Opção", ("call", "put"))
    price = black_scholes(S0, K, T, r, sigma, option_type=option_kind)
    st.write(f"Preço da Opção Europeia ({option_kind.capitalize()}): ${price:.2f}")
    
    # Gerando gráfico do preço da opção européia Black-Scholes com Plotly
    fig = go.Figure()
    x = np.linspace(0, 2 * K, 100)
    y_call = black_scholes(x, K, T, r, sigma, option_type='call')
    y_put = black_scholes(x, K, T, r, sigma, option_type='put')
    fig.add_trace(go.Scatter(x=x, y=y_call, mode='lines', name="Call", line=dict(color='green')))
    fig.add_trace(go.Scatter(x=x, y=y_put, mode='lines', name="Put", line=dict(color='red')))
    fig.update_layout(
        title="Preço da Opção Europeia - Black-Scholes",
        xaxis_title="Preço do Ativo Subjacente",
        yaxis_title="Preço da Opção",
        legend_title="Tipo de Opção"
    )
    st.plotly_chart(fig)

elif option_type == "Opção Europeia/Americana - Árvore Binomial":
    option_kind = st.selectbox("Tipo de Opção", ("call", "put"))
    N = st.number_input("Número de Passos da Árvore Binomial", min_value=1, value=100)
    exercise_type = st.selectbox("Tipo de Exercício", ("european", "american"))
    price = binomial_tree_option(S0, K, T, r, sigma, N, option_type=option_kind, exercise=exercise_type)
    st.write(f"Preço da Opção Binomial ({exercise_type.capitalize()}): ${price:.2f}")
