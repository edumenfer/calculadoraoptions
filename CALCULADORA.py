import streamlit as st
import numpy as np
from scipy.stats import norm
import math

# Configuração da calculadora
st.title("Calculadora de Oções")
st.subheader("Eduardo Mendes, Isadora Bueno, Rafael Rezende e Lisa Mandetta")

# Seleciona o tipo de opção
option_type = st.selectbox("Escolha o tipo de opção:", 
                           ("Asiática - Simulação de Monte Carlo", 
                            "Opção Binária", 
                            "Opção Europeia - Black-Scholes", 
                            "Opção Europeia/Americana - Árvore Binomial"))

# Inputs comuns
st.write("### Parâmetros da Opção")
S0 = st.number_input("Preço Inicial do Ativo (S0)", min_value=0.0, value=100.0)
K = st.number_input("Preço de Exercício (K)", min_value=0.0, value=100.0)
r = st.number_input("Taxa de Juros Anual (r) em %", min_value=0.0, value=5.0) / 100
sigma = st.number_input("Volatilidade Anual (%)", min_value=0.0, value=20.0) / 100
T = st.number_input("Tempo até o Vencimento (em anos)", min_value=0.0, value=1.0)
n_simulations = st.number_input("Número de Simulações para Monte Carlo (apenas para opções asiáticas)", min_value=1, value=10000)

# Função de cálculo para opções asiáticas com Simulação de Monte Carlo
def monte_carlo_asian(S, X, T, r, sigma, num_simulations=10000, option_type='call'):
    dt = T / 1000  # Dividindo o tempo em pequenos passos
    payoff_sum = 0
    
    for _ in range(num_simulations):
        path = [S]
        for _ in range(1000):  # 1000 passos na simulação
            S_t = path[-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.randn())
            path.append(S_t)
        
        # Calculando a média do preço
        S_avg = np.mean(path)
        
        # Calculando o payoff
        if option_type == 'call':
            payoff = max(0, S_avg - X)
        elif option_type == 'put':
            payoff = max(0, X - S_avg)
        else:
            raise ValueError("Invalid option type. Use 'call' or 'put'.")
        
        payoff_sum += payoff
    
    # Calculando o preço da opção como o valor médio descontado do payoff
    option_price = (payoff_sum / num_simulations) * np.exp(-r * T)
    return option_price

# Função de cálculo para opção binária
def binary_option(S0, K, r, sigma, T):
    d2 = (np.log(S0 / K) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    call_price = np.exp(-r * T) * norm.cdf(d2)
    return call_price

# Função de cálculo para opções europeias com Black-Scholes
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

# Função de cálculo para árvore binomial (europeias e americanas)
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

# Cálculo com base no tipo de opção selecionado
if option_type == "Asiática - Simulação de Monte Carlo":
    option_kind = st.selectbox("Tipo de Opção", ("call", "put"))
    price = monte_carlo_asian(S0, K, T, r, sigma, num_simulations=n_simulations, option_type=option_kind)
    st.write(f"Preço da Opção Asiática ({option_kind.capitalize()}) - Simulação de Monte Carlo: ${price:.2f}")

elif option_type == "Opção Binária":
    price = binary_option(S0, K, r, sigma, T)
    st.write(f"Preço da Opção Binária: ${price:.2f}")

elif option_type == "Opção Europeia - Black-Scholes":
    option_kind = st.selectbox("Tipo de Opção", ("call", "put"))
    price = black_scholes(S0, K, T, r, sigma, option_type=option_kind)
    st.write(f"Preço da Opção Europeia ({option_kind.capitalize()}): ${price:.2f}")

elif option_type == "Opção Europeia/Americana - Árvore Binomial":
    option_kind = st.selectbox("Tipo de Opção", ("call", "put"))
    exercise_type = st.selectbox("Tipo de Exercício", ("european", "american"))
    steps = st.number_input("Número de Passos na Árvore", min_value=1, value=100)
    price = binomial_tree_option(S0, K, T, r, sigma, steps, option_type=option_kind, exercise=exercise_type)
    st.write(f"Preço da Opção ({exercise_type.capitalize()} - {option_kind.capitalize()}): ${price:.2f}")
