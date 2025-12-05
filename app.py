import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from math import erf, sqrt, exp, pi

# --- Configuração Inicial ---
st.set_page_config(layout="wide", page_title="Simulador Ciclo Motores S/A")

st.title("Otimização de expansão de capacidade - Ciclo Motores S/A")
st.markdown("""
Esta aplicação utiliza **Programação Dinâmica Estocástica** para determinar o plano ótimo de expansão.
O modelo calcula a perda esperada de vendas matematicamente (solução algébrica) e valida os resultados via simulação de Monte Carlo.
""")

# --- Funções Matemáticas Auxiliares ---

def normal_pdf(z):
    """Retorna a Função densidade de robabilidade (PDF) da normal padrão."""
    return (1.0 / sqrt(2 * pi)) * exp(-0.5 * z * z)

def normal_cdf(z):
    """Retorna a Função distribuição acumulada (CDF) da normal padrão."""
    return 0.5 * (1 + erf(z / sqrt(2)))

def calcular_venda_esperada(capacidade, mu, sigma):
    """
    Calcula algebricamente o valor esperado das vendas: E[min(Demanda, Capacidade)].
    Utiliza a 'Linear Loss Function' da distribuição Normal.
    """
    if sigma == 0:
        return min(capacidade, mu)

    z = (capacidade - mu) / sigma

    # Termo 1: Contribuição quando Demanda < Capacidade
    term1 = mu * normal_cdf(z) - sigma * normal_pdf(z)
    
    # Termo 2: Contribuição quando Demanda >= Capacidade
    term2 = capacidade * (1 - normal_cdf(z))
    
    return term1 + term2

def simular_venda_monte_carlo(capacidade, mu, sigma, n_simulacoes=10000):
    """
    Executa simulação de Monte Carlo para validar o cálculo algébrico.
    """
    if sigma == 0:
        return min(capacidade, mu)
    
    demandas = np.random.normal(mu, sigma, n_simulacoes)
    demandas = np.maximum(demandas, 0) 
    vendas_reais = np.minimum(demandas, capacidade)
    
    return np.mean(vendas_reais)

# --- Interface de Parâmetros ---

st.sidebar.header("1. Parâmetros econômicos")
tma_percent = st.sidebar.number_input("TMA (% ao ano)", value=9.3, step=0.1)
tma = tma_percent / 100
lucro_unit = st.sidebar.number_input("Lucro por unidade (US$)", value=1088.0, step=10.0)
cap_inicial = st.sidebar.number_input("Capacidade inicial", value=150000, step=10000)

st.sidebar.header("2. Opções de expansão")
st.sidebar.info("Custo de investimento para cada degrau de expansão:")

df_opcoes_default = pd.DataFrame({
    "Expansão": [0, 50000, 100000, 150000, 200000],
    "Investimento ($)": [0, 100_000_000, 190_000_000, 270_000_000, 350_000_000]
})
df_opcoes = st.sidebar.data_editor(df_opcoes_default, num_rows="dynamic", hide_index=True)
opcoes_expansao = dict(zip(df_opcoes["Expansão"], df_opcoes["Investimento ($)"]))

st.subheader("3. Previsão de demanda estocástica")
st.markdown("Defina a média (µ) e o desvio padrão (σ) da demanda para cada ano.")

dados_demanda_default = pd.DataFrame({
    'Ano': range(1, 10 + 1),
    'Media': [140000, 182000, 231100, 281100, 323600, 352900, 372100, 384300, 391800, 396400],
    'StdDev': [13300, 17300, 22000, 26700, 30700, 33500, 35400, 36500, 37200, 37700]
})
df_demanda = st.data_editor(dados_demanda_default, num_rows="dynamic", use_container_width=True)

# --- Algoritmo de Programação Dinâmica ---

def resolver_dp(anos_df, opcoes_dict, tma, lucro_un, cap_init):
    """
    Resolve o problema de otimização utilizando Programação Dinâmica (Backward Induction).
    """
    demanda_dict = anos_df.set_index('Ano').to_dict('index')
    max_ano = anos_df['Ano'].max()
    max_demanda = anos_df['Media'].max() + 6 * anos_df['StdDev'].max()
    
    passo_cap = 50000 
    estados_possiveis = [cap_init + i * passo_cap for i in range(int((max_demanda - cap_init) / passo_cap) + 6)]
    
    memo = {}

    # Backward Induction
    for ano in range(max_ano, 0, -1):
        memo[ano] = {}
        mu = demanda_dict[ano]['Media']
        sigma = demanda_dict[ano]['StdDev']
        
        for cap_atual in estados_possiveis:
            melhor_vpl = -np.inf
            melhor_decisao = None
            
            # Receita Operacional Esperada
            vendas_esperadas = calcular_venda_esperada(cap_atual, mu, sigma)
            fluxo_operacional = vendas_esperadas * lucro_un
            
            # Avaliação das decisões
            for expansao, investimento in opcoes_dict.items():
                cap_proximo = cap_atual + expansao
                
                if ano == max_ano:
                    # Valor Terminal (Perpetuidade)
                    vendas_term = calcular_venda_esperada(cap_proximo, mu, sigma)
                    fluxo_perpetuo = vendas_term * lucro_un
                    valor_futuro = fluxo_perpetuo / tma
                else:
                    # Recuperação do valor futuro na tabela
                    valor_futuro = memo.get(ano + 1, {}).get(cap_proximo, (0,))[0]
                
                # Equação de Bellman
                vpl_decisao = fluxo_operacional - investimento + valor_futuro / (1 + tma)
                
                if vpl_decisao > melhor_vpl:
                    melhor_vpl = vpl_decisao
                    melhor_decisao = expansao
            
            memo[ano][cap_atual] = (melhor_vpl, melhor_decisao)
    
    return memo

def reconstruir_plano(memo, cap_init, max_ano, opcoes_dict, demanda_df, lucro_un):
    """
    Reconstrói a trajetória ótima (Forward Pass).
    """
    plano = []
    cap_atual = cap_init
    demanda_dict = demanda_df.set_index('Ano').to_dict('index')
    
    for ano in range(1, max_ano + 1):
        decisao = memo[ano][cap_atual][1]
        custo = opcoes_dict[decisao]
        
        mu = demanda_dict[ano]['Media']
        sigma = demanda_dict[ano]['StdDev']
        
        vendas_esp = calcular_venda_esperada(cap_atual, mu, sigma)
        receita = vendas_esp * lucro_un
        
        plano.append({
            "Ano": ano,
            "Capacidade Atual": cap_atual,
            "Demanda Média": mu,
            "Desvio Padrão": sigma,
            "Decisão (Expansão)": decisao,
            "Investimento": custo,
            "Receita Esperada": receita,
            "Fluxo Líquido": receita - custo,
            "Nova Capacidade": cap_atual + decisao
        })
        
        cap_atual += decisao
    
    return pd.DataFrame(plano)

# --- Execução e Visualização ---

if st.button("Calcular plano ótimo", type="primary"):
    with st.spinner('Processando otimização e validação...'):
        # 1. Resolução
        memo_table = resolver_dp(df_demanda, opcoes_expansao, tma, lucro_unit, cap_inicial)
        vpl_total = memo_table[1][cap_inicial][0]
        
        # 2. Reconstrução
        df_resultado = reconstruir_plano(memo_table, cap_inicial, df_demanda['Ano'].max(), opcoes_expansao, df_demanda, lucro_unit)
        
        # 3. Resultados
        st.success("Otimização concluída!")
        st.metric(label="Valor presente líquido (VPL) total", value=f"US$ {vpl_total:,.2f}")
        
        st.subheader("Plano estratégico detalhado")
        st.dataframe(df_resultado.style.format({
            "Capacidade Atual": "{:,.0f}",
            "Demanda Média": "{:,.0f}",
            "Desvio Padrão": "{:,.0f}",
            "Decisão (Expansão)": "{:,.0f}",
            "Investimento": "US$ {:,.0f}",
            "Receita Esperada": "US$ {:,.0f}",
            "Fluxo Líquido": "US$ {:,.0f}",
            "Nova Capacidade": "{:,.0f}"
        }), use_container_width=True)

        st.subheader("Trajetória de capacidade vs. demanda")
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df_resultado['Ano'], 
            y=df_resultado['Capacidade Atual'],
            mode='lines+markers',
            name='Capacidade instalada',
            line=dict(color='green', width=4)
        ))
        
        fig.add_trace(go.Scatter(
            x=df_demanda['Ano'], 
            y=df_demanda['Media'],
            mode='lines',
            name='Demanda média',
            line=dict(color='blue', dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            x=pd.concat([df_demanda['Ano'], df_demanda['Ano'][::-1]]),
            y=pd.concat([df_demanda['Media'] + df_demanda['StdDev'], (df_demanda['Media'] - df_demanda['StdDev'])[::-1]]),
            fill='toself',
            fillcolor='rgba(0,0,255,0.1)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            name='Incerteza (±1σ)'
        ))
        
        st.plotly_chart(fig, use_container_width=True)

        # --- Validação ---
        st.markdown("---")
        st.subheader("Validação do modelo (algébrico vs. simulação)")
        st.markdown("""
        **Verificação de robustez:** O gráfico abaixo compara três abordagens para o cálculo de vendas esperadas:
        1.  **Ingênua (vermelho):** Assume $E[V] = \min(\mu, Q)$. Superestima receitas em cenários de incerteza.
        2.  **Algébrica (verde):** Cálculo exato $E[\min(D, Q)]$ utilizado no modelo.
        3.  **Simulação de Monte Carlo (azul):** Validação empírica.
        """)

        ano_exemplo = 1
        mu_ex = df_demanda.loc[df_demanda['Ano'] == ano_exemplo, 'Media'].values[0]
        sigma_ex = df_demanda.loc[df_demanda['Ano'] == ano_exemplo, 'StdDev'].values[0]
        
        caps_teste = np.linspace(mu_ex - 2.5*sigma_ex, mu_ex + 2.5*sigma_ex, 50)
        vendas_algebricas = [calcular_venda_esperada(c, mu_ex, sigma_ex) for c in caps_teste]
        vendas_ingenuas = [min(c, mu_ex) for c in caps_teste]
        
        caps_sim = np.linspace(mu_ex - 2.5*sigma_ex, mu_ex + 2.5*sigma_ex, 15)
        vendas_sim = [simular_venda_monte_carlo(c, mu_ex, sigma_ex) for c in caps_sim]
        
        fig_val = go.Figure()
        
        fig_val.add_trace(go.Scatter(
            x=caps_teste, y=vendas_ingenuas,
            mode='lines', name='Abordagem ingênua (incorreta)',
            line=dict(color='red', dash='dot', width=2)
        ))
        
        fig_val.add_trace(go.Scatter(
            x=caps_teste, y=vendas_algebricas,
            mode='lines', name='Cálculo algébrico (modelo)',
            line=dict(color='green', width=4)
        ))
        
        fig_val.add_trace(go.Scatter(
            x=caps_sim, y=vendas_sim,
            mode='markers', name='Validação Monte Carlo',
            marker=dict(color='blue', size=10, symbol='x')
        ))

        fig_val.update_layout(
            title=f"Curva de vendas esperadas: Ano {ano_exemplo} (µ={mu_ex:,.0f}, σ={sigma_ex:,.0f})",
            xaxis_title="Capacidade instalada",
            yaxis_title="Vendas esperadas (unidades)",
            hovermode="x unified",
            height=500
        )
        
        st.plotly_chart(fig_val, use_container_width=True)
        st.info("Nota: A convergência entre a curva algébrica e a simulação de Monte Carlo confirma a precisão matemática do modelo.")

else:
    st.info("Configure os parâmetros acima e clique no botão para gerar o plano ótimo.")