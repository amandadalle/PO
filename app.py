import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from math import erf, sqrt, exp, pi

# ==========================================
# CONFIGURAÇÃO INICIAL DA PÁGINA
# ==========================================
st.set_page_config(layout="wide", page_title="Simulador Ciclo Motores S/A")

st.title("Otimização de expansão de capacidade - Ciclo Motores S/A")
st.markdown("""
Esta aplicação utiliza **Programação Dinâmica Estocástica** para determinar o plano ótimo de expansão.
O modelo implementa a **Linear Loss Function** para cálculo de vendas e inclui o **Ajuste de Meio de Período (Mid-Year Adjustment)** para a perpetuidade.
""")

# ==========================================
# 1. FUNÇÕES MATEMÁTICAS AUXILIARES
# ==========================================

def normal_pdf(z):
    """Retorna a Função Densidade de Probabilidade (PDF) da normal padrão."""
    return (1.0 / sqrt(2 * pi)) * exp(-0.5 * z * z)

def normal_cdf(z):
    """Retorna a Função Distribuição Acumulada (CDF) da normal padrão."""
    return 0.5 * (1 + erf(z / sqrt(2)))

def calcular_venda_esperada(capacidade, mu, sigma):
    """
    Calcula algebricamente o valor esperado das vendas: E[min(Demanda, Capacidade)].
    Fórmula: mu*Phi(z) - sigma*phi(z) + Q*(1-Phi(z))
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
    demandas = np.maximum(demandas, 0) # Demanda não pode ser negativa
    vendas_reais = np.minimum(demandas, capacidade)
    
    return np.mean(vendas_reais)

# ==========================================
# 2. INTERFACE DE PARÂMETROS
# ==========================================

st.sidebar.header("1. Parâmetros Econômicos")

tma_percent = st.sidebar.number_input("TMA (% ao ano)", value=9.3, step=0.1, help="Taxa Mínima de Atratividade")
tma = tma_percent / 100

lucro_unit = st.sidebar.number_input("Lucro por unidade (US$)", value=1088.0, step=10.0, help="Margem de contribuição unitária")
cap_inicial = st.sidebar.number_input("Capacidade Inicial", value=150000, step=10000)

st.sidebar.header("2. Opções de Expansão")
st.sidebar.info("Custo de investimento para cada degrau de expansão:")

# Tabela de custos padrão
df_opcoes_default = pd.DataFrame({
    "Expansão": [0, 50000, 100000, 150000, 200000],
    "Investimento ($)": [0, 100_000_000, 190_000_000, 270_000_000, 350_000_000]
})
df_opcoes = st.sidebar.data_editor(df_opcoes_default, num_rows="dynamic", hide_index=True)
opcoes_expansao = dict(zip(df_opcoes["Expansão"], df_opcoes["Investimento ($)"]))

st.subheader("3. Previsão de demanda estocástica")
st.markdown("Defina a média (µ) e o desvio padrão (σ) da demanda para cada ano.")

# Dados padrão do problema
dados_demanda_default = pd.DataFrame({
    'Ano': range(1, 10 + 1),
    'Media': [140000, 182000, 231100, 281100, 323600, 352900, 372100, 384300, 391800, 396400],
    'StdDev': [13300, 17300, 22000, 26700, 30700, 33500, 35400, 36500, 37200, 37700]
})
df_demanda = st.data_editor(dados_demanda_default, num_rows="dynamic", use_container_width=True)

# ==========================================
# 3. LÓGICA DE PROGRAMAÇÃO DINÂMICA
# ==========================================

def resolver_dp(anos_df, opcoes_dict, tma, lucro_un, cap_init):
    """
    Resolve o problema utilizando Backward Induction.
    Inclui Fator de Ajuste Mid-Year para a Perpetuidade.
    """
    demanda_dict = anos_df.set_index('Ano').to_dict('index')
    max_ano = anos_df['Ano'].max()
    # Define limite superior para o espaço de estados (média + 6 sigmas)
    max_demanda = anos_df['Media'].max() + 6 * anos_df['StdDev'].max()
    
    # --- Fator Mid-Year ---
    # Assume que os fluxos ocorrem ao longo do ano, aumentando o VP
    fator_mid_year = (1 + tma)**0.5
    
    passo_cap = 50000 
    # Cria espaço de estados discretos
    estados_possiveis = [cap_init + i * passo_cap for i in range(int((max_demanda - cap_init) / passo_cap) + 6)]
    
    memo = {} # Tabela de memorização

    # Backward Induction (Do futuro para o presente)
    for ano in range(max_ano, 0, -1):
        memo[ano] = {}
        mu = demanda_dict[ano]['Media']
        sigma = demanda_dict[ano]['StdDev']
        
        for cap_atual in estados_possiveis:
            melhor_vpl = -np.inf
            melhor_decisao = None
            
            # 1. Fluxo Operacional do Ano Corrente
            vendas_esperadas = calcular_venda_esperada(cap_atual, mu, sigma)
            fluxo_operacional = vendas_esperadas * lucro_un
            
            # 2. Avaliar Decisões de Expansão
            for expansao, investimento in opcoes_dict.items():
                cap_proximo = cap_atual + expansao
                
                if ano == max_ano:
                    # --- CÁLCULO DA PERPETUIDADE ---
                    # Usa a demanda do último ano estabilizada
                    vendas_term = calcular_venda_esperada(cap_proximo, mu, sigma)
                    fluxo_perpetuo = vendas_term * lucro_un
                    
                    # Perpetuidade (Fluxo/i) ajustada pelo fator Mid-Year
                    valor_futuro = (fluxo_perpetuo / tma) * fator_mid_year
                else:
                    # Recupera valor da tabela DP (se o estado existir, senão 0)
                    valor_futuro = memo.get(ano + 1, {}).get(cap_proximo, (0,))[0]
                
                # Equação de Bellman
                # VPL = Fluxo Op - Investimento + VP do Futuro
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
        # Recupera a melhor decisão armazenada na tabela memo
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
        
        # Atualiza o estado para o próximo ano
        cap_atual += decisao
    
    return pd.DataFrame(plano)

# ==========================================
# 4. EXECUÇÃO E VISUALIZAÇÃO
# ==========================================

if st.button("Calcular Plano Ótimo", type="primary"):
    with st.spinner('Executando Otimização Estocástica...'):
        # 1. Resolução (Backward)
        memo_table = resolver_dp(df_demanda, opcoes_expansao, tma, lucro_unit, cap_inicial)
        vpl_total = memo_table[1][cap_inicial][0]
        
        # 2. Reconstrução (Forward)
        df_resultado = reconstruir_plano(memo_table, cap_inicial, df_demanda['Ano'].max(), opcoes_expansao, df_demanda, lucro_unit)
        
        # 3. Exibição dos Resultados Principais
        st.success("Otimização concluída com sucesso!")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="VPL Total do Projeto", value=f"US$ {vpl_total:,.2f}")
        with col2:
            st.info("O cálculo do VPL inclui o valor da perpetuidade ajustado pelo fator de meio de período.")
        
        st.subheader("Plano Estratégico Detalhado")
        # Formatação da tabela para exibição
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

        # 4. Gráfico de Trajetória
        st.subheader("Evolução da Capacidade vs. Demanda")
        fig = go.Figure()

        # Linha da Capacidade
        fig.add_trace(go.Scatter(
            x=df_resultado['Ano'], 
            y=df_resultado['Capacidade Atual'],
            mode='lines+markers',
            name='Capacidade Instalada',
            line=dict(color='green', width=4)
        ))
        
        # Linha da Demanda Média
        fig.add_trace(go.Scatter(
            x=df_demanda['Ano'], 
            y=df_demanda['Media'],
            mode='lines',
            name='Demanda Média',
            line=dict(color='blue', dash='dash')
        ))
        
        # Faixa de Incerteza (Intervalo de Confiança +/- 1 Sigma)
        fig.add_trace(go.Scatter(
            x=pd.concat([df_demanda['Ano'], df_demanda['Ano'][::-1]]),
            y=pd.concat([df_demanda['Media'] + df_demanda['StdDev'], (df_demanda['Media'] - df_demanda['StdDev'])[::-1]]),
            fill='toself',
            fillcolor='rgba(0,0,255,0.1)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            name='Incerteza (±1σ)'
        ))
        
        fig.update_layout(xaxis_title="Ano", yaxis_title="Unidades", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        # 5. Validação (Monte Carlo)
        st.markdown("---")
        st.subheader("Validação do Modelo Matemático")
        st.markdown("""
        O gráfico abaixo compara a precisão do modelo algébrico (usado na otimização) contra:
        1.  **Abordagem Ingênua:** $E[V] = \min(\mu, Q)$ (Ignora variabilidade, tende a superestimar).
        2.  **Simulação de Monte Carlo:** Validação empírica via amostragem aleatória.
        """)

        # Dados para validação (usando Ano 1 como exemplo)
        ano_exemplo = 1
        mu_ex = df_demanda.loc[df_demanda['Ano'] == ano_exemplo, 'Media'].values[0]
        sigma_ex = df_demanda.loc[df_demanda['Ano'] == ano_exemplo, 'StdDev'].values[0]
        
        # Gera pontos para as curvas
        caps_teste = np.linspace(mu_ex - 2.5*sigma_ex, mu_ex + 2.5*sigma_ex, 50)
        vendas_algebricas = [calcular_venda_esperada(c, mu_ex, sigma_ex) for c in caps_teste]
        vendas_ingenuas = [min(c, mu_ex) for c in caps_teste]
        
        caps_sim = np.linspace(mu_ex - 2.5*sigma_ex, mu_ex + 2.5*sigma_ex, 15)
        vendas_sim = [simular_venda_monte_carlo(c, mu_ex, sigma_ex) for c in caps_sim]
        
        fig_val = go.Figure()
        
        fig_val.add_trace(go.Scatter(
            x=caps_teste, y=vendas_ingenuas,
            mode='lines', name='Ingênua (Incorreta)',
            line=dict(color='red', dash='dot', width=2)
        ))
        
        fig_val.add_trace(go.Scatter(
            x=caps_teste, y=vendas_algebricas,
            mode='lines', name='Algébrica (Modelo)',
            line=dict(color='green', width=4)
        ))
        
        fig_val.add_trace(go.Scatter(
            x=caps_sim, y=vendas_sim,
            mode='markers', name='Simulação Monte Carlo',
            marker=dict(color='blue', size=10, symbol='x')
        ))

        fig_val.update_layout(
            title=f"Curva de Vendas Esperadas: Ano {ano_exemplo} (µ={mu_ex:,.0f}, σ={sigma_ex:,.0f})",
            xaxis_title="Capacidade Instalada",
            yaxis_title="Vendas Esperadas (unidades)",
            hovermode="x unified",
            height=500
        )
        
        st.plotly_chart(fig_val, use_container_width=True)

else:
    st.info("Ajuste os parâmetros na barra lateral se necessário e clique no botão para calcular.")