import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(layout="wide", page_title="Simulador Ciclo Motores S/A")

st.title("Otimização de Expansão de Capacidade - Ciclo Motores S/A")
st.markdown("""
Esta aplicação utiliza **Programação Dinâmica Estocástica** para determinar o plano ótimo de expansão, 
considerando a incerteza da demanda (Distribuição Normal) e parâmetros econômicos editáveis.
""")

# --- BARRA LATERAL (PARÂMETROS GLOBAIS) ---
st.sidebar.header("1. Parâmetros Econômicos")

tma_percent = st.sidebar.number_input("TMA (% ao ano)", value=9.3, step=0.1)
tma = tma_percent / 100

lucro_unit = st.sidebar.number_input("Lucro por Unidade (US$)", value=1088.0, step=10.0)
cap_inicial = st.sidebar.number_input("Capacidade Inicial", value=150000, step=10000)

st.sidebar.header("2. Opções de Expansão")
st.sidebar.info("Edite os custos na tabela abaixo:")

# Tabela editável de Custos de Expansão
df_opcoes_default = pd.DataFrame({
    "Expansão": [0, 50000, 100000, 150000, 200000],
    "Investimento ($)": [0, 100_000_000, 190_000_000, 270_000_000, 350_000_000]
})
df_opcoes = st.sidebar.data_editor(df_opcoes_default, num_rows="dynamic", hide_index=True)

# Converter DataFrame de opções para Dicionário para usar na lógica
opcoes_expansao = dict(zip(df_opcoes["Expansão"], df_opcoes["Investimento ($)"]))

# --- ÁREA PRINCIPAL (TABELA DE DEMANDA) ---
st.subheader("3. Previsão de Demanda (Média e Desvio Padrão)")
st.markdown("Edite os valores abaixo para simular diferentes cenários de mercado.")

# Dados iniciais do problema
dados_demanda_default = pd.DataFrame({
    'Ano': range(1, 11),
    'Media': [140000, 182000, 231100, 281100, 323600, 352900, 372100, 384300, 391800, 396400],
    'StdDev': [13300, 17300, 22000, 26700, 30700, 33500, 35400, 36500, 37200, 37700]
})

df_demanda = st.data_editor(dados_demanda_default, num_rows="dynamic", use_container_width=True)

# --- FUNÇÕES DE CÁLCULO ---

def calcular_venda_esperada(capacidade, mu, sigma):
    if sigma == 0:
        return min(capacidade, mu)
    z = (capacidade - mu) / sigma
    term1 = mu * norm.cdf(z) - sigma * norm.pdf(z)
    term2 = capacidade * (1 - norm.cdf(z))
    return term1 + term2

def resolver_dp(anos_df, opcoes_dict, tma, lucro_un, cap_init):
    demanda_dict = anos_df.set_index('Ano').to_dict('index')
    max_ano = anos_df['Ano'].max()
    
    max_demanda = anos_df['Media'].max() + 2 * anos_df['StdDev'].max()
    passo_cap = 50000 
    estados_possiveis = [cap_init + i * passo_cap for i in range(int((max_demanda - cap_init)/passo_cap) + 5)]
    
    memo = {}

    for ano in range(max_ano, 0, -1):
        memo[ano] = {}
        mu = demanda_dict[ano]['Media']
        sigma = demanda_dict[ano]['StdDev']
        
        for cap_atual in estados_possiveis:
            melhor_vpl = -np.inf
            melhor_decisao = None
            
            vendas = calcular_venda_esperada(cap_atual, mu, sigma)
            fluxo_operacional = vendas * lucro_un
            
            for expansao, investimento in opcoes_dict.items():
                cap_proximo = cap_atual + expansao
                
                if ano == max_ano:
                    vendas_term = calcular_venda_esperada(cap_proximo, mu, sigma)
                    fluxo_perpetuo = vendas_term * lucro_un
                    valor_futuro = fluxo_perpetuo / tma
                else:
                    if cap_proximo in memo[ano+1]:
                        valor_futuro = memo[ano+1][cap_proximo][0]
                    else:
                        valor_futuro = 0 
                
                vpl_decisao = fluxo_operacional - investimento + (valor_futuro / (1 + tma))
                
                if vpl_decisao > melhor_vpl:
                    melhor_vpl = vpl_decisao
                    melhor_decisao = expansao
            
            memo[ano][cap_atual] = (melhor_vpl, melhor_decisao)
            
    return memo

def reconstruir_plano(memo, cap_init, max_ano, opcoes_dict, demanda_df, lucro_un):
    plano = []
    cap_atual = cap_init
    demanda_dict = demanda_df.set_index('Ano').to_dict('index')
    
    for ano in range(1, max_ano + 1):
        if cap_atual not in memo[ano]:
            st.error(f"Erro: Estado de capacidade {cap_atual} não encontrado na tabela DP do ano {ano}.")
            break
            
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
            "Decisão (Expansão)": decisao,
            "Investimento": custo,
            "Receita Esperada": receita,
            "Fluxo Líquido": receita - custo,
            "Nova Capacidade (disp. próx ano)": cap_atual + decisao
        })
        cap_atual += decisao
        
    return pd.DataFrame(plano)

# --- BOTÃO DE AÇÃO E RESULTADOS ---
if st.button("Calcular Plano Ótimo", type="primary"):
    with st.spinner('Otimizando trajetórias...'):
        try:
            # 1. Cálculos
            memo_table = resolver_dp(df_demanda, opcoes_expansao, tma, lucro_unit, cap_inicial)
            vpl_total = memo_table[1][cap_inicial][0]
            df_resultado = reconstruir_plano(memo_table, cap_inicial, df_demanda['Ano'].max(), opcoes_expansao, df_demanda, lucro_unit)
            
            # 2. Exibição (Layout Corrigido)
            st.success("Cálculo realizado com sucesso!")
            
            st.markdown("### Resultados Financeiros")
            
            # Exibe o KPI em destaque, ocupando largura suficiente
            st.metric(label="Valor da Empresa (VPL)", value=f"US$ {vpl_total:,.2f}")
                
            st.divider()

            st.subheader("Plano de Expansão Detalhado")
            st.dataframe(df_resultado.style.format({
                "Capacidade Atual": "{:,.0f}",
                "Demanda Média": "{:,.0f}",
                "Decisão (Expansão)": "{:,.0f}",
                "Investimento": "US$ {:,.0f}",
                "Receita Esperada": "US$ {:,.0f}",
                "Fluxo Líquido": "US$ {:,.0f}",
                "Nova Capacidade (disp. próx ano)": "{:,.0f}"
            }), use_container_width=True)

            # 3. Gráfico
            st.subheader("Visualização Gráfica: Capacidade vs. Demanda")
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df_resultado['Ano'], 
                y=df_resultado['Capacidade Atual'],
                mode='lines+markers',
                name='Capacidade Instalada',
                line=dict(color='green', width=3)
            ))
            
            fig.add_trace(go.Scatter(
                x=df_demanda['Ano'], 
                y=df_demanda['Media'],
                mode='lines',
                name='Demanda Média',
                line=dict(color='blue', dash='dash')
            ))
            
            fig.add_trace(go.Scatter(
                x=pd.concat([df_demanda['Ano'], df_demanda['Ano'][::-1]]),
                y=pd.concat([df_demanda['Media'] + df_demanda['StdDev'], (df_demanda['Media'] - df_demanda['StdDev'])[::-1]]),
                fill='toself',
                fillcolor='rgba(0,0,255,0.1)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                name='Desvio Padrão (Incerteza)'
            ))

            fig.update_layout(
                xaxis_title="Ano",
                yaxis_title="Unidades",
                hovermode="x unified",
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Ocorreu um erro no cálculo. Verifique se os parâmetros de expansão cobrem a demanda necessária. Detalhe: {e}")

else:
    st.info("Ajuste os parâmetros na barra lateral e na tabela acima, depois clique em 'Calcular Plano Ótimo'.")