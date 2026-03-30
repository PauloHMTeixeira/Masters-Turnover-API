import streamlit as st
import pandas as pd
import requests

# -------------------------------------------------------
# CONFIGURAÇÕES DO APP
# -------------------------------------------------------
st.set_page_config(
    page_title="Analisador de Turnover",
    layout="wide",
)

# Estado persistente
if "result_paths" not in st.session_state:
    st.session_state["result_paths"] = None



# -------------------------------------------------------
# MENU LATERAL
# -------------------------------------------------------
menu = st.sidebar.selectbox(
    "Menu",
    ["Início", "Processar Arquivo", "Tutorial", "Sobre / Contato"]
)


# -------------------------------------------------------
# 1. PÁGINA INICIAL (LANDING PAGE)
# -------------------------------------------------------
if menu == "Início":
    st.title("Analisador de Turnover")
    st.markdown("""
    Bem-vindo ao **Analisador de Turnover**, uma ferramenta desenvolvida para automatizar:

    - A análise exploratória do seu dataset de RH  
    - A modelagem de machine learning para prever turnover  
    - A geração automática de PDFs profissionais com gráficos e insights  
    
    ---
    ### Como funciona?
    1. Você sobe um arquivo CSV usando o menu **Processar Arquivo**  
    2. Escolhe qual coluna representa o turnover (ex.: *Attrition*)  
    3. A pipeline roda automaticamente  
    4. PDFs são gerados e disponibilizados para download  
    ---
    ### Vantagens
    - Interface simples e intuitiva  
    - Pipeline completa pronta para reuso  
    - Resultados salvos automaticamente  
    - Ideal para apresentações, relatórios e auditorias internas  

    Utilize o menu lateral para começar.
    """)



# -------------------------------------------------------
# 2. PROCESSAMENTO DO ARQUIVO
# -------------------------------------------------------
elif menu == "Processar Arquivo":

    st.title("Processar Arquivo")

    # Upload
    uploaded_file = st.file_uploader(
        "Arraste seu arquivo CSV ou clique para selecionar",
        type=["csv"],
    )

    df_preview = None
    columns = []

    if uploaded_file:
        try:
            df_preview = pd.read_csv(uploaded_file)
            st.success("Arquivo carregado com sucesso!")
            st.subheader("Prévia do Dataset")
            st.dataframe(df_preview.head(10))
            columns = df_preview.columns.tolist()
        except Exception as e:
            st.error(f"Erro ao ler o arquivo: {e}")

    # Dropdown
    turnover_col = None
    if columns:
        st.subheader("Selecione a coluna de Turnover")
        turnover_col = st.selectbox(
            "Coluna target", 
            options=columns,
            help="Selecione a coluna que representa turnover/demissão/saída."
        )

    # Processar
    if uploaded_file and turnover_col:
        if st.button("Processar Pipeline"):
            with st.spinner("Processando dados e gerando PDFs..."):

                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type,
                    )
                }
                data = {"turnover_col": turnover_col}

                try:
                    response = requests.post(
                        "http://localhost:5000/executar-interface",
                        files=files,
                        data=data,
                        timeout=600000
                    )

                    if response.status_code == 200:
                        result = response.json().get("resultado")
                        st.session_state["result_paths"] = result
                        st.success("Pipeline executada com sucesso!")
                        if result.get("HR Summary"):
                            st.subheader("Resumo para RH (gerado automaticamente)")
                            st.write(result["HR Summary"])

                    else:
                        st.error(f"Erro da API: {response.text}")

                except Exception as e:
                    st.error(f"Erro ao conectar com a API: {e}")


    # Mostrar resultados persistidos
    result_paths = st.session_state.get("result_paths")
    if result_paths:
        st.subheader("Relatório Final (Unificado)")
        st.code(result_paths.get("Final Report"))

        try:
            with open(result_paths.get("Final Report"), "rb") as f:
                pdf_final = f.read()

            st.download_button(
                label="Baixar Relatório Completo",
                data=pdf_final,
                file_name=result_paths.get("Final Report").split("/")[-1],
                mime="application/pdf",
            )
        except Exception as e:
            st.warning(f"Não foi possível carregar o PDF final para download. {e}")



# -------------------------------------------------------
# 3. TUTORIAL
# -------------------------------------------------------
elif menu == "Tutorial":

    st.title("Tutorial de Uso")

    st.markdown("""
    ### **1. Preparando o arquivo**
    - O arquivo deve estar no formato **CSV**  
    - Deve conter **ao menos uma coluna de turnover** (ex.: Attrition, Saída, Demissão)  
    - Deve conter tanto colunas numéricas quanto categóricas  

    ---

    ### **2. Enviando o arquivo**
    Vá até o menu **Processar Arquivo** e:
    1. Arraste o arquivo para a área de upload  
    2. Confira a prévia do dataset  
    3. Selecione a coluna de turnover no dropdown  

    ---

    ### **3. Execução da Pipeline**
    A pipeline irá:
    - Criar análises exploratórias  
    - Gerar gráficos por categoria e número  
    - Treinar modelo de ML  
    - Executar seleção de variáveis  
    - Criar PDF com **Análise Descritiva**, **Importância de Variáveis** e **Insights via IA**    

    ---

    ### **4. Download dos Resultados**
    Após o processamento:
    - Os caminhos dos arquivos aparecem na tela  
    - Você pode fazer download direto pela interface 
    """)



# -------------------------------------------------------
# 4. SOBRE / CONTATO
# -------------------------------------------------------
elif menu == "Sobre / Contato":

    st.title("Sobre o Projeto")

    st.markdown("""
    Este projeto faz parte da iniciativa para automatizar análises de **turnover voluntário** 
    utilizando ciência de dados e machine learning.

    **Desenvolvido por:**  
    [Paulo H. M. Teixeira](https://github.com/PauloHMTeixeira)

    Repositório oficial:  
    [github.com/PauloHMTeixeira](https://github.com/PauloHMTeixeira/Masters-Turnover-API)

    **Tecnologias utilizadas:**
    - Python  
    - Pandas  
    - Seaborn / Matplotlib  
    - Scikit-Learn  
    - Flask (API)  
    - Streamlit (Interface)  
    - PDFReports  
    """)

