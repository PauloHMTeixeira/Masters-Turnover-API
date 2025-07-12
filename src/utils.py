import pandas as pd
import logging
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report
from sklearn.feature_selection import VarianceThreshold
from beartype import beartype


@beartype
def data_pipeline(df: pd.DataFrame, file_name: str) -> str:
    """
    Pipeline de dados para gerar análises estatísticas.

    Args:
        df (pd.DataFrame): Dataframe recebido como input, de onde vão ser criadas as análises.
    
    Returns:
        path (str): Path para o PDF gerado com as análises.
    """
    # Garante coluna padrão/target está no dataframe
    if 'Attrition' not in df.columns:
        raise ValueError("A coluna 'Attrition' precisa estar presente no DataFrame.")

    # Separa as colunas baseada no tipo
    numericas = df.select_dtypes(include='number').columns.tolist()
    categóricas = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    # Remove 'Attrition' das análises
    if 'Attrition' in numericas:
        numericas.remove('Attrition')
    if 'Attrition' in categóricas:
        categóricas.remove('Attrition')

    # Criação do PDF para salvar
    with PdfPages(f'{file_name}.pdf') as pdf:

        # Gráficos para variáveis numéricas
        for col in numericas:
            plt.figure(figsize=(8, 5))
            sns.kdeplot(data=df, x=col, hue='Attrition', fill=True, common_norm=False)
            plt.title(f'Distribuição de {col} por Attrition')
            plt.tight_layout()
            pdf.savefig()
            plt.close()

        # Gráficos para variáveis categóricas
        for col in categóricas:
            # Ignora colunas com cardinalidade muito alta
            if df[col].nunique() > 15:
                continue
            plt.figure(figsize=(8, 5))
            sns.countplot(data=df, x=col, hue='Attrition', order=df[col].value_counts().index)
            plt.title(f'Contagem de {col} por Attrition')
            plt.xticks(rotation=45)
            plt.tight_layout()
            pdf.savefig()
            plt.close()

    print(f"PDF gerado: {file_name}.pdf")

    return f"{file_name}.pdf"

@beartype
def generate_pdf_feature_importance(
        model: RandomForestClassifier,
        feature_names: list[str],
        pdf_path: str,
    ) -> str:
    """
    Gera um gráfico das top N importâncias de variáveis de um modelo Random Forest
    e adiciona como uma nova página a um arquivo PDF existente.

    Args:
        model (RandomForestClassifier): Modelo treinado do qual extrair as importâncias.
        feature_names (list[str]): Lista com os nomes das features.
        pdf_path (str): Caminho do arquivo PDF onde o gráfico será salvo (como nova página).

    Returns:
        str: Caminho do arquivo PDF atualizado.
    """
    importances = model.feature_importances_

    feat_imp_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values(by='importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=feat_imp_df, y='feature', x='importance', palette='viridis')
    plt.title('Importâncias de Variáveis Selecionadas - Random Forest')
    plt.xlabel('Importância')
    plt.ylabel('Variável')
    plt.tight_layout()

    with PdfPages(pdf_path + '_importance.pdf') as pdf:
        pdf.savefig()
        plt.close()

    return pdf_path + '_importance.pdf'

@beartype
def ml_pipeline(df: pd.DataFrame, file_name: str) -> str:
    """
    Pipeline de modelagem para gerar análise de importância das variáveis.

    Args:
        df (pd.DataFrame): Dataframe recebido como input, de onde vai ser criadas a análise.
    
    Returns:
        path (str): Path para o PDF gerado com a análise.
    """
    df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

    # Variáveis categóricas
    categorical = df.select_dtypes(include='object').columns
    df = pd.get_dummies(df, columns=categorical, drop_first=True)

    # Drop de colunas constantes ou quasi-constantes
    X_temp = df.drop(columns=['Attrition'])
    selector = VarianceThreshold(threshold=0.01)
    selector.fit(X_temp)
    selected_cols = X_temp.columns[selector.get_support()]
    df = df[selected_cols.tolist() + ['Attrition']]

    # Separaçãodo target
    X = df.drop(columns=['Attrition'])
    y = df['Attrition']

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)


    # Seleção de variáveis - Max out
    selected_features = []
    remaining_features = list(X_train.columns)
    best_f1 = 0
    improvement = True

    while improvement and remaining_features:
        improvement = False
        scores = []

        for feature in remaining_features:
            features_to_test = selected_features + [feature]
            model = LogisticRegression(max_iter=1000, class_weight='balanced')
            model.fit(X_train[features_to_test], y_train)
            preds = model.predict(X_test[features_to_test])
            score = f1_score(y_test, preds)
            scores.append((score, feature))

        scores.sort(reverse=True)
        top_score, top_feature = scores[0]

        if top_score > best_f1:
            best_f1 = top_score
            selected_features.append(top_feature)
            remaining_features.remove(top_feature)
            improvement = True
        else:
            break

    logging.info("Variáveis selecionadas: %s", selected_features)

    # Modelagem
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf.fit(X_train[selected_features], y_train)

    # Predição
    y_pred = rf.predict(X_test[selected_features])

    logging.info(classification_report(y_test, y_pred))

    # Geração do gráfico de feature importance no PDF
    pdf_path = generate_pdf_feature_importance(rf, selected_features, file_name)

    return pdf_path
