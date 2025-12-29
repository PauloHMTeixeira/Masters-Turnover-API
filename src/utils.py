import logging
import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # evita erro tkinter em ambiente server/streamlit
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from beartype import beartype
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from datetime import datetime
from PyPDF2 import PdfMerger
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.metrics import (
    f1_score,
    classification_report,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    confusion_matrix
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def _get_feature_names(preprocessor: ColumnTransformer, X: pd.DataFrame) -> list[str]:
    # Após fit, pega nomes finais (numéricas + onehot)
    try:
        return preprocessor.get_feature_names_out().tolist()
    except Exception:
        # fallback simples (não deveria ocorrer nas versões recentes)
        num_cols = X.select_dtypes(include=["number"]).columns.tolist()
        cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        return num_cols + cat_cols


def _extract_importance(best_pipeline: Pipeline, feature_names: list[str]) -> pd.DataFrame:
    """
    Retorna DF com colunas: feature, importance (sempre em ordem decrescente).
    Para:
      - RF/GB: feature_importances_
      - LogReg: abs(coef_)
    """
    model = best_pipeline.named_steps["model"]

    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "coef_"):
        # binário -> coef_ shape (1, n_features)
        imp = np.abs(model.coef_).ravel()
    else:
        # fallback: tudo zero
        imp = np.zeros(len(feature_names))

    feat_imp_df = pd.DataFrame({"feature": feature_names, "importance": imp})
    feat_imp_df = feat_imp_df.sort_values("importance", ascending=False)
    return feat_imp_df


def generate_ml_report_pdf(
    output_path: str,
    leaderboard_df: pd.DataFrame,
    best_model_name: str,
    holdout_metrics: dict,
    confusion_matrix,
    feat_imp_df: pd.DataFrame,
    top_n: int = 25,
) -> str:
    """
    Gera um PDF com:
      1) Leaderboard (CV)
      2) Métricas hold-out + matriz de confusão
      3) Importância de variáveis do modelo final
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with PdfPages(output_path) as pdf:

        # ---------- Página 1: Leaderboard ----------
        fig, ax = plt.subplots(figsize=(11.7, 8.3))  # A4 landscape-ish
        ax.axis("off")
        ax.set_title("Leaderboard de Modelos (Cross-Validation)", fontsize=16, pad=20)

        df_show = leaderboard_df.copy()
        # Destaque do melhor modelo
        df_show["Melhor?"] = df_show["model"].apply(lambda x: "✅" if x == best_model_name else "")

        table = ax.table(
            cellText=df_show.values,
            colLabels=df_show.columns,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 1.6)

        ax.text(
            0.5, 0.08,
            "Critério de seleção: F1 médio no CV (maior é melhor).",
            ha="center", va="center", fontsize=10
        )

        pdf.savefig(fig)
        plt.close(fig)

        # ---------- Página 2: Hold-out + Matriz de Confusão ----------
        fig, ax = plt.subplots(figsize=(11.7, 8.3))
        ax.axis("off")
        ax.set_title("Avaliação no Hold-out (Teste)", fontsize=16, pad=20)

        # Texto de métricas
        metrics_text = (
            f"Modelo final (tunado): {best_model_name}\n\n"
            f"F1: {holdout_metrics.get('f1'):.4f}\n"
            f"Precision: {holdout_metrics.get('precision'):.4f}\n"
            f"Recall: {holdout_metrics.get('recall'):.4f}\n"
        )
        ax.text(0.05, 0.85, metrics_text, fontsize=13, va="top")

        # Matriz de confusão como subplot inserido
        ax_cm = fig.add_axes([0.55, 0.25, 0.35, 0.5])  # [left, bottom, width, height]
        disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
        disp.plot(ax=ax_cm, colorbar=False)
        ax_cm.set_title("Matriz de Confusão")

        pdf.savefig(fig)
        plt.close(fig)

        # ---------- Página 3: Feature Importance ----------
        df_plot = feat_imp_df.head(top_n).copy()
        fig, ax = plt.subplots(figsize=(11.7, 8.3))
        ax.barh(df_plot["feature"][::-1], df_plot["importance"][::-1])
        ax.set_title(f"Top {top_n} Importâncias (Modelo final)", fontsize=16, pad=20)
        ax.set_xlabel("Importância")
        ax.set_ylabel("Variável")
        plt.tight_layout()

        pdf.savefig(fig)
        plt.close(fig)

    return output_path

def generate_pdf_feature_importance_from_df(feat_imp_df: pd.DataFrame, pdf_path: str, top_n: int = 25) -> str:
    df_plot = feat_imp_df.head(top_n).copy()
    plt.figure(figsize=(10, 7))
    plt.barh(df_plot["feature"][::-1], df_plot["importance"][::-1])
    plt.title(f"Top {top_n} Importâncias (modelo selecionado)")
    plt.xlabel("Importância")
    plt.ylabel("Variável")
    plt.tight_layout()

    out = pdf_path.replace(".pdf", "") + "_importance.pdf"
    with PdfPages(out) as pdf:
        pdf.savefig()
        plt.close()

    return out

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
def ml_pipeline(df: pd.DataFrame, file_name: str, gender: bool = False) -> str:
    """
    Executes the machine learning pipeline for turnover prediction, including
    model comparison, hyperparameter optimization, evaluation, and reporting.

    The pipeline performs the following steps:
    1. Validates and prepares the target variable (`Attrition`) as a binary outcome.
    2. Splits the dataset into stratified train and test sets.
    3. Applies a unified preprocessing pipeline for numerical and categorical features.
    4. Trains and compares multiple baseline models using cross-validation (F1 score).
    5. Selects the best-performing baseline model.
    6. Performs hyperparameter optimization only on the selected model.
    7. Evaluates the tuned model on a hold-out test set.
    8. Computes performance metrics and feature importance.
    9. Generates a comprehensive PDF report containing:
        - Model leaderboard (cross-validation results),
        - Hold-out evaluation metrics and confusion matrix,
        - Feature importance of the final model.

    Args:
        df (pd.DataFrame):
            Input dataset containing predictor variables and a binary
            `Attrition` column encoded as 0/1.
        file_name (str):
            Base name used to generate output artifacts (PDF reports).
        gender (bool, optional):
            Indicates whether the analysis is gender-specific. When True,
            a suffix is appended to the output file name. Defaults to False.

    Returns:
        str:
            File path to the generated machine learning PDF report, which
            includes the leaderboard, evaluation metrics, and feature importance.

    Raises:
        ValueError:
            If the `Attrition` column contains invalid or non-numeric values
            after preprocessing.
    """

    # -----------------------------
    # Garantia final do target (0/1)
    # -----------------------------
    df["Attrition"] = pd.to_numeric(df["Attrition"], errors="coerce")

    if df["Attrition"].isna().any():
        raise ValueError(
            "A coluna 'Attrition' contém valores inválidos após o mapeamento do target."
        )

    X = df.drop(columns=["Attrition"])
    y = df["Attrition"].astype(int)

    # Split único
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, random_state=42, test_size=0.25
    )

    preprocessor = _build_preprocessor(X_train)

    # -----------------------------
    # Modelos baseline (padronizados)
    # -----------------------------
    candidates = {
        "logreg": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "rf": RandomForestClassifier(random_state=42, class_weight="balanced"),
        "gb": GradientBoostingClassifier(random_state=42),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 1) Comparação baseline (CV)
    scores = []
    for name, model in candidates.items():
        pipe = Pipeline(steps=[("prep", preprocessor), ("model", model)])
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1")
        mean_f1 = float(np.mean(cv_scores))
        std_f1 = float(np.std(cv_scores))
        scores.append((mean_f1, std_f1, name))
        logging.info("Baseline %s | f1=%.4f ± %.4f", name, mean_f1, std_f1)

    scores.sort(reverse=True, key=lambda x: x[0])
    best_name = scores[0][2]
    logging.info("Melhor baseline: %s", best_name)

    # Leaderboard DF (PASSO 3)
    leaderboard_df = pd.DataFrame(
        [{"model": name, "f1_cv_mean": mean, "f1_cv_std": std} for (mean, std, name) in scores]
    ).sort_values("f1_cv_mean", ascending=False)

    # -----------------------------
    # 2) Hiperparametrização do vencedor
    # -----------------------------
    base_pipe = Pipeline(steps=[("prep", preprocessor), ("model", candidates[best_name])])

    if best_name == "logreg":
        param_dist = {
            "model__C": np.logspace(-3, 2, 25),
            "model__penalty": ["l2"],
            "model__solver": ["lbfgs", "liblinear"],
        }
    elif best_name == "rf":
        param_dist = {
            "model__n_estimators": [200, 400, 600],
            "model__max_depth": [None, 5, 10, 20],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2", None],
        }
    else:  # gb
        param_dist = {
            "model__n_estimators": [100, 200, 300],
            "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "model__max_depth": [2, 3, 4],
            "model__subsample": [0.7, 0.85, 1.0],
        }

    search = RandomizedSearchCV(
        estimator=base_pipe,
        param_distributions=param_dist,
        n_iter=25,
        scoring="f1",
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)

    best_pipe = search.best_estimator_
    logging.info("Best params (%s): %s", best_name, search.best_params_)

    # -----------------------------
    # 3) Avaliação final em hold-out (PASSO 3)
    # -----------------------------
    preds = best_pipe.predict(X_test)

    f1 = float(f1_score(y_test, preds))
    prec = float(precision_score(y_test, preds, zero_division=0))
    rec = float(recall_score(y_test, preds, zero_division=0))
    cm = confusion_matrix(y_test, preds)

    holdout_metrics = {"f1": f1, "precision": prec, "recall": rec}

    logging.info("F1 hold-out: %.4f", f1)
    logging.info("\n%s", classification_report(y_test, preds))

    # -----------------------------
    # 4) Feature importance / coef
    # -----------------------------
    fitted_prep = best_pipe.named_steps["prep"]
    feat_names = _get_feature_names(fitted_prep, X_train)
    feat_imp_df = _extract_importance(best_pipe, feat_names)

    # -----------------------------
    # 5) Gera PDF de ML completo (Leaderboard + Hold-out + Importance)
    # -----------------------------
    os.makedirs("data", exist_ok=True)

    suffix = "_gender" if gender else ""
    ml_report_path = os.path.join("data", f"{file_name}{suffix}_ML_REPORT.pdf")

    generate_ml_report_pdf(
        output_path=ml_report_path,
        leaderboard_df=leaderboard_df,
        best_model_name=best_name,
        holdout_metrics=holdout_metrics,
        confusion_matrix=cm,
        feat_imp_df=feat_imp_df,
        top_n=25,
    )

    return ml_report_path

@beartype
def generate_cover_pdf(output_path: str, file_name: str, turnover_col: str):
    """
    Gera um PDF contendo apenas a capa do relatório final.
    """
    c = canvas.Canvas(output_path, pagesize=A4)

    width, height = A4
    margin = 50

    # Título
    c.setFont("Helvetica-Bold", 24)
    c.drawCentredString(width / 2, height - 120, "Relatório de Turnover")

    # Subtítulo
    c.setFont("Helvetica", 16)
    c.drawCentredString(width / 2, height - 160, "Análise Exploratória + Importância de Variáveis")

    # Linha divisória
    c.line(margin, height - 180, width - margin, height - 180)

    # Informações do arquivo
    c.setFont("Helvetica", 12)
    c.drawString(margin, height - 220, f"Arquivo analisado: {file_name}.csv")
    c.drawString(margin, height - 240, f"Coluna de Turnover: {turnover_col}")

    # Data
    data_atual = datetime.now().strftime("%d/%m/%Y %H:%M")
    c.drawString(margin, height - 280, f"Data de geração: {data_atual}")

    # Rodapé com GitHub
    c.setFont("Helvetica-Oblique", 10)
    c.drawCentredString(width / 2, 50, "Gerado automaticamente pelo Analisador de Turnover")
    c.drawCentredString(width / 2, 35, "https://github.com/PauloHMTeixeira")

    c.showPage()
    c.save()

    return output_path

@beartype
def merge_pdfs(output_path: str, pdf_paths: list):
    """
    Junta vários PDFs em um único arquivo final.
    """
    merger = PdfMerger()

    for pdf in pdf_paths:
        merger.append(pdf)

    merger.write(output_path)
    merger.close()

    return output_path

def infer_binary_target_mapping(series: pd.Series) -> pd.Series:
    """
    Recebe uma Series (target) e devolve uma Series 0/1 robusta.
    Regras:
      - aceita Yes/No, True/False, 0/1
      - para outros 2 rótulos: define o minoritário como 1 (turnover) e o majoritário como 0
      - se não for binário: levanta ValueError
    """
    non_null = series.dropna()

    # Se vazio após dropna
    if non_null.empty:
        raise ValueError("A coluna de turnover está vazia (só nulos).")

    # Normaliza para análise de valores
    if non_null.dtype == object:
        vals_norm = non_null.astype(str).str.strip()
    else:
        vals_norm = non_null

    unique_vals = pd.Series(vals_norm).dropna().unique()

    if len(unique_vals) != 2:
        raise ValueError(
            f"A coluna de turnover precisa ser binária (2 valores). "
            f"Encontrado: {len(unique_vals)} valores únicos (ex.: {unique_vals[:5]})."
        )

    lower_set = set([str(v).strip().lower() for v in unique_vals])

    # Mapeamentos comuns
    common_maps = [
        ({"yes", "no"}, {"yes": 1, "no": 0}),
        ({"true", "false"}, {"true": 1, "false": 0}),
        ({"1", "0"}, {"1": 1, "0": 0}),
    ]

    for keyset, mapping in common_maps:
        if lower_set == keyset:
            return series.astype(str).str.strip().str.lower().map(mapping).astype("Int64")

    # Caso genérico: minoritário = 1
    counts = non_null.value_counts()
    positive_label = counts.idxmin()
    return series.map(lambda x: 1 if x == positive_label else 0).astype("Int64")


def prepare_attrition_column(df: pd.DataFrame, turnover_col: str) -> pd.DataFrame:
    """
    Garante que a coluna 'Attrition' exista no df, como 0/1,
    a partir da coluna turnover_col escolhida pelo usuário.
    """
    if turnover_col not in df.columns:
        raise ValueError(f"A coluna '{turnover_col}' não existe no dataset enviado.")

    df = df.copy()
    df["Attrition"] = infer_binary_target_mapping(df[turnover_col])
    return df

