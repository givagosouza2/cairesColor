import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy.stats import chi2_contingency, fisher_exact
from sklearn.metrics import cohen_kappa_score

st.set_page_config(
    page_title="Categorização de cores",
    page_icon="🎨",
    layout="wide",
)

COLOR_COLUMNS = ["vermelho", "azul", "verde", "amarelo"]
CATEGORY_ORDER = ["Vermelho", "Azul", "Verde", "Amarelo"]
GROUP_ORDER = ["tricromata", "dicromata"]


def normalize_columns(df):
    """Padroniza os nomes das colunas."""
    df = df.copy()
    df.columns = (
        pd.Index(df.columns)
        .astype(str)
        .str.strip()
        .str.lower()
        .str.normalize("NFKD")
        .str.encode("ascii", errors="ignore")
        .str.decode("utf-8")
    )
    return df


def validate_structure(df):
    required = {
        "participante",
        "vermelho",
        "azul",
        "verde",
        "amarelo",
        "tentativa",
        "fenotipo",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            "Estão ausentes as seguintes colunas obrigatórias: "
            + ", ".join(missing)
        )


def prepare_data(df):
    """
    Converte os quatro indicadores binários em uma categoria nominal.
    Como o arquivo não possui uma coluna explícita da peça, a peça é criada
    pela ordem das linhas dentro de cada participante e tentativa.
    """
    df = normalize_columns(df)
    validate_structure(df)

    df = df.copy()
    df["participante"] = df["participante"].astype(str).str.strip()
    df["fenotipo"] = df["fenotipo"].astype(str).str.strip().str.lower()
    df["tentativa"] = pd.to_numeric(df["tentativa"], errors="coerce")

    for col in COLOR_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # A peça é inferida pela ordem original em cada participante/tentativa.
    df["peca"] = df.groupby(
        ["participante", "tentativa"],
        sort=False,
        dropna=False,
    ).cumcount() + 1

    df["soma_indicadores"] = df[COLOR_COLUMNS].sum(axis=1, min_count=1)
    df["linha_valida"] = (
        df[COLOR_COLUMNS].isin([0, 1]).all(axis=1)
        & (df["soma_indicadores"] == 1)
        & df["tentativa"].isin([1, 2])
        & df["fenotipo"].isin(GROUP_ORDER)
        & df["participante"].ne("")
    )

    category_map = {
        "vermelho": "Vermelho",
        "azul": "Azul",
        "verde": "Verde",
        "amarelo": "Amarelo",
    }

    # Inicializa explicitamente como texto para evitar erro de atribuição
    # de categorias como "Vermelho" em uma coluna inferida como float.
    df["categoria"] = pd.Series(pd.NA, index=df.index, dtype="string")

    for col, category in category_map.items():
        mask = df["linha_valida"] & (df[col] == 1)
        df.loc[mask, "categoria"] = category

    return df


def shannon_entropy(series):
    counts = series.value_counts()
    if counts.sum() == 0:
        return np.nan
    probabilities = counts / counts.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


def consensus(series):
    counts = series.value_counts()
    if counts.sum() == 0:
        return np.nan
    return float(counts.max() / counts.sum())


def benjamini_hochberg(p_values):
    p_values = np.asarray(p_values, dtype=float)
    n = len(p_values)

    order = np.argsort(p_values)
    ranked = p_values[order]

    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)

    result = np.empty(n)
    result[order] = adjusted
    return result


def cramers_v(table):
    chi2 = chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    rows, columns = table.shape
    denominator = min(rows - 1, columns - 1)

    if n == 0 or denominator <= 0:
        return np.nan

    return float(np.sqrt((chi2 / n) / denominator))


def compare_groups_by_piece(data):
    results = []

    for piece, subset in data.groupby("peca"):
        table = pd.crosstab(
            subset["fenotipo"],
            subset["categoria"],
        ).reindex(
            index=GROUP_ORDER,
            columns=CATEGORY_ORDER,
            fill_value=0,
        )

        active_columns = table.columns[table.sum(axis=0) > 0]
        reduced = table[active_columns]

        if reduced.shape[1] < 2:
            continue

        if (reduced.sum(axis=1) == 0).any():
            continue

        try:
            chi2, p_value, dof, expected = chi2_contingency(reduced)
            low_expected = bool((expected < 5).any())
            method = "Qui-quadrado"
            statistic = chi2

            if reduced.shape == (2, 2) and low_expected:
                statistic, p_value = fisher_exact(reduced.to_numpy())
                method = "Fisher exato"

            results.append(
                {
                    "Peça": int(piece),
                    "Método": method,
                    "Estatística": statistic,
                    "p": p_value,
                    "V de Cramér": cramers_v(reduced),
                    "Alguma frequência esperada < 5": low_expected,
                }
            )
        except ValueError:
            continue

    result = pd.DataFrame(results)

    if not result.empty:
        result["p ajustado (BH)"] = benjamini_hochberg(result["p"].values)
        result["Significativo após correção"] = result["p ajustado (BH)"] < 0.05

    return result


def to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8-sig")


st.title("🎨 Análise de categorização de cores")
st.caption(
    "Aplicativo ajustado ao formato do arquivo Dados.csv: um único arquivo, "
    "quatro indicadores binários de cor, duas tentativas e grupo informado em `fenotipo`."
)

uploaded_file = st.file_uploader(
    "Selecione o arquivo CSV",
    type=["csv"],
)

if uploaded_file is None:
    st.info(
        "O arquivo deve conter as colunas: participante, vermelho, azul, verde, "
        "amarelo, tentativa e fenotipo."
    )
    st.stop()

try:
    raw_data = pd.read_csv(uploaded_file, sep=None, engine="python")
    data = prepare_data(raw_data)
except Exception as exc:
    st.error(f"Não foi possível processar o arquivo: {exc}")
    st.stop()

valid_data = data[data["linha_valida"]].copy()
invalid_data = data[~data["linha_valida"]].copy()

valid_data["categoria"] = pd.Categorical(
    valid_data["categoria"],
    categories=CATEGORY_ORDER,
    ordered=True,
)

# Verificações estruturais
counts_per_session = (
    data.groupby(["participante", "tentativa", "fenotipo"])
    .size()
    .rename("Número de peças")
    .reset_index()
)

phenotype_consistency = (
    data.groupby("participante")["fenotipo"]
    .nunique()
    .rename("Número de fenótipos")
    .reset_index()
)

tab_overview, tab_distribution, tab_repeatability, tab_groups, tab_quality, tab_downloads = st.tabs(
    [
        "Visão geral",
        "Distribuição por peça",
        "Repetibilidade",
        "Comparação dos grupos",
        "Qualidade dos dados",
        "Downloads",
    ]
)

with tab_overview:
    participant_counts = (
        valid_data.groupby("fenotipo")["participante"]
        .nunique()
        .reindex(GROUP_ORDER, fill_value=0)
    )

    metric_1, metric_2, metric_3, metric_4 = st.columns(4)
    metric_1.metric("Tricromatas", int(participant_counts["tricromata"]))
    metric_2.metric("Dicromatas", int(participant_counts["dicromata"]))
    metric_3.metric("Peças identificadas", int(valid_data["peca"].nunique()))
    metric_4.metric("Respostas válidas", int(len(valid_data)))

    st.subheader("Dados convertidos")
    st.dataframe(
        valid_data[
            [
                "participante",
                "fenotipo",
                "tentativa",
                "peca",
                "categoria",
                "vermelho",
                "azul",
                "verde",
                "amarelo",
            ]
        ],
        use_container_width=True,
        height=420,
    )

    st.caption(
        "A coluna `peca` é criada automaticamente pela ordem das 85 linhas "
        "de cada participante em cada tentativa."
    )

with tab_distribution:
    selected_attempt = st.selectbox(
        "Tentativa",
        ["Ambas", "Tentativa 1", "Tentativa 2"],
    )

    distribution_data = valid_data.copy()

    if selected_attempt == "Tentativa 1":
        distribution_data = distribution_data[
            distribution_data["tentativa"] == 1
        ]
    elif selected_attempt == "Tentativa 2":
        distribution_data = distribution_data[
            distribution_data["tentativa"] == 2
        ]

    distribution = (
        distribution_data.groupby(
            ["fenotipo", "peca", "categoria"],
            observed=False,
        )
        .size()
        .rename("N")
        .reset_index()
    )

    distribution["Proporção"] = distribution.groupby(
        ["fenotipo", "peca"]
    )["N"].transform(
        lambda values: values / values.sum() if values.sum() else 0
    )

    st.subheader("Proporção de categorias em cada peça")
    fig = px.bar(
        distribution,
        x="peca",
        y="Proporção",
        color="categoria",
        facet_row="fenotipo",
        category_orders={
            "categoria": CATEGORY_ORDER,
            "fenotipo": GROUP_ORDER,
        },
        barmode="stack",
        labels={
            "peca": "Peça",
            "categoria": "Categoria",
            "fenotipo": "Fenótipo",
        },
        height=680,
    )
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

    metrics = (
        distribution_data.groupby(
            ["fenotipo", "peca"],
            observed=False,
        )["categoria"]
        .agg(
            Categoria_predominante=lambda x: x.value_counts().idxmax(),
            Consenso=consensus,
            Entropia_bits=shannon_entropy,
            N="size",
        )
        .reset_index()
    )

    st.subheader("Categoria predominante, consenso e entropia")
    st.dataframe(metrics, use_container_width=True, height=420)

    selected_group = st.radio(
        "Grupo exibido no mapa de calor",
        GROUP_ORDER,
        horizontal=True,
        format_func=lambda x: x.capitalize(),
    )

    heat_data = distribution[
        distribution["fenotipo"] == selected_group
    ].pivot_table(
        index="peca",
        columns="categoria",
        values="Proporção",
        fill_value=0,
        observed=False,
    ).reindex(columns=CATEGORY_ORDER, fill_value=0)

    fig_heat = px.imshow(
        heat_data,
        aspect="auto",
        zmin=0,
        zmax=1,
        labels={
            "x": "Categoria",
            "y": "Peça",
            "color": "Proporção",
        },
        height=750,
    )
    st.plotly_chart(fig_heat, use_container_width=True)

with tab_repeatability:
    paired = valid_data.pivot_table(
        index=["fenotipo", "participante", "peca"],
        columns="tentativa",
        values="categoria",
        aggfunc="first",
        observed=False,
    ).reset_index()

    if 1 not in paired.columns or 2 not in paired.columns:
        st.warning("Não foram encontradas as duas tentativas.")
    else:
        paired = paired.dropna(subset=[1, 2]).copy()
        paired["Concordante"] = paired[1] == paired[2]

        repeatability = (
            paired.groupby(["fenotipo", "participante"])
            .agg(
                Número_de_peças=("peca", "size"),
                Concordância_percentual=("Concordante", "mean"),
            )
            .reset_index()
        )

        kappas = []

        for (group, participant), subset in paired.groupby(
            ["fenotipo", "participante"]
        ):
            kappa = cohen_kappa_score(
                subset[1],
                subset[2],
                labels=CATEGORY_ORDER,
            )

            kappas.append(
                {
                    "fenotipo": group,
                    "participante": participant,
                    "Kappa de Cohen": kappa,
                }
            )

        repeatability = repeatability.merge(
            pd.DataFrame(kappas),
            on=["fenotipo", "participante"],
            how="left",
        )
        repeatability["Concordância percentual"] *= 100

        st.subheader("Repetibilidade por participante")
        st.dataframe(repeatability, use_container_width=True)

        summary = (
            repeatability.groupby("fenotipo")
            .agg(
                Participantes=("participante", "nunique"),
                Concordância_média_pct=("Concordância percentual", "mean"),
                Concordância_mediana_pct=("Concordância percentual", "median"),
                Kappa_médio=("Kappa de Cohen", "mean"),
                Kappa_mediano=("Kappa de Cohen", "median"),
            )
            .reset_index()
        )

        st.subheader("Resumo por grupo")
        st.dataframe(summary, use_container_width=True)

        fig_box = px.box(
            repeatability,
            x="fenotipo",
            y="Concordância percentual",
            points="all",
            category_orders={"fenotipo": GROUP_ORDER},
            labels={
                "fenotipo": "Fenótipo",
                "Concordância percentual": "Concordância (%)",
            },
        )
        st.plotly_chart(fig_box, use_container_width=True)

        st.subheader("Matrizes de transição")
        column_1, column_2 = st.columns(2)

        for container, group in zip(
            [column_1, column_2],
            GROUP_ORDER,
        ):
            with container:
                subset = paired[paired["fenotipo"] == group]

                matrix = pd.crosstab(
                    subset[1],
                    subset[2],
                ).reindex(
                    index=CATEGORY_ORDER,
                    columns=CATEGORY_ORDER,
                    fill_value=0,
                )

                st.markdown(f"**{group.capitalize()}s**")
                st.dataframe(matrix, use_container_width=True)

        piece_repeatability = (
            paired.groupby(["fenotipo", "peca"])["Concordante"]
            .agg(["mean", "size"])
            .reset_index()
            .rename(
                columns={
                    "mean": "Concordância percentual",
                    "size": "N",
                }
            )
        )
        piece_repeatability["Concordância percentual"] *= 100

        st.subheader("Concordância por peça")
        st.dataframe(
            piece_repeatability,
            use_container_width=True,
            height=420,
        )

with tab_groups:
    selected_comparison = st.selectbox(
        "Dados usados na comparação",
        ["Ambas as tentativas", "Somente tentativa 1", "Somente tentativa 2"],
    )

    comparison_data = valid_data.copy()

    if selected_comparison == "Somente tentativa 1":
        comparison_data = comparison_data[
            comparison_data["tentativa"] == 1
        ]
    elif selected_comparison == "Somente tentativa 2":
        comparison_data = comparison_data[
            comparison_data["tentativa"] == 2
        ]

    comparison_results = compare_groups_by_piece(comparison_data)

    st.subheader("Testes por peça")

    if comparison_results.empty:
        st.warning("Não foi possível calcular as comparações.")
    else:
        st.dataframe(
            comparison_results.sort_values("p ajustado (BH)"),
            use_container_width=True,
            height=480,
        )

        significant_count = int(
            comparison_results["Significativo após correção"].sum()
        )
        st.metric(
            "Peças com diferença após correção",
            significant_count,
        )

        fig_p = px.scatter(
            comparison_results,
            x="Peça",
            y="p ajustado (BH)",
            size="V de Cramér",
            hover_data=["Método", "Estatística", "p"],
        )
        fig_p.add_hline(y=0.05, line_dash="dash")
        fig_p.update_yaxes(type="log")
        st.plotly_chart(fig_p, use_container_width=True)

    st.info(
        "Como cada participante responde a muitas peças e duas tentativas, "
        "os testes peça a peça são exploratórios. Para a análise inferencial "
        "principal, recomenda-se um modelo multinomial com efeito aleatório "
        "de participante."
    )

with tab_quality:
    st.subheader("Linhas inconsistentes")

    if invalid_data.empty:
        st.success("Todas as linhas possuem exatamente uma categoria marcada.")
    else:
        st.warning(
            f"Foram encontradas {len(invalid_data)} linhas inválidas. "
            "Cada linha deve ter somente um indicador igual a 1."
        )

        st.dataframe(
            invalid_data[
                [
                    "participante",
                    "fenotipo",
                    "tentativa",
                    "peca",
                    "vermelho",
                    "azul",
                    "verde",
                    "amarelo",
                    "soma_indicadores",
                ]
            ],
            use_container_width=True,
        )

    st.subheader("Número de peças por participante e tentativa")
    abnormal_counts = counts_per_session[
        counts_per_session["Número de peças"] != 85
    ]

    if abnormal_counts.empty:
        st.success(
            "Todos os participantes possuem 85 linhas em cada tentativa."
        )
    else:
        st.warning(
            "Há participantes ou tentativas com número de linhas diferente de 85."
        )
        st.dataframe(abnormal_counts, use_container_width=True)

    st.subheader("Consistência do fenótipo")
    inconsistent_phenotypes = phenotype_consistency[
        phenotype_consistency["Número de fenótipos"] != 1
    ]

    if inconsistent_phenotypes.empty:
        st.success(
            "Cada participante está associado a apenas um fenótipo."
        )
    else:
        st.warning(
            "Alguns participantes aparecem associados a mais de um fenótipo."
        )
        st.dataframe(inconsistent_phenotypes, use_container_width=True)

with tab_downloads:
    st.subheader("Exportar resultados")

    st.download_button(
        "Baixar dados válidos organizados",
        data=to_csv_bytes(
            valid_data[
                [
                    "participante",
                    "fenotipo",
                    "tentativa",
                    "peca",
                    "categoria",
                ]
            ]
        ),
        file_name="dados_categorizacao_validos.csv",
        mime="text/csv",
    )

    if not invalid_data.empty:
        st.download_button(
            "Baixar linhas inconsistentes",
            data=to_csv_bytes(invalid_data),
            file_name="linhas_inconsistentes.csv",
            mime="text/csv",
        )

    distribution_export = (
        valid_data.groupby(
            ["fenotipo", "tentativa", "peca", "categoria"],
            observed=False,
        )
        .size()
        .rename("N")
        .reset_index()
    )

    distribution_export["Proporção"] = distribution_export.groupby(
        ["fenotipo", "tentativa", "peca"]
    )["N"].transform(
        lambda values: values / values.sum() if values.sum() else 0
    )

    st.download_button(
        "Baixar distribuição das categorias",
        data=to_csv_bytes(distribution_export),
        file_name="distribuicao_categorias.csv",
        mime="text/csv",
    )

    if "repeatability" in locals():
        st.download_button(
            "Baixar repetibilidade individual",
            data=to_csv_bytes(repeatability),
            file_name="repetibilidade_individual.csv",
            mime="text/csv",
        )

    if (
        "comparison_results" in locals()
        and not comparison_results.empty
    ):
        st.download_button(
            "Baixar comparação entre grupos",
            data=to_csv_bytes(comparison_results),
            file_name="comparacao_tricromatas_dicromatas.csv",
            mime="text/csv",
        )
