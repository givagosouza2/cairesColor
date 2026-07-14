import io
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from scipy.stats import chi2_contingency, fisher_exact
from sklearn.metrics import cohen_kappa_score, confusion_matrix

st.set_page_config(
    page_title="Categorização de cores",
    page_icon="🎨",
    layout="wide",
)

CATEGORIES = ["Verde", "Vermelho", "Azul", "Amarelo"]


def normalize_category(value, numeric_map):
    """Converte diferentes formas de entrada para as quatro categorias."""
    if pd.isna(value):
        return np.nan

    text = str(value).strip()
    if not text:
        return np.nan

    # Mapeamento numérico definido pelo usuário.
    if text in numeric_map:
        return numeric_map[text]

    normalized = (
        text.lower()
        .replace("á", "a")
        .replace("ã", "a")
        .replace("â", "a")
        .replace("é", "e")
        .replace("ê", "e")
        .replace("í", "i")
        .replace("ó", "o")
        .replace("ô", "o")
        .replace("õ", "o")
        .replace("ú", "u")
        .replace("ç", "c")
    )

    aliases = {
        "verde": "Verde",
        "v": "Verde",
        "green": "Verde",
        "vermelho": "Vermelho",
        "verm": "Vermelho",
        "red": "Vermelho",
        "azul": "Azul",
        "a": "Azul",
        "blue": "Azul",
        "amarelo": "Amarelo",
        "amar": "Amarelo",
        "yellow": "Amarelo",
    }
    return aliases.get(normalized, np.nan)


def parse_header(column_name):
    """
    Tenta identificar participante e teste no cabeçalho.

    Exemplos reconhecidos:
    P01_T1, P01-T2, P01 Teste 1, P01.2, participante01_av1
    """
    text = str(column_name).strip()

    patterns = [
        r"^(.*?)[_\-\s\.]+(?:t|teste|av|avaliacao|sessao|s)?\s*([12])$",
        r"^(.*?)(?:t|teste|av|avaliacao|sessao|s)\s*([12])$",
    ]

    for pattern in patterns:
        match = re.match(pattern, text, flags=re.IGNORECASE)
        if match:
            participant = match.group(1).strip(" _-.")
            test = int(match.group(2))
            if participant:
                return participant, test

    return None, None


def dataframe_to_long(df, group, parsing_mode, numeric_map):
    """Transforma a planilha larga em uma linha por resposta."""
    df = df.copy()
    piece_col = df.columns[0]
    response_cols = list(df.columns[1:])

    if len(response_cols) < 2:
        raise ValueError("O arquivo deve ter pelo menos duas colunas de respostas além da coluna da peça.")

    records = []

    if parsing_mode == "Colunas consecutivas em pares":
        if len(response_cols) % 2 != 0:
            st.warning(
                f"O arquivo do grupo {group} possui número ímpar de colunas de resposta. "
                "A última coluna será ignorada."
            )
        usable_cols = response_cols[: len(response_cols) - (len(response_cols) % 2)]

        column_map = {}
        for i in range(0, len(usable_cols), 2):
            participant = f"{group[:3]}_{i // 2 + 1:03d}"
            column_map[usable_cols[i]] = (participant, 1)
            column_map[usable_cols[i + 1]] = (participant, 2)

    else:
        column_map = {}
        unidentified = []
        for col in response_cols:
            participant, test = parse_header(col)
            if participant is None:
                unidentified.append(col)
            else:
                column_map[col] = (participant, test)

        if unidentified:
            raise ValueError(
                "Não foi possível identificar participante e teste nestas colunas: "
                + ", ".join(map(str, unidentified[:10]))
                + (", ..." if len(unidentified) > 10 else "")
                + ". Use o modo de colunas consecutivas em pares ou renomeie os cabeçalhos."
            )

    for col, (participant, test) in column_map.items():
        temp = pd.DataFrame(
            {
                "Grupo": group,
                "Participante": participant,
                "Teste": test,
                "Peca": df[piece_col],
                "Resposta_original": df[col],
            }
        )
        temp["Categoria"] = temp["Resposta_original"].apply(
            lambda x: normalize_category(x, numeric_map)
        )
        temp["Coluna_origem"] = str(col)
        records.append(temp)

    long_df = pd.concat(records, ignore_index=True)
    long_df["Peca"] = long_df["Peca"].astype(str).str.strip()
    return long_df


def shannon_entropy(series):
    counts = series.value_counts()
    if counts.sum() == 0:
        return np.nan
    probs = counts / counts.sum()
    return float(-(probs * np.log2(probs)).sum())


def consensus(series):
    counts = series.value_counts()
    if counts.sum() == 0:
        return np.nan
    return float(counts.max() / counts.sum())


def cramers_v(table):
    chi2 = chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    r, k = table.shape
    denominator = min(k - 1, r - 1)
    if n == 0 or denominator <= 0:
        return np.nan
    return float(np.sqrt((chi2 / n) / denominator))


def benjamini_hochberg(p_values):
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    result = np.empty(n)
    result[order] = adjusted
    return result


def compare_groups_per_piece(data):
    results = []

    for piece, subset in data.groupby("Peca"):
        table = pd.crosstab(subset["Grupo"], subset["Categoria"]).reindex(
            index=["Tricromatas", "Dicromatas"],
            columns=CATEGORIES,
            fill_value=0,
        )

        active_cols = table.columns[table.sum(axis=0) > 0]
        reduced = table[active_cols]

        if reduced.shape[1] < 2 or (reduced.sum(axis=1) == 0).any():
            continue

        try:
            chi2, p_value, dof, expected = chi2_contingency(reduced)
            low_expected = bool((expected < 5).any())
            method = "Qui-quadrado"
            statistic = chi2

            # Fisher exato apenas para tabelas 2x2.
            if reduced.shape == (2, 2) and low_expected:
                odds_ratio, p_value = fisher_exact(reduced.to_numpy())
                statistic = odds_ratio
                method = "Fisher exato"

            results.append(
                {
                    "Peca": piece,
                    "Metodo": method,
                    "Estatistica": statistic,
                    "p": p_value,
                    "V_de_Cramer": cramers_v(reduced),
                    "Frequencia_esperada_menor_5": low_expected,
                }
            )
        except ValueError:
            continue

    result_df = pd.DataFrame(results)
    if not result_df.empty:
        result_df["p_ajustado_BH"] = benjamini_hochberg(result_df["p"].values)
        result_df["Significativo_5pct"] = result_df["p_ajustado_BH"] < 0.05
    return result_df


def csv_download(df):
    return df.to_csv(index=False).encode("utf-8-sig")


st.title("🎨 Análise de categorização de cores")
st.caption(
    "Comparação entre tricromatas e dicromatas, com duas avaliações por participante."
)

with st.sidebar:
    st.header("Configuração dos dados")

    parsing_mode = st.radio(
        "Como as colunas de respostas estão organizadas?",
        [
            "Colunas consecutivas em pares",
            "Cabeçalhos identificam participante e teste",
        ],
        help=(
            "No primeiro modo, as colunas são interpretadas como Participante 1 - Teste 1, "
            "Participante 1 - Teste 2, Participante 2 - Teste 1, etc."
        ),
    )

    st.subheader("Mapeamento de códigos numéricos")
    st.caption("Use esta opção caso as respostas sejam registradas como 1, 2, 3 e 4.")
    code_green = st.text_input("Código de Verde", "1")
    code_red = st.text_input("Código de Vermelho", "2")
    code_blue = st.text_input("Código de Azul", "3")
    code_yellow = st.text_input("Código de Amarelo", "4")

    numeric_map = {
        str(code_green).strip(): "Verde",
        str(code_red).strip(): "Vermelho",
        str(code_blue).strip(): "Azul",
        str(code_yellow).strip(): "Amarelo",
    }

    delimiter = st.selectbox(
        "Separador do CSV",
        ["Detecção automática", ";", ",", "\t"],
    )


def read_csv(uploaded_file):
    sep = None if delimiter == "Detecção automática" else delimiter
    return pd.read_csv(uploaded_file, sep=sep, engine="python")


col1, col2 = st.columns(2)

with col1:
    tri_file = st.file_uploader(
        "Arquivo dos tricromatas",
        type=["csv"],
        key="tri",
    )

with col2:
    di_file = st.file_uploader(
        "Arquivo dos dicromatas",
        type=["csv"],
        key="di",
    )

if tri_file is None or di_file is None:
    st.info(
        "Inclua os dois arquivos CSV. A primeira coluna deve conter o número da peça; "
        "as demais devem conter as respostas dos participantes nos dois testes."
    )
    st.stop()

try:
    tri_raw = read_csv(tri_file)
    di_raw = read_csv(di_file)

    tri_long = dataframe_to_long(
        tri_raw, "Tricromatas", parsing_mode, numeric_map
    )
    di_long = dataframe_to_long(
        di_raw, "Dicromatas", parsing_mode, numeric_map
    )

    data = pd.concat([tri_long, di_long], ignore_index=True)

except Exception as exc:
    st.error(f"Não foi possível processar os arquivos: {exc}")
    st.stop()

invalid = data["Categoria"].isna()
if invalid.any():
    invalid_values = (
        data.loc[invalid, "Resposta_original"]
        .dropna()
        .astype(str)
        .value_counts()
        .rename_axis("Valor não reconhecido")
        .reset_index(name="Frequência")
    )
    st.warning(
        f"Foram encontradas {invalid.sum()} respostas ausentes ou não reconhecidas. "
        "Elas não serão usadas nas análises."
    )
    if not invalid_values.empty:
        st.dataframe(invalid_values, use_container_width=True)

data_valid = data.dropna(subset=["Categoria"]).copy()

if data_valid.empty:
    st.error("Nenhuma resposta válida foi reconhecida.")
    st.stop()

# Ordenação numérica das peças, quando possível.
piece_numeric = pd.to_numeric(data_valid["Peca"], errors="coerce")
if piece_numeric.notna().all():
    piece_order = (
        data_valid.assign(_piece_num=piece_numeric)
        .sort_values("_piece_num")["Peca"]
        .drop_duplicates()
        .tolist()
    )
else:
    piece_order = sorted(data_valid["Peca"].unique().tolist())

tab_overview, tab_distribution, tab_repeat, tab_compare, tab_download = st.tabs(
    [
        "Visão geral",
        "Distribuição por peça",
        "Repetibilidade",
        "Comparação entre grupos",
        "Downloads",
    ]
)

with tab_overview:
    n_participants = data_valid.groupby("Grupo")["Participante"].nunique()
    n_pieces = data_valid["Peca"].nunique()
    n_responses = len(data_valid)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Tricromatas", int(n_participants.get("Tricromatas", 0)))
    m2.metric("Dicromatas", int(n_participants.get("Dicromatas", 0)))
    m3.metric("Peças", int(n_pieces))
    m4.metric("Respostas válidas", int(n_responses))

    st.subheader("Dados organizados")
    st.dataframe(
        data_valid[
            ["Grupo", "Participante", "Teste", "Peca", "Categoria", "Coluna_origem"]
        ],
        use_container_width=True,
        height=360,
    )

with tab_distribution:
    test_filter = st.selectbox(
        "Avaliação",
        ["Ambos os testes", "Teste 1", "Teste 2"],
        key="distribution_test",
    )

    plot_data = data_valid.copy()
    if test_filter != "Ambos os testes":
        plot_data = plot_data[
            plot_data["Teste"] == int(test_filter.split()[-1])
        ]

    distribution = (
        plot_data.groupby(["Grupo", "Peca", "Categoria"])
        .size()
        .rename("N")
        .reset_index()
    )
    distribution["Proporcao"] = distribution.groupby(
        ["Grupo", "Peca"]
    )["N"].transform(lambda x: x / x.sum())

    distribution["Peca"] = pd.Categorical(
        distribution["Peca"], categories=piece_order, ordered=True
    )

    st.subheader("Proporção de respostas por peça")
    fig = px.bar(
        distribution.sort_values("Peca"),
        x="Peca",
        y="Proporcao",
        color="Categoria",
        facet_row="Grupo",
        category_orders={"Categoria": CATEGORIES, "Peca": piece_order},
        barmode="stack",
        labels={"Peca": "Peça", "Proporcao": "Proporção"},
        height=650,
    )
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

    metrics = (
        plot_data.groupby(["Grupo", "Peca"])["Categoria"]
        .agg(
            Categoria_predominante=lambda x: x.value_counts().idxmax(),
            Consenso=consensus,
            Entropia_bits=shannon_entropy,
            N="size",
        )
        .reset_index()
    )
    metrics["Peca"] = pd.Categorical(
        metrics["Peca"], categories=piece_order, ordered=True
    )
    metrics = metrics.sort_values(["Grupo", "Peca"])

    st.subheader("Consenso e entropia por peça")
    st.caption(
        "Consenso próximo de 1 indica elevada concordância. "
        "Entropia próxima de 2 bits indica respostas distribuídas entre as quatro categorias."
    )
    st.dataframe(metrics, use_container_width=True)

    heat = distribution.pivot_table(
        index=["Grupo", "Peca"],
        columns="Categoria",
        values="Proporcao",
        fill_value=0,
    ).reindex(columns=CATEGORIES, fill_value=0)

    st.subheader("Mapa de calor das proporções")
    fig_heat = px.imshow(
        heat,
        aspect="auto",
        labels={"x": "Categoria", "y": "Grupo e peça", "color": "Proporção"},
        zmin=0,
        zmax=1,
    )
    st.plotly_chart(fig_heat, use_container_width=True)

with tab_repeat:
    paired = data_valid.pivot_table(
        index=["Grupo", "Participante", "Peca"],
        columns="Teste",
        values="Categoria",
        aggfunc="first",
    ).reset_index()

    if 1 not in paired.columns or 2 not in paired.columns:
        st.warning("Não foram encontrados dois testes completos para calcular repetibilidade.")
    else:
        paired = paired.dropna(subset=[1, 2]).copy()
        paired["Concordante"] = paired[1] == paired[2]

        participant_repeat = (
            paired.groupby(["Grupo", "Participante"])
            .agg(
                N_pecas=("Peca", "size"),
                Concordancia_percentual=("Concordante", "mean"),
            )
            .reset_index()
        )

        kappas = []
        for (group, participant), subset in paired.groupby(
            ["Grupo", "Participante"]
        ):
            if len(subset) >= 2:
                kappa = cohen_kappa_score(
                    subset[1],
                    subset[2],
                    labels=CATEGORIES,
                )
            else:
                kappa = np.nan
            kappas.append(
                {
                    "Grupo": group,
                    "Participante": participant,
                    "Kappa_de_Cohen": kappa,
                }
            )

        participant_repeat = participant_repeat.merge(
            pd.DataFrame(kappas),
            on=["Grupo", "Participante"],
            how="left",
        )

        st.subheader("Repetibilidade individual")
        participant_repeat["Concordancia_percentual"] *= 100
        st.dataframe(participant_repeat, use_container_width=True)

        summary_repeat = (
            participant_repeat.groupby("Grupo")
            .agg(
                Participantes=("Participante", "nunique"),
                Concordancia_media_pct=("Concordancia_percentual", "mean"),
                Concordancia_mediana_pct=("Concordancia_percentual", "median"),
                Kappa_medio=("Kappa_de_Cohen", "mean"),
                Kappa_mediano=("Kappa_de_Cohen", "median"),
            )
            .reset_index()
        )
        st.subheader("Resumo por grupo")
        st.dataframe(summary_repeat, use_container_width=True)

        fig_box = px.box(
            participant_repeat,
            x="Grupo",
            y="Concordancia_percentual",
            points="all",
            labels={"Concordancia_percentual": "Concordância (%)"},
        )
        st.plotly_chart(fig_box, use_container_width=True)

        st.subheader("Matrizes de transição entre os testes")
        c1, c2 = st.columns(2)
        for container, group in zip(c1, ["Tricromatas"]):
            with container:
                subset = paired[paired["Grupo"] == group]
                matrix = pd.crosstab(subset[1], subset[2]).reindex(
                    index=CATEGORIES, columns=CATEGORIES, fill_value=0
                )
                st.markdown(f"**{group}**")
                st.dataframe(matrix, use_container_width=True)

        with c2:
            group = "Dicromatas"
            subset = paired[paired["Grupo"] == group]
            matrix = pd.crosstab(subset[1], subset[2]).reindex(
                index=CATEGORIES, columns=CATEGORIES, fill_value=0
            )
            st.markdown(f"**{group}**")
            st.dataframe(matrix, use_container_width=True)

        piece_repeat = (
            paired.groupby(["Grupo", "Peca"])["Concordante"]
            .agg(["mean", "size"])
            .reset_index()
            .rename(columns={"mean": "Concordancia", "size": "N"})
        )
        piece_repeat["Concordancia"] *= 100
        st.subheader("Concordância entre testes por peça")
        st.dataframe(piece_repeat, use_container_width=True)

with tab_compare:
    st.subheader("Comparação da distribuição das categorias entre os grupos")
    st.caption(
        "Os testes são feitos separadamente para cada peça. "
        "O valor de p é corrigido por Benjamini–Hochberg."
    )

    compare_test = st.selectbox(
        "Dados usados na comparação",
        ["Ambos os testes", "Somente Teste 1", "Somente Teste 2"],
        key="compare_test",
    )

    comparison_data = data_valid.copy()
    if compare_test == "Somente Teste 1":
        comparison_data = comparison_data[comparison_data["Teste"] == 1]
    elif compare_test == "Somente Teste 2":
        comparison_data = comparison_data[comparison_data["Teste"] == 2]

    comparison_results = compare_groups_per_piece(comparison_data)

    if comparison_results.empty:
        st.warning("Não foi possível calcular os testes com os dados disponíveis.")
    else:
        st.dataframe(
            comparison_results.sort_values("p_ajustado_BH"),
            use_container_width=True,
        )

        sig = comparison_results["Significativo_5pct"].sum()
        st.metric("Peças com diferença após correção", int(sig))

        fig_p = px.scatter(
            comparison_results,
            x="Peca",
            y="p_ajustado_BH",
            size="V_de_Cramer",
            hover_data=["Metodo", "Estatistica", "p"],
            labels={
                "Peca": "Peça",
                "p_ajustado_BH": "p ajustado",
                "V_de_Cramer": "V de Cramér",
            },
        )
        fig_p.add_hline(y=0.05, line_dash="dash")
        fig_p.update_yaxes(type="log")
        st.plotly_chart(fig_p, use_container_width=True)

    st.info(
        "Esta comparação por peça trata as observações como independentes. "
        "Para inferência confirmatória, recomenda-se complementar com um modelo "
        "logístico multinomial de efeitos mistos, incluindo participante como efeito aleatório."
    )

with tab_download:
    st.subheader("Exportar resultados")

    st.download_button(
        "Baixar dados em formato longo",
        csv_download(data_valid),
        file_name="dados_categorizacao_formato_longo.csv",
        mime="text/csv",
    )

    distribution_export = (
        data_valid.groupby(["Grupo", "Teste", "Peca", "Categoria"])
        .size()
        .rename("N")
        .reset_index()
    )
    distribution_export["Proporcao"] = distribution_export.groupby(
        ["Grupo", "Teste", "Peca"]
    )["N"].transform(lambda x: x / x.sum())

    st.download_button(
        "Baixar distribuição por peça",
        csv_download(distribution_export),
        file_name="distribuicao_categorias_por_peca.csv",
        mime="text/csv",
    )

    if "participant_repeat" in locals():
        st.download_button(
            "Baixar repetibilidade individual",
            csv_download(participant_repeat),
            file_name="repetibilidade_individual.csv",
            mime="text/csv",
        )

    if "comparison_results" in locals() and not comparison_results.empty:
        st.download_button(
            "Baixar comparação entre grupos",
            csv_download(comparison_results),
            file_name="comparacao_tricromatas_dicromatas.csv",
            mime="text/csv",
        )

st.divider()
st.caption(
    "Formato recomendado: primeira coluna = peça; depois, duas colunas consecutivas "
    "por participante, correspondentes ao Teste 1 e ao Teste 2."
)
