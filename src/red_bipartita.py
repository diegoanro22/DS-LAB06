"""Construcción, validación y exportación de la red autor-video."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
import pandas as pd


COLUMNAS_COMENTARIOS = {
    "comment_id",
    "video_id",
    "author_channel_id",
    "author_name",
    "author_handle",
}
COLUMNAS_VIDEOS = {
    "video_id",
    "title",
    "channel_name",
    "category",
    "view_count",
}


@dataclass(frozen=True)
class ResultadoRed:
    """Objetos derivados de una misma versión de los datos."""

    nodos: pd.DataFrame
    aristas: pd.DataFrame
    red: nx.Graph
    validacion: pd.DataFrame


def _validar_columnas(df: pd.DataFrame, requeridas: set[str], nombre: str) -> None:
    faltantes = sorted(requeridas.difference(df.columns))
    if faltantes:
        raise ValueError(f"Faltan columnas en {nombre}: {', '.join(faltantes)}")


def _primer_texto(serie: pd.Series, respaldo: str) -> str:
    valores = serie.dropna().astype(str).str.strip()
    valores = valores[valores.ne("")]
    return valores.iloc[0] if not valores.empty else respaldo


def _atributos_validos(fila: pd.Series, columnas: list[str]) -> dict[str, object]:
    atributos: dict[str, object] = {}
    for columna in columnas:
        valor = fila[columna]
        if pd.isna(valor):
            continue
        if isinstance(valor, np.generic):
            valor = valor.item()
        atributos[columna] = valor
    return atributos


def construir_red_bipartita(
    comentarios: pd.DataFrame,
    videos: pd.DataFrame,
) -> ResultadoRed:
    """Crea la red completa y sus tablas a partir de comentarios y videos.

    Cada pareja autor-video produce una sola arista no dirigida. Su peso es el
    número de comentarios observados para esa pareja. Se conservan todos los
    videos recolectados, incluso cuando no tienen una arista en la muestra.
    """

    _validar_columnas(comentarios, COLUMNAS_COMENTARIOS, "comentarios")
    _validar_columnas(videos, COLUMNAS_VIDEOS, "videos")

    comentarios_red = comentarios.copy()
    for columna in ["comment_id", "video_id", "author_channel_id"]:
        comentarios_red[columna] = comentarios_red[columna].astype("string").str.strip()

    if comentarios_red["comment_id"].isna().any():
        raise ValueError("Hay comentarios sin comment_id.")
    if comentarios_red["comment_id"].duplicated().any():
        raise ValueError("comment_id debe ser único para contar comentarios.")
    if comentarios_red[["video_id", "author_channel_id"]].isna().any().any():
        raise ValueError("La red requiere video_id y author_channel_id en cada comentario.")

    videos_red = videos.copy()
    videos_red["video_id"] = videos_red["video_id"].astype("string").str.strip()
    if videos_red["video_id"].isna().any() or videos_red["video_id"].duplicated().any():
        raise ValueError("video_id debe existir y ser único en la tabla de videos.")

    ids_sin_video = sorted(set(comentarios_red["video_id"]) - set(videos_red["video_id"]))
    if ids_sin_video:
        raise ValueError(f"Hay {len(ids_sin_video)} video_id sin correspondencia.")

    aristas = (
        comentarios_red.groupby(["author_channel_id", "video_id"], as_index=False)
        .agg(weight=("comment_id", "size"))
        .sort_values(["video_id", "author_channel_id"], kind="stable")
        .reset_index(drop=True)
    )
    aristas["weight"] = aristas["weight"].astype("int64")

    etiquetas_autor = (
        comentarios_red.groupby("author_channel_id", sort=True)
        .agg(
            author_name=("author_name", lambda x: _primer_texto(x, "")),
            author_handle=("author_handle", lambda x: _primer_texto(x, "")),
        )
        .reset_index()
    )
    etiquetas_autor["label"] = etiquetas_autor.apply(
        lambda fila: fila["author_name"]
        or fila["author_handle"]
        or fila["author_channel_id"],
        axis=1,
    )

    videos_con_arista = set(aristas["video_id"])
    vistas = pd.to_numeric(videos_red["view_count"], errors="coerce").astype("Int64")
    nodos_autor = [
        {
            "node_id": f"author::{fila.author_channel_id}",
            "node_type": "author",
            "label": fila.label,
            "author_channel_id": fila.author_channel_id,
            "video_id": pd.NA,
            "title": pd.NA,
            "channel_name": pd.NA,
            "category": pd.NA,
            "view_count": pd.NA,
            "has_observed_edge": True,
        }
        for fila in etiquetas_autor.itertuples(index=False)
    ]
    nodos_video = [
        {
            "node_id": f"video::{fila.video_id}",
            "node_type": "video",
            "label": fila.title if pd.notna(fila.title) else fila.video_id,
            "author_channel_id": pd.NA,
            "video_id": fila.video_id,
            "title": fila.title,
            "channel_name": fila.channel_name,
            "category": fila.category,
            "view_count": vistas.iloc[indice],
            "has_observed_edge": fila.video_id in videos_con_arista,
        }
        for indice, fila in enumerate(videos_red.itertuples(index=False))
    ]
    nodos = pd.DataFrame.from_records(nodos_autor + nodos_video)
    nodos["view_count"] = pd.array(nodos["view_count"], dtype="Int64")

    red = nx.Graph(
        name="Red bipartita autor-video",
        edge_definition="Un autor publico al menos un comentario en un video.",
        weight_definition="Numero de comentarios observados de la pareja autor-video.",
    )
    columnas_atributos = [
        "node_type",
        "label",
        "author_channel_id",
        "video_id",
        "title",
        "channel_name",
        "category",
        "view_count",
        "has_observed_edge",
    ]
    for _, fila in nodos.iterrows():
        red.add_node(fila["node_id"], **_atributos_validos(fila, columnas_atributos))
    for fila in aristas.itertuples(index=False):
        red.add_edge(
            f"author::{fila.author_channel_id}",
            f"video::{fila.video_id}",
            weight=int(fila.weight),
            author_channel_id=str(fila.author_channel_id),
            video_id=str(fila.video_id),
        )

    tipos_validos = all(
        red.nodes[u]["node_type"] != red.nodes[v]["node_type"]
        for u, v in red.edges
    )
    suma_pesos = int(aristas["weight"].sum())
    total_comentarios = int(len(comentarios_red))
    validacion = pd.DataFrame(
        [
            {"indicador": "comentarios_analizados", "valor": total_comentarios},
            {"indicador": "suma_pesos_aristas", "valor": suma_pesos},
            {"indicador": "pesos_equivalen_a_comentarios", "valor": suma_pesos == total_comentarios},
            {"indicador": "autores", "valor": int(len(nodos_autor))},
            {"indicador": "videos_recolectados", "valor": int(len(nodos_video))},
            {"indicador": "videos_con_arista_observada", "valor": int(len(videos_con_arista))},
            {"indicador": "nodos", "valor": int(red.number_of_nodes())},
            {"indicador": "aristas_autor_video", "valor": int(red.number_of_edges())},
            {"indicador": "nodos_aislados", "valor": int(nx.number_of_isolates(red))},
            {"indicador": "componentes", "valor": int(nx.number_connected_components(red))},
            {"indicador": "todas_las_aristas_son_bipartitas", "valor": tipos_validos},
        ]
    )

    if suma_pesos != total_comentarios:
        raise AssertionError("La suma de pesos no coincide con los comentarios analizados.")
    if not tipos_validos:
        raise AssertionError("Se detectó una arista entre nodos del mismo tipo.")

    return ResultadoRed(nodos=nodos, aristas=aristas, red=red, validacion=validacion)


def visualizar_red_completa(
    red: nx.Graph,
    ruta_salida: Path | None = None,
    semilla: int = 42,
) -> tuple[plt.Figure, plt.Axes]:
    """Dibuja todos los nodos, aristas y componentes de la red."""

    nodos_con_arista = [nodo for nodo, grado in red.degree() if grado > 0]
    nodos_aislados = sorted(set(red.nodes).difference(nodos_con_arista))
    subred = red.subgraph(nodos_con_arista)

    posiciones: dict[str, np.ndarray] = {}
    if nodos_con_arista:
        posiciones.update(
            nx.spring_layout(
                subred,
                seed=semilla,
                weight="weight",
                k=1.5 / np.sqrt(max(len(nodos_con_arista), 1)),
                iterations=250,
                scale=1.0,
            )
        )

    autores = [n for n in nodos_con_arista if red.nodes[n]["node_type"] == "author"]
    videos = [n for n in nodos_con_arista if red.nodes[n]["node_type"] == "video"]
    pesos = np.array([subred.edges[arista]["weight"] for arista in subred.edges], dtype=float)
    anchos = 0.25 + np.log1p(pesos) * 0.45 if len(pesos) else []

    figura, (eje, eje_aislados) = plt.subplots(
        1,
        2,
        figsize=(18, 11),
        gridspec_kw={"width_ratios": [4.5, 1]},
    )
    nx.draw_networkx_edges(
        subred,
        posiciones,
        ax=eje,
        edge_color="#667085",
        width=anchos,
        alpha=0.18,
    )
    nx.draw_networkx_nodes(
        subred,
        posiciones,
        nodelist=autores,
        node_color="#2878B5",
        node_shape="o",
        node_size=18,
        linewidths=0,
        alpha=0.82,
        ax=eje,
    )
    nx.draw_networkx_nodes(
        subred,
        posiciones,
        nodelist=videos,
        node_color="#E76F51",
        node_shape="s",
        node_size=46,
        linewidths=0.25,
        edgecolors="white",
        alpha=0.9,
        ax=eje,
    )

    eje.set_title("Componentes con participación observada", fontsize=13, pad=12)
    eje_aislados.set_title(
        f"Videos aislados\n({len(nodos_aislados):,})",
        fontsize=13,
        pad=12,
    )
    if nodos_aislados:
        columnas = 14
        indices = np.arange(len(nodos_aislados))
        eje_aislados.scatter(
            indices % columnas,
            -(indices // columnas),
            s=30,
            marker="s",
            color="#E76F51",
            edgecolors="white",
            linewidths=0.25,
            alpha=0.9,
        )
        eje_aislados.set_aspect("equal")
    eje.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#2878B5", markersize=7, label="Autor"),
            Line2D([0], [0], marker="s", color="none", markerfacecolor="#E76F51", markersize=8, label="Video"),
        ],
        loc="upper right",
        frameon=False,
    )
    eje.set_axis_off()
    eje_aislados.set_axis_off()
    figura.suptitle(
        "Red bipartita completa autor-video\n"
        f"{red.number_of_nodes():,} nodos · {red.number_of_edges():,} aristas",
        fontsize=17,
        y=0.98,
    )
    figura.text(
        0.5,
        0.015,
        "Los videos aislados no tuvieron comentarios recuperados en la muestra; "
        "esto no implica ausencia de actividad en YouTube.",
        ha="center",
        va="bottom",
        fontsize=10,
        color="#475467",
    )
    figura.tight_layout(rect=(0, 0.04, 1, 0.94))

    if ruta_salida is not None:
        ruta_salida = Path(ruta_salida)
        ruta_salida.parent.mkdir(parents=True, exist_ok=True)
        figura.savefig(ruta_salida, dpi=220, bbox_inches="tight", facecolor="white")
    return figura, eje


def exportar_resultados(resultado: ResultadoRed, directorio: Path) -> dict[str, Path]:
    """Exporta tablas, red GEXF y la figura completa."""

    directorio = Path(directorio)
    tablas = directorio / "tablas"
    figuras = directorio / "figuras"
    tablas.mkdir(parents=True, exist_ok=True)
    figuras.mkdir(parents=True, exist_ok=True)

    rutas = {
        "nodos": tablas / "nodos_red_bipartita.csv",
        "aristas": tablas / "aristas_red_bipartita.csv",
        "validacion": tablas / "validacion_red_bipartita.csv",
        "red": directorio / "red_bipartita.gexf",
        "figura": figuras / "red_bipartita_completa.png",
    }
    resultado.nodos.to_csv(rutas["nodos"], index=False, encoding="utf-8-sig")
    resultado.aristas.to_csv(rutas["aristas"], index=False, encoding="utf-8-sig")
    resultado.validacion.to_csv(rutas["validacion"], index=False, encoding="utf-8-sig")
    nx.write_gexf(resultado.red, rutas["red"])
    figura, _ = visualizar_red_completa(resultado.red, rutas["figura"])
    plt.close(figura)
    return rutas


def ejecutar(raiz: Path) -> ResultadoRed:
    """Carga los CSV originales y regenera todos los productos de la red."""

    raiz = Path(raiz).resolve()
    comentarios = pd.read_csv(raiz / "data" / "youtube_comments.csv", encoding="utf-8-sig")
    videos = pd.read_csv(raiz / "data" / "youtube_videos.csv", encoding="utf-8-sig")
    resultado = construir_red_bipartita(comentarios, videos)
    exportar_resultados(resultado, raiz / "resultados")
    return resultado


def main() -> None:
    analizador = argparse.ArgumentParser(description="Genera la red bipartita autor-video.")
    analizador.add_argument(
        "--raiz",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Directorio raíz del proyecto.",
    )
    argumentos = analizador.parse_args()
    resultado = ejecutar(argumentos.raiz)
    print(resultado.validacion.to_string(index=False))


if __name__ == "__main__":
    main()
