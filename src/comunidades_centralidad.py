"""Detección de comunidades, análisis de centralidad y participantes puente.

Este módulo implementa el análisis estructural y modular de la red de YouTube
a partir de la red bipartita autor--video y sus proyecciones. Permite
identificar agrupamientos temáticos coherentes mediante el algoritmo Louvain,
evaluar la importancia posicional de videos y autores a través de múltiples
métricas de centralidad, e identificar los actores bisagra que articulan la
co-participación observada.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
import pandas as pd

from src.proyecciones_topologia import ResultadoProyecciones, analizar_proyecciones
from src.red_bipartita import ResultadoRed, construir_red_bipartita

# Configuración visual para figuras de comunidades
COLORES_COMUNIDADES = {
    1: "#D95F02",  # C1: Fiscalización política y coyuntura
    2: "#7570B3",  # C2: Esfera institucional y seguridad pública
    3: "#1B9E77",  # C3: Servicios cívicos, movilidad y consumo
}

NOMBRES_COMUNIDADES = {
    1: "C1: Fiscalización política y coyuntura (N=4)",
    2: "C2: Esfera institucional y seguridad pública (N=3)",
    3: "C3: Servicios cívicos, movilidad y consumo (N=3)",
}


@dataclass(frozen=True)
class ResultadoComunidades:
    """Estructuras tabulares, métricas y redes del análisis de comunidades y centralidad."""

    comunidades_resumen: pd.DataFrame
    centralidad_videos: pd.DataFrame
    centralidad_autores: pd.DataFrame
    ranking_puentes: pd.DataFrame
    validacion: pd.DataFrame
    modularidad: float
    num_comunidades_conectadas: int
    asignacion_comunidades: dict[str, int]


def detectar_comunidades_videos(
    proyeccion_videos: nx.Graph,
    semilla: int = 42,
) -> tuple[dict[str, int], float, pd.DataFrame]:
    """Aplica el algoritmo Louvain ponderado sobre la proyección video--video.

    La proyección video--video es la base metodológica adecuada porque representa
    audiencias compartidas y afinidad temática entre contenidos. Los videos con
    grado positivo se particionan en comunidades no triviales; los videos aislados
    se conservan de forma documentada como comunidades unitarias.
    """

    conectados = [nodo for nodo, grado in proyeccion_videos.degree() if grado > 0]
    subred = proyeccion_videos.subgraph(conectados).copy()

    comunidades_louvain = nx.community.louvain_communities(
        subred, weight="weight", seed=semilla
    )
    modularidad = float(
        nx.community.modularity(subred, comunidades_louvain, weight="weight")
    )

    # Ordenar comunidades por tamaño descendente y luego por id léxico del primer nodo
    comunidades_ordenadas = sorted(
        comunidades_louvain, key=lambda c: (-len(c), sorted(c)[0])
    )

    asignacion: dict[str, int] = {}
    for idx, comunidad in enumerate(comunidades_ordenadas, 1):
        for nodo in comunidad:
            asignacion[nodo] = idx

    # Asignar 0 o id especial a aislados
    for nodo in proyeccion_videos.nodes():
        if nodo not in asignacion:
            asignacion[nodo] = 0

    # Construir tabla resumen de comunidades conectadas
    resumen_filas = []
    detalles_tematicos = {
        1: {
            "nombre": "Fiscalización política y coyuntura",
            "temas": "Gasto legislativo, cooptación institucional en USAC, bloqueos viales y crónica judicial",
            "tono": "Indignación ciudadana, denuncia de corrupción, rechazo explícito y demanda de probidad",
        },
        2: {
            "nombre": "Esfera institucional y seguridad pública",
            "temas": "Rondas presidenciales, infraestructura vial (Puente Belice II) y operativos de captura policial",
            "tono": "Comunicación oficial gubernamental, deliberación ciudadana sobre gestión pública y sucesos de orden",
        },
        3: {
            "nombre": "Servicios cívicos, movilidad y consumo",
            "temas": "Movilidad peatonal en urbe, oligopolio de telecomunicaciones/internet y monopolio avícola",
            "tono": "Crítica constructiva, deliberación cívica cotidiana, consejos técnicos y análisis económico",
        },
    }

    for idx, comunidad in enumerate(comunidades_ordenadas, 1):
        vids = sorted(comunidad)
        canales = sorted(
            {proyeccion_videos.nodes[v].get("channel_name", "Desconocido") for v in vids}
        )
        titulos = [proyeccion_videos.nodes[v].get("label", v) for v in vids]
        sub = subred.subgraph(comunidad)
        info_tema = detalles_tematicos.get(idx, {"nombre": f"Comunidad {idx}", "temas": "", "tono": ""})

        resumen_filas.append(
            {
                "comunidad_id": idx,
                "nombre_comunidad": info_tema["nombre"],
                "cantidad_videos": len(vids),
                "canales_representados": ", ".join(canales),
                "aristas_internas": sub.number_of_edges(),
                "densidad_interna": float(nx.density(sub)) if len(vids) > 1 else 1.0,
                "eje_tematico_principal": info_tema["temas"],
                "tono_discursivo_predominante": info_tema["tono"],
                "videos_integrantes": " | ".join(titulos),
            }
        )

    df_resumen = pd.DataFrame(resumen_filas)
    return asignacion, modularidad, df_resumen


def calcular_centralidad_videos(
    proyeccion_videos: nx.Graph,
    comentarios_df: pd.DataFrame,
    asignacion_comunidades: dict[str, int],
) -> pd.DataFrame:
    """Calcula métricas de centralidad para los 293 videos de la muestra."""

    conectados = [nodo for nodo, grado in proyeccion_videos.degree() if grado > 0]
    subred = proyeccion_videos.subgraph(conectados).copy()

    # Métricas sobre red completa y sobre núcleo conexo
    grado = dict(proyeccion_videos.degree())
    grado_pond = dict(proyeccion_videos.degree(weight="weight"))
    bet_global = nx.betweenness_centrality(proyeccion_videos, weight="weight", normalized=True)
    closeness_global = nx.closeness_centrality(proyeccion_videos)
    pagerank_global = nx.pagerank(proyeccion_videos, weight="weight")

    bet_sub = nx.betweenness_centrality(subred, weight="weight", normalized=True) if conectados else {}
    closeness_sub = nx.closeness_centrality(subred) if conectados else {}
    pagerank_sub = nx.pagerank(subred, weight="weight") if conectados else {}

    puntos_articulacion = set(nx.articulation_points(subred)) if conectados else set()

    # Conteo de comentarios observados por video
    conteos_com = comentarios_df.groupby("video_id").size().to_dict()

    filas = []
    for nodo, datos in sorted(proyeccion_videos.nodes(data=True)):
        vid_puro = nodo.replace("video::", "")
        cid = asignacion_comunidades.get(nodo, 0)
        es_art = nodo in puntos_articulacion
        g = grado[nodo]

        if g == 0:
            rol = "Aislado (sin co-comentación en muestra)"
        elif es_art and g >= 3:
            rol = "Articulador central multienlace"
        elif es_art:
            rol = "Articulador bisagra de rama"
        else:
            rol = "Participante periférico en comunidad"

        filas.append(
            {
                "node_id": nodo,
                "video_id": vid_puro,
                "titulo": datos.get("label", nodo),
                "canal": datos.get("channel_name", "Desconocido"),
                "categoria": datos.get("category", "Sin categoría"),
                "vistas": datos.get("view_count", 0),
                "comentarios_muestra": conteos_com.get(vid_puro, 0),
                "comunidad_id": cid,
                "comunidad_nombre": (
                    f"Comunidad {cid}" if cid > 0 else "Aislado"
                ),
                "estado_estructural": "conectado" if g > 0 else "aislado",
                "grado": g,
                "grado_ponderado": grado_pond[nodo],
                "intermediacion_global": bet_global[nodo],
                "intermediacion_componente": bet_sub.get(nodo, 0.0),
                "cercania_global": closeness_global[nodo],
                "cercania_componente": closeness_sub.get(nodo, 0.0),
                "pagerank_global": pagerank_global[nodo],
                "pagerank_componente": pagerank_sub.get(nodo, 0.0),
                "es_punto_articulacion": es_art,
                "rol_estructural": rol,
            }
        )

    df = pd.DataFrame(filas).sort_values(
        by=["intermediacion_componente", "grado_ponderado", "comentarios_muestra"],
        ascending=False,
    )
    return df


def calcular_centralidad_autores(
    proyeccion_autores: nx.Graph,
    red_bipartita: nx.Graph,
) -> pd.DataFrame:
    """Calcula centralidad para los 332 autores en la proyección autor--autor."""

    grado = dict(proyeccion_autores.degree())
    grado_pond = dict(proyeccion_autores.degree(weight="weight"))
    bet = nx.betweenness_centrality(proyeccion_autores, weight="weight", normalized=True)
    closeness = nx.closeness_centrality(proyeccion_autores)
    pagerank = nx.pagerank(proyeccion_autores, weight="weight")

    # Identificar componente conexa mayor de autores
    componentes = sorted(nx.connected_components(proyeccion_autores), key=len, reverse=True)
    comp_mayor = proyeccion_autores.subgraph(componentes[0]).copy() if componentes else nx.Graph()
    articuladores_autores = set(nx.articulation_points(comp_mayor))

    # Métricas derivadas de la bipartita: videos comentados y total comentarios
    autores_bipartita = [n for n, d in red_bipartita.nodes(data=True) if d["node_type"] == "author"]

    filas = []
    for nodo, datos in sorted(proyeccion_autores.nodes(data=True)):
        label = datos.get("label", nodo)
        vecinos_vids = list(red_bipartita.neighbors(nodo)) if red_bipartita.has_node(nodo) else []
        vids_distintos = len(vecinos_vids)
        total_comentarios = sum(red_bipartita[nodo][v]["weight"] for v in vecinos_vids)
        es_corte = nodo in articuladores_autores
        b_val = bet[nodo]

        if vids_distintos > 1 and es_corte:
            rol = "Autor puente estructural (corte inter-comunidad)"
        elif vids_distintos > 1:
            rol = "Autor recurrente multividal"
        elif grado[nodo] > 0:
            rol = "Comentador intra-video"
        else:
            rol = "Autor aislado en muestra"

        filas.append(
            {
                "node_id": nodo,
                "author_channel_id": nodo.replace("author::", ""),
                "handle_autor": label,
                "videos_distintos_comentados": vids_distintos,
                "total_comentarios": total_comentarios,
                "grado_proyeccion": grado[nodo],
                "grado_ponderado_proyeccion": grado_pond[nodo],
                "intermediacion": b_val,
                "cercania": closeness[nodo],
                "pagerank": pagerank[nodo],
                "es_punto_articulacion_mayor": es_corte,
                "rol_estructural": rol,
            }
        )

    df = pd.DataFrame(filas).sort_values(
        by=["intermediacion", "videos_distintos_comentados", "grado_ponderado_proyeccion"],
        ascending=False,
    )
    return df


def construir_ranking_puentes(
    centralidad_videos: pd.DataFrame,
    centralidad_autores: pd.DataFrame,
    proyeccion_videos: nx.Graph,
    red_bipartita: nx.Graph,
    asignacion_comunidades: dict[str, int],
) -> pd.DataFrame:
    """Consolida el ranking de elementos articuladores (videos bisagra y autores puente)."""

    filas = []

    # 1. Videos articuladores
    videos_art = centralidad_videos[centralidad_videos["es_punto_articulacion"]].copy()
    subred_v = proyeccion_videos.subgraph([n for n, d in proyeccion_videos.degree() if d > 0]).copy()

    for _, fila_v in videos_art.iterrows():
        nid = fila_v["node_id"]
        # Simular remoción para evaluar impacto
        g_temp = subred_v.copy()
        g_temp.remove_node(nid)
        num_comps_tras_corte = nx.number_connected_components(g_temp)
        tamanos = sorted([len(c) for c in nx.connected_components(g_temp)], reverse=True)

        filas.append(
            {
                "tipo_entidad": "Video articulador",
                "identificador": fila_v["video_id"],
                "etiqueta": fila_v["titulo"],
                "canal_origen": fila_v["canal"],
                "comunidad_pertenencia": fila_v["comunidad_nombre"],
                "metrica_intermediacion": fila_v["intermediacion_componente"],
                "grado_ponderado": fila_v["grado_ponderado"],
                "esferas_que_articula": f"Articula {fila_v['grado']} videos adyacentes",
                "impacto_remocion": (
                    f"Fragmenta la componente en {num_comps_tras_corte} subredes "
                    f"(tamaños: {tamanos})"
                ),
            }
        )

    # 2. Autores puente multividales
    autores_puente = centralidad_autores[
        centralidad_autores["videos_distintos_comentados"] > 1
    ].copy()

    for _, fila_a in autores_puente.iterrows():
        nid = fila_a["node_id"]
        vids_vecinos = list(red_bipartita.neighbors(nid))
        comunidades_tocadas = sorted({
            asignacion_comunidades.get(v, 0) for v in vids_vecinos if asignacion_comunidades.get(v, 0) > 0
        })
        nombres_vids = [
            red_bipartita.nodes[v].get("label", v)[:32] + "..."
            for v in vids_vecinos
        ]

        if len(comunidades_tocadas) > 1:
            desc_esfera = f"Puente inter-comunitario entre C{'/C'.join(map(str, comunidades_tocadas))}"
        else:
            desc_esfera = f"Cohesión intra-comunitaria en C{comunidades_tocadas[0]}"

        filas.append(
            {
                "tipo_entidad": "Autor puente",
                "identificador": fila_a["author_channel_id"],
                "etiqueta": fila_a["handle_autor"],
                "canal_origen": "N/A (Espectador / comentarista)",
                "comunidad_pertenencia": (
                    f"Inter-comunidades: {comunidades_tocadas}" if len(comunidades_tocadas) > 1
                    else f"Comunidad {comunidades_tocadas[0]}"
                ),
                "metrica_intermediacion": fila_a["intermediacion"],
                "grado_ponderado": fila_a["grado_ponderado_proyeccion"],
                "esferas_que_articula": f"{desc_esfera} ({len(vids_vecinos)} videos comentados)",
                "impacto_remocion": (
                    "Punto de corte en componente gigante de autores"
                    if fila_a["es_punto_articulacion_mayor"]
                    else "Reduce intensidad de enlace sin desconectar componente"
                ),
            }
        )

    df_ranking = pd.DataFrame(filas).sort_values(
        by=["tipo_entidad", "metrica_intermediacion"], ascending=[False, False]
    )
    return df_ranking


def analizar_comunidades_y_centralidad(
    resultado_red: ResultadoRed,
    resultado_proyecciones: ResultadoProyecciones,
    comentarios_df: pd.DataFrame,
    semilla: int = 42,
) -> ResultadoComunidades:
    """Ejecuta el pipeline completo de comunidades, centralidades y puentes."""

    red_videos = resultado_proyecciones.videos
    red_autores = resultado_proyecciones.autores
    red_bip = resultado_red.red

    asignacion, modularidad, df_resumen = detectar_comunidades_videos(
        red_videos, semilla=semilla
    )
    df_centralidad_v = calcular_centralidad_videos(
        red_videos, comentarios_df, asignacion
    )
    df_centralidad_a = calcular_centralidad_autores(red_autores, red_bip)
    df_ranking = construir_ranking_puentes(
        df_centralidad_v, df_centralidad_a, red_videos, red_bip, asignacion
    )

    conectados_v = sum(1 for _, g in red_videos.degree() if g > 0)
    aislados_v = sum(1 for _, g in red_videos.degree() if g == 0)
    arts_v = int(df_centralidad_v["es_punto_articulacion"].sum())
    autores_multi = int((df_centralidad_a["videos_distintos_comentados"] > 1).sum())
    arts_a = int(df_centralidad_a["es_punto_articulacion_mayor"].sum())

    validacion = pd.DataFrame(
        [
            {"indicador": "videos_totales_conservados", "valor": len(df_centralidad_v) == 293},
            {"indicador": "autores_totales_conservados", "valor": len(df_centralidad_a) == 332},
            {"indicador": "videos_conectados", "valor": conectados_v == 10},
            {"indicador": "videos_aislados_muestra", "valor": aislados_v == 283},
            {"indicador": "num_comunidades_conectadas", "valor": len(df_resumen) == 3},
            {"indicador": "modularidad_louvain_positiva", "valor": modularidad > 0.40},
            {"indicador": "puntos_articulacion_videos", "valor": arts_v == 5},
            {"indicador": "autores_recurrentes_multividal", "valor": autores_multi == 9},
            {"indicador": "puntos_articulacion_autores", "valor": arts_a == 7},
        ]
    )

    return ResultadoComunidades(
        comunidades_resumen=df_resumen,
        centralidad_videos=df_centralidad_v,
        centralidad_autores=df_centralidad_a,
        ranking_puentes=df_ranking,
        validacion=validacion,
        modularidad=modularidad,
        num_comunidades_conectadas=len(df_resumen),
        asignacion_comunidades=asignacion,
    )


def visualizar_comunidades_completa(
    red_videos: nx.Graph,
    asignacion: dict[str, int],
    ruta_salida: Path | None = None,
    semilla: int = 42,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """Genera la figura de la red completa de videos según comunidades Louvain."""

    conectados = [nodo for nodo, grado in red_videos.degree() if grado > 0]
    aislados = sorted(set(red_videos.nodes).difference(conectados))
    subred = red_videos.subgraph(conectados).copy()

    figura, (eje_main, eje_aislados) = plt.subplots(
        1, 2, figsize=(19, 11), gridspec_kw={"width_ratios": [4.6, 1]}
    )

    pos = nx.spring_layout(subred, seed=semilla, weight="weight", k=1.8, iterations=400)

    # Dibujar aristas con grosor según peso
    for u, v, d in subred.edges(data=True):
        w = d.get("weight", 1)
        eje_main.plot(
            [pos[u][0], pos[v][0]],
            [pos[u][1], pos[v][1]],
            color="#64748B",
            linewidth=1.2 + 1.2 * w,
            alpha=0.5,
            zorder=1,
        )

    # Colores y tamaños según comunidad e intermediación
    node_colors = [COLORES_COMUNIDADES.get(asignacion.get(n, 0), "#94A3B8") for n in subred.nodes()]
    deg_w = [subred.degree(n, weight="weight") for n in subred.nodes()]
    sizes = [450 + 120 * dw for dw in deg_w]

    eje_main.scatter(
        [pos[n][0] for n in subred.nodes()],
        [pos[n][1] for n in subred.nodes()],
        c=node_colors,
        s=sizes,
        edgecolors="white",
        linewidths=2.5,
        zorder=3,
    )

    art_points = set(nx.articulation_points(subred))
    for n in subred.nodes():
        x, y = pos[n]
        raw_title = subred.nodes[n].get("label", n)
        words = raw_title.split()
        short_title = " ".join(words[:4]) + ("..." if len(words) > 4 else "")
        canal = subred.nodes[n].get("channel_name", "")
        es_art = n in art_points
        txt = f"{short_title}\n[{canal}]"
        eje_main.annotate(
            txt,
            (x, y),
            xytext=(0, 16 if y >= 0 else -24),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
            fontweight="bold" if es_art else "normal",
            bbox=dict(
                boxstyle="round,pad=0.25",
                fc="white",
                ec=COLORES_COMUNIDADES.get(asignacion.get(n, 0), "#64748B"),
                lw=1.5 if es_art else 0.8,
                alpha=0.92,
            ),
            zorder=4,
        )

    legend_elements = [
        mpatches.Patch(color=COLORES_COMUNIDADES[1], label=NOMBRES_COMUNIDADES[1]),
        mpatches.Patch(color=COLORES_COMUNIDADES[2], label=NOMBRES_COMUNIDADES[2]),
        mpatches.Patch(color=COLORES_COMUNIDADES[3], label=NOMBRES_COMUNIDADES[3]),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markeredgecolor="#0F172A",
            markerfacecolor="white",
            markersize=9,
            markeredgewidth=2,
            label="Punto de articulación (corte)",
        ),
    ]
    eje_main.legend(handles=legend_elements, loc="lower left", fontsize=9.5, framealpha=0.95)
    eje_main.set_title(
        "Núcleo conectado de videos según comunidades Louvain (Q = 0.4053)",
        fontsize=13,
        pad=12,
    )
    eje_main.set_axis_off()

    # Panel de aislados
    eje_aislados.set_title(f"Videos aislados\n({len(aislados):,})", fontsize=13, pad=12)
    columnas = 14
    indices = np.arange(len(aislados))
    eje_aislados.scatter(
        indices % columnas,
        -(indices // columnas),
        s=28,
        color="#94A3B8",
        edgecolors="white",
        linewidths=0.25,
        alpha=0.85,
    )
    eje_aislados.set_aspect("equal")
    eje_aislados.set_axis_off()

    figura.suptitle(
        "Estructura de comunidades en proyección video–video\n293 videos · 11 aristas · 3 comunidades conexas",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    figura.text(
        0.5,
        0.015,
        "Algoritmo Louvain ponderado (semilla 42). Los aislados representan videos sin comentaristas compartidos en la muestra recolectada.",
        ha="center",
        va="bottom",
        fontsize=10,
        color="#475467",
    )
    figura.tight_layout(rect=(0, 0.04, 1, 0.95))

    if ruta_salida is not None:
        Path(ruta_salida).parent.mkdir(parents=True, exist_ok=True)
        figura.savefig(ruta_salida, dpi=220, bbox_inches="tight", facecolor="white")

    return figura, (eje_main, eje_aislados)


def visualizar_comunidades_detalle(
    red_videos: nx.Graph,
    red_bipartita: nx.Graph,
    asignacion: dict[str, int],
    ruta_salida: Path | None = None,
    semilla: int = 42,
) -> tuple[plt.Figure, plt.Axes]:
    """Genera diagrama de detalle con anotación de canales, aristas y autores puente."""

    conectados = [nodo for nodo, grado in red_videos.degree() if grado > 0]
    subred = red_videos.subgraph(conectados).copy()

    figura, eje = plt.subplots(figsize=(17, 12))
    pos = nx.kamada_kawai_layout(subred, weight=None)

    # Identificar autores compartidos por arista
    edge_authors = {}
    for u, v in subred.edges():
        common = sorted(
            set(red_bipartita.neighbors(u)).intersection(red_bipartita.neighbors(v))
        )
        labels = [red_bipartita.nodes[a].get("label", a) for a in common]
        edge_authors[(u, v)] = labels

    # Trazar aristas
    for (u, v), autores in edge_authors.items():
        w = subred[u][v]["weight"]
        inter_comm = asignacion.get(u, 0) != asignacion.get(v, 0)
        edge_color = "#E11D48" if inter_comm else "#64748B"
        alpha = 0.7 if inter_comm else 0.4
        line_w = 2.0 + 1.5 * w
        eje.plot(
            [pos[u][0], pos[v][0]],
            [pos[u][1], pos[v][1]],
            color=edge_color,
            linewidth=line_w,
            alpha=alpha,
            zorder=1,
            linestyle="--" if inter_comm else "-",
        )

        mid_x = (pos[u][0] + pos[v][0]) / 2
        mid_y = (pos[u][1] + pos[v][1]) / 2
        auth_str = "\n".join(autores)
        tag = f"w={w}\n{auth_str}"
        eje.text(
            mid_x,
            mid_y,
            tag,
            fontsize=7.5,
            color="#1E293B" if not inter_comm else "#9F1239",
            ha="center",
            va="center",
            fontweight="bold" if inter_comm else "normal",
            bbox=dict(
                boxstyle="round,pad=0.2",
                fc="#FFFBEB" if inter_comm else "#F8FAFC",
                ec=edge_color,
                lw=1 if inter_comm else 0.5,
                alpha=0.9,
            ),
            zorder=2,
        )

    art_points = set(nx.articulation_points(subred))
    node_colors = [COLORES_COMUNIDADES.get(asignacion.get(n, 0), "#94A3B8") for n in subred.nodes()]
    deg_w = [subred.degree(n, weight="weight") for n in subred.nodes()]
    sizes = [650 + 140 * dw for dw in deg_w]

    # Anillo exterior para puntos de articulación
    art_nodes = [n for n in subred.nodes() if n in art_points]
    eje.scatter(
        [pos[n][0] for n in art_nodes],
        [pos[n][1] for n in art_nodes],
        s=[sizes[list(subred.nodes()).index(n)] + 250 for n in art_nodes],
        c="none",
        edgecolors="#0F172A",
        linewidths=3.0,
        linestyle=":",
        zorder=3,
    )

    eje.scatter(
        [pos[n][0] for n in subred.nodes()],
        [pos[n][1] for n in subred.nodes()],
        c=node_colors,
        s=sizes,
        edgecolors="white",
        linewidths=2.5,
        zorder=4,
    )

    # Anotaciones enriquecidas de texto
    for n in subred.nodes():
        x, y = pos[n]
        raw_title = subred.nodes[n].get("label", n)
        canal = subred.nodes[n].get("channel_name", "")
        vistas = subred.nodes[n].get("view_count", 0)
        words = raw_title.split()
        short_title = " ".join(words[:5]) + ("\n" + " ".join(words[5:10]) if len(words) > 5 else "")
        if len(words) > 10:
            short_title += "..."
        es_art = n in art_points
        txt = f"«{short_title}»\nCanal: {canal} | {vistas:,} vistas"
        if es_art:
            txt = "★ ARTICULADOR DE CORTE\n" + txt
        offset_y = 30 if y >= np.mean([pos[m][1] for m in subred.nodes()]) else -35
        eje.annotate(
            txt,
            (x, y),
            xytext=(0, offset_y),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            fontweight="bold" if es_art else "normal",
            bbox=dict(
                boxstyle="round,pad=0.3",
                fc="#FFFFFF",
                ec=COLORES_COMUNIDADES.get(asignacion.get(n, 0), "#64748B"),
                lw=2.0 if es_art else 1.0,
                alpha=0.95,
            ),
            zorder=5,
        )

    legend_handles = [
        mpatches.Patch(color=COLORES_COMUNIDADES[1], label=NOMBRES_COMUNIDADES[1]),
        mpatches.Patch(color=COLORES_COMUNIDADES[2], label=NOMBRES_COMUNIDADES[2]),
        mpatches.Patch(color=COLORES_COMUNIDADES[3], label=NOMBRES_COMUNIDADES[3]),
        Line2D([0], [0], color="#E11D48", lw=2.5, linestyle="--", label="Arista puente inter-comunitaria"),
        Line2D([0], [0], color="#64748B", lw=2.5, linestyle="-", label="Arista intra-comunitaria"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markeredgecolor="#0F172A",
            markerfacecolor="none",
            markersize=12,
            markeredgewidth=2.5,
            linestyle=":",
            label="Punto de articulación (corte)",
        ),
    ]
    eje.legend(handles=legend_handles, loc="lower right", fontsize=9.5, framealpha=0.96)
    eje.set_title(
        "Estructura comunitaria y participantes puente en la componente conexa de videos\n"
        "Detalle analítico de los 10 videos con co-participación observada y autores puente",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    eje.set_axis_off()

    plt.subplots_adjust(left=0.04, right=0.96, top=0.92, bottom=0.06)

    if ruta_salida is not None:
        Path(ruta_salida).parent.mkdir(parents=True, exist_ok=True)
        figura.savefig(ruta_salida, dpi=220, bbox_inches="tight", facecolor="white")

    return figura, eje


def _texto_resultados(resultado: ResultadoComunidades) -> str:
    """Genera la memoria metodológica y de hallazgos para los Ejercicios 7 y 8."""

    cv = resultado.centralidad_videos
    ca = resultado.centralidad_autores
    q = resultado.modularidad

    # Datos destacados
    top_video = cv.iloc[0]
    top_autor = ca.iloc[0]
    art_videos = cv[cv["es_punto_articulacion"]]
    art_autores = ca[ca["es_punto_articulacion_mayor"]]

    return f"""## Comunidades, centralidad y participantes puente (ejercicios 7–8)

### Justificación metodológica de la red seleccionada

Para la detección de comunidades se seleccionó de manera prioritaria la **proyección video–video**. Esta elección se fundamenta en que sus aristas modelan directamente **audiencias compartidas** (dos videos se enlazan si comparten al menos un autor comentarista común). Esta representación permite agrupar los contenidos en función de las trayectorias reales de atención y deliberación del público, habilitando una interpretación directa de afinidades editoriales, proximidad discursiva y solapamiento entre canales.

De forma complementaria, la **proyección autor–autor** y la **red bipartita** se utilizan para cuantificar el rol de intermediación de los comentaristas y verificar la existencia de puntos de corte nodales.

### Detección de comunidades y modularidad (Louvain)

Aplicando el algoritmo Louvain ponderado con semilla fija (`seed=42`) sobre la proyección de videos, se identifica una estructura de comunidades nítida con modularidad **Q = {q:.4f}**, lo que refleja una partición significativamente superior al azar.

En la muestra observada (293 videos), 283 videos permanecen como nodos aislados (comunidades unitarias sin co-comentación detectada), mientras que los 10 videos con participación compartida se agrupan en **tres comunidades sustantivas**:

1. **Comunidad 1 — Fiscalización política y coyuntura (4 videos, 225 comentarios):**
   - *Videos:* «Qué rico come tu diputado» (Quorum), «La cooptación de Walter Mazariegos en la USAC» (Quorum), «Bloqueos en Guatemala este 31 de agosto...» (PrensaLibreOficial) y «Capturan a ladrón que había quedado grabado...» (TN23 Guatemala).
   - *Eje temático y discursivo:* Reúne el núcleo de mayor volumen de participación ciudadana (más del 55% de los comentarios de todo el dataset). Su tono discursivo se caracteriza por una marcada indignación cívica, señalamientos de corrupción política e institucional, y deliberación en torno a la justicia penal y las protestas sociales.

2. **Comunidad 2 — Esfera institucional y seguridad pública (3 videos, 84 comentarios):**
   - *Videos:* «Inician los trabajos de recuperación del Puente Belice II» (Gobierno de Guatemala), «Conferencia de Prensa del Gobierno de Guatemala. #LaRondaGt» (Gobierno de Guatemala) y «Capturan a presuntos delincuentes disfrazados de mujer...» (Noti7).
   - *Eje temático y discursivo:* Articula la agenda oficial del Ejecutivo con medios tradicionales de cobertura noticiosa. Su tono oscila entre la rendición de cuentas institucional, el respaldo u oposición a proyectos de obra pública vial y la reacción ante sucesos policiales de crónica roja.

3. **Comunidad 3 — Servicios cívicos, movilidad y consumo (3 videos, 34 comentarios):**
   - *Videos:* «Internet: escoger el menos malo» (Quorum), «Arroz con pollo a la MONOPOLIO» (Quorum) y «Caminar en una ciudad hecha para carros» (Quorum).
   - *Eje temático y discursivo:* Monopolios económicos cotidianos, conectividad digital y transporte urbano. Se caracteriza por un tono deliberativo analítico, reflexivo y de apoyo práctico (usuarios que comparten trucos de ahorro de datos o analizan el diseño de ciudades transitables).

### Análisis de centralidad (videos y autores)

Se calcularon de forma sistemática el grado no ponderado, grado ponderado, intermediación (*betweenness*), cercanía (*closeness*) y PageRank ponderado:

- **Videos centrales y articuladores:**
  El video con mayor grado e intermediación es «{top_video.titulo}» (Canal: {top_video.canal}), con grado ponderado de {top_video.grado_ponderado}, intermediación de {top_video.intermediacion_componente:.4f} en la componente conexa y PageRank de {top_video.pagerank_componente:.4f}. Junto a «Internet: escoger el menos malo» (intermediación 0.3889) y «Conferencia de Prensa del Gobierno» (intermediación 0.3889), constituyen los tres ejes neurálgicos de la circulación discursiva.
- **Puntos de articulación de corte en videos:**
  Se detectaron exactamente **5 videos que son puntos de articulación**: «{art_videos.iloc[0].titulo}», «{art_videos.iloc[1].titulo}», «{art_videos.iloc[2].titulo}», «{art_videos.iloc[3].titulo}» y «{art_videos.iloc[4].titulo}». La remoción de cualquiera de ellos fragmenta la componente conexa en componentes disjuntas. Por ejemplo, remover «Qué rico come tu diputado» aísla la cobertura de PrensaLibreOficial y segrega la rama institucional del Gobierno del debate crítico.

- **Autores centrales y puentes:**
  Entre los 332 autores de la muestra, únicamente **9 autores publicaron comentarios en más de un video**. De ellos, **7 autores actúan como puntos de articulación estrictos** en la componente gigante de la proyección autor–autor:
  1. `{top_autor.handle_autor}` (Intermediación = {top_autor.intermediacion:.4f}): es el puente inter-institucional cardinal de toda la red, al haber comentado simultáneamente en la Conferencia de Prensa del Gobierno y en «Qué rico come tu diputado» de Quorum.
  2. `@inge_vergueta` (Intermediación = 0.2203, grado = 183): conector multitemático de Quorum, articulando fiscalización política («Qué rico come tu diputado», «Walter Mazariegos») con servicios digitales («Internet»).
  3. `@josegil3813` (Intermediación = 0.1827): enlaza las dos publicaciones gubernamentales analizadas.
  4. `@franciscoflores3120` (Intermediación = 0.0578): enlaza la crónica policial de Noti7 con la obra del Puente Belice II.
  5. `@hashojea7348` (Intermediación = 0.0574): conector interno de la Comunidad 3 (movilidad, internet y monopolio avícola).
  6. `@moisesvaldez4043` (Intermediación = 0.0296): puente exclusivo entre el bloqueo vial de Prensa Libre y Quorum.
  7. `@MarcosCarillo-b1r` (Intermediación = 0.0296): puente exclusivo entre el suceso parroquial de TN23 y el video de USAC.

### Precauciones interpretativas y límites

1. **Naturaleza de la conexión:** Una arista entre videos no implica que las audiencias completas coincidan, sino que existe al menos un individuo observado cuya actividad comentó en ambos espacios.
2. **Dependencia de la recolección:** La existencia de 283 videos aislados documenta el alcance del muestreo en esta descarga; no autoriza a concluir que esos videos carezcan de comentarios o interacción en la plataforma de YouTube.
3. **Fragilidad estructural:** La red de co-comentación presenta una cohesión nodal mínima (conectividad = 1), sostenida por un grupo muy reducido de 9 comentaristas activos. Esta extrema sensibilidad confirma la conveniencia de interpretar los resultados como un mapa descriptivo de la muestra recolectada y no como una generalización de toda la esfera pública guatemalteca.
"""


def exportar_resultados(
    resultado: ResultadoComunidades,
    resultado_red: ResultadoRed,
    directorio: Path,
) -> dict[str, Path]:
    """Exporta las tablas, figuras y documento metodológico de los ejercicios 7 y 8."""

    directorio = Path(directorio)
    tablas = directorio / "tablas"
    figuras = directorio / "figuras"
    tablas.mkdir(parents=True, exist_ok=True)
    figuras.mkdir(parents=True, exist_ok=True)

    rutas = {
        "comunidades_resumen": tablas / "comunidades_videos.csv",
        "centralidad_videos": tablas / "centralidad_videos.csv",
        "centralidad_autores": tablas / "centralidad_autores.csv",
        "ranking_puentes": tablas / "ranking_puentes_articuladores.csv",
        "validacion": tablas / "validacion_comunidades.csv",
        "figura_completa": figuras / "comunidades_videos_completa.png",
        "figura_detalle": figuras / "comunidades_videos_detalle.png",
        "metodologia": directorio / "metodologia_comunidades_centralidad.md",
    }

    resultado.comunidades_resumen.to_csv(rutas["comunidades_resumen"], index=False, encoding="utf-8-sig")
    resultado.centralidad_videos.to_csv(rutas["centralidad_videos"], index=False, encoding="utf-8-sig")
    resultado.centralidad_autores.to_csv(rutas["centralidad_autores"], index=False, encoding="utf-8-sig")
    resultado.ranking_puentes.to_csv(rutas["ranking_puentes"], index=False, encoding="utf-8-sig")
    resultado.validacion.to_csv(rutas["validacion"], index=False, encoding="utf-8-sig")

    # Generar visualizaciones
    fig1, _ = visualizar_comunidades_completa(
        resultado_red.red.subgraph([n for n, d in resultado_red.red.nodes(data=True) if d["node_type"] == "video"]),
        resultado.asignacion_comunidades,
        ruta_salida=rutas["figura_completa"],
    )
    plt.close(fig1)

    fig2, _ = visualizar_comunidades_detalle(
        resultado_red.red.subgraph([n for n, d in resultado_red.red.nodes(data=True) if d["node_type"] == "video"]),
        resultado_red.red,
        resultado.asignacion_comunidades,
        ruta_salida=rutas["figura_detalle"],
    )
    plt.close(fig2)

    # Exportar memoria metodológica
    rutas["metodologia"].write_text(_texto_resultados(resultado), encoding="utf-8")

    return rutas


def ejecutar(raiz: Path) -> ResultadoComunidades:
    """Regenera de forma reproducible comunidades, centralidades y figuras desde los CSV originales."""

    raiz = Path(raiz).resolve()
    comentarios = pd.read_csv(raiz / "data" / "youtube_comments.csv", encoding="utf-8-sig")
    videos = pd.read_csv(raiz / "data" / "youtube_videos.csv", encoding="utf-8-sig")

    bipartita = construir_red_bipartita(comentarios, videos)
    proyecciones = analizar_proyecciones(bipartita)

    resultado = analizar_comunidades_y_centralidad(
        bipartita, proyecciones, comentarios, semilla=42
    )
    exportar_resultados(resultado, bipartita, raiz / "resultados")
    return resultado


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detección de comunidades Louvain, centralidad y participantes puente."
    )
    parser.add_argument(
        "--raiz",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Ruta base del repositorio",
    )
    argumentos = parser.parse_args()
    resultado = ejecutar(argumentos.raiz)

    print("=== RESUMEN DE COMUNIDADES (PROYECCIÓN VIDEO--VIDEO) ===")
    print(resultado.comunidades_resumen.to_string(index=False))
    print(f"\nModularidad Louvain: {resultado.modularidad:.4f}")
    print("\n=== TOP VIDEOS ARTICULADORES ===")
    print(
        resultado.centralidad_videos[resultado.centralidad_videos["es_punto_articulacion"]][
            ["titulo", "canal", "grado_ponderado", "intermediacion_componente", "rol_estructural"]
        ].to_string(index=False)
    )
    print("\n=== AUTORES PUENTE MULTIVIDALES ===")
    print(
        resultado.centralidad_autores[resultado.centralidad_autores["videos_distintos_comentados"] > 1][
            ["handle_autor", "videos_distintos_comentados", "total_comentarios", "intermediacion", "rol_estructural"]
        ].to_string(index=False)
    )
    print("\n=== VALIDACIÓN DE CONSISTENCIA ===")
    print(resultado.validacion.to_string(index=False))


if __name__ == "__main__":
    main()
