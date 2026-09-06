"""Proyecciones y topología de la red bipartita autor--video.

Las proyecciones conservan todos los nodos de su partición. Por tanto, un
video sin comentarios recuperados queda como aislado en la proyección de
videos; esto permite distinguir la ausencia de participación *observada* de
una ausencia real de actividad en YouTube.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
import pandas as pd

from src.red_bipartita import ResultadoRed, construir_red_bipartita


@dataclass(frozen=True)
class ResultadoProyecciones:
    """Redes derivadas y productos tabulares del análisis estructural."""

    autores: nx.Graph
    videos: nx.Graph
    metricas: pd.DataFrame
    distribucion_grados: pd.DataFrame
    componentes: pd.DataFrame
    nodos_estructura: pd.DataFrame
    validacion: pd.DataFrame


def _peso_entero(valor: object) -> int:
    """Devuelve el peso como entero seguro para atributos de NetworkX."""

    return int(valor) if pd.notna(valor) else 0


def construir_proyecciones(red_bipartita: nx.Graph) -> tuple[nx.Graph, nx.Graph]:
    """Construye proyecciones ponderadas autor--autor y video--video.

    En la proyección de autores, ``weight`` es el número de videos distintos
    comentados por ambos autores. En la proyección de videos, es el número de
    autores distintos compartidos. Un comentario repetido en la misma pareja
    autor--video no incrementa estos pesos, tal como exige la definición.
    """

    autores = sorted(
        nodo for nodo, datos in red_bipartita.nodes(data=True)
        if datos["node_type"] == "author"
    )
    videos = sorted(
        nodo for nodo, datos in red_bipartita.nodes(data=True)
        if datos["node_type"] == "video"
    )

    proyeccion_autores = nx.Graph(
        name="Proyeccion autor-autor",
        node_type="author",
        edge_definition="Dos autores comentaron al menos un mismo video.",
        weight_definition="Numero de videos distintos comentados por ambos autores.",
    )
    proyeccion_videos = nx.Graph(
        name="Proyeccion video-video",
        node_type="video",
        edge_definition="Dos videos recibieron comentarios de al menos un mismo autor.",
        weight_definition="Numero de autores distintos compartidos por ambos videos.",
    )
    proyeccion_autores.add_nodes_from((nodo, dict(red_bipartita.nodes[nodo])) for nodo in autores)
    proyeccion_videos.add_nodes_from((nodo, dict(red_bipartita.nodes[nodo])) for nodo in videos)

    for video in videos:
        vecinos = sorted(red_bipartita.neighbors(video))
        for origen, destino in combinations(vecinos, 2):
            if proyeccion_autores.has_edge(origen, destino):
                proyeccion_autores[origen][destino]["weight"] += 1
            else:
                proyeccion_autores.add_edge(origen, destino, weight=1)
    for autor in autores:
        vecinos = sorted(red_bipartita.neighbors(autor))
        for origen, destino in combinations(vecinos, 2):
            if proyeccion_videos.has_edge(origen, destino):
                proyeccion_videos[origen][destino]["weight"] += 1
            else:
                proyeccion_videos.add_edge(origen, destino, weight=1)

    return proyeccion_autores, proyeccion_videos


def _cohesion(red: nx.Graph) -> int:
    """Cohesión como conectividad por nodos global; vale 0 si está desconexa."""

    if red.number_of_nodes() < 2 or not nx.is_connected(red):
        return 0
    return int(nx.node_connectivity(red))


def _metricas_red(nombre: str, red: nx.Graph, tipo: str) -> dict[str, object]:
    nodos = red.number_of_nodes()
    aristas = red.number_of_edges()
    grados = [grado for _, grado in red.degree()]
    componentes = list(nx.connected_components(red)) if nodos else []
    mayor = max((len(componente) for componente in componentes), default=0)
    componente_mayor = max(componentes, key=len, default=set())
    subred_mayor = red.subgraph(componente_mayor).copy()
    return {
        "red": nombre,
        "tipo_de_red": tipo,
        "nodos": nodos,
        "aristas": aristas,
        "densidad": nx.density(red),
        "grado_medio": (2 * aristas / nodos) if nodos else 0.0,
        "grado_medio_ponderado": (sum(dict(red.degree(weight="weight")).values()) / nodos) if nodos else 0.0,
        "grado_maximo": max(grados, default=0),
        "componentes": len(componentes),
        "tamano_componente_mayor": mayor,
        "proporcion_componente_mayor": (mayor / nodos) if nodos else 0.0,
        "aislados": nx.number_of_isolates(red),
        "perifericos_grado_1": sum(grado == 1 for grado in grados),
        "cohesion_conectividad_nodal": _cohesion(red),
        "cohesion_componente_mayor": _cohesion(subred_mayor),
        "transitividad": nx.transitivity(red),
    }


def _tabla_distribucion(nombre: str, red: nx.Graph) -> pd.DataFrame:
    grados = pd.Series(dict(red.degree()), name="grado")
    ponderados = pd.Series(dict(red.degree(weight="weight")), name="grado_ponderado")
    tabla = (
        pd.DataFrame({"grado": grados, "grado_ponderado": ponderados})
        .value_counts("grado")
        .rename("nodos")
        .reset_index()
        .sort_values("grado")
    )
    tabla.insert(0, "red", nombre)
    tabla["proporcion_nodos"] = tabla["nodos"] / red.number_of_nodes() if red.number_of_nodes() else 0.0
    tabla["grado_ponderado_medio"] = [
        float(ponderados[grados == grado].mean()) for grado in tabla["grado"]
    ]
    return tabla


def _tablas_estructura(nombre: str, red: nx.Graph) -> tuple[pd.DataFrame, pd.DataFrame]:
    componentes_ordenadas = sorted(nx.connected_components(red), key=lambda x: (-len(x), sorted(x)[0]))
    componente_por_nodo = {
        nodo: indice + 1
        for indice, componente in enumerate(componentes_ordenadas)
        for nodo in componente
    }
    componentes = pd.DataFrame(
        [
            {
                "red": nombre,
                "componente": indice + 1,
                "tamano": len(componente),
                "aristas_internas": red.subgraph(componente).number_of_edges(),
                "es_componente_mayor": indice == 0,
            }
            for indice, componente in enumerate(componentes_ordenadas)
        ]
    )
    nodos = pd.DataFrame(
        [
            {
                "red": nombre,
                "node_id": nodo,
                "node_type": datos.get("node_type"),
                "label": datos.get("label", nodo),
                "grado": red.degree(nodo),
                "grado_ponderado": _peso_entero(red.degree(nodo, weight="weight")),
                "componente": componente_por_nodo[nodo],
                "estado_estructural": (
                    "aislado" if red.degree(nodo) == 0 else "periferico_grado_1"
                    if red.degree(nodo) == 1 else "conectado"
                ),
            }
            for nodo, datos in sorted(red.nodes(data=True))
        ]
    )
    return componentes, nodos


def analizar_proyecciones(resultado_bipartita: ResultadoRed) -> ResultadoProyecciones:
    """Deriva proyecciones y todas las métricas requeridas para ejercicios 5--6."""

    autores, videos = construir_proyecciones(resultado_bipartita.red)
    redes = [
        ("bipartita_autor_video", resultado_bipartita.red, "bipartita_no_dirigida"),
        ("proyeccion_autor_autor", autores, "proyeccion_no_dirigida"),
        ("proyeccion_video_video", videos, "proyeccion_no_dirigida"),
    ]
    metricas = pd.DataFrame([_metricas_red(nombre, red, tipo) for nombre, red, tipo in redes])
    distribuciones = pd.concat([_tabla_distribucion(nombre, red) for nombre, red, _ in redes], ignore_index=True)
    productos = [_tablas_estructura(nombre, red) for nombre, red, _ in redes]
    componentes = pd.concat([producto[0] for producto in productos], ignore_index=True)
    nodos = pd.concat([producto[1] for producto in productos], ignore_index=True)

    pares_autor = sum(1 for _ in autores.edges)
    pares_video = sum(1 for _ in videos.edges)
    validacion = pd.DataFrame([
        {"indicador": "autores_conservados", "valor": autores.number_of_nodes() == sum(d["node_type"] == "author" for _, d in resultado_bipartita.red.nodes(data=True))},
        {"indicador": "videos_conservados", "valor": videos.number_of_nodes() == sum(d["node_type"] == "video" for _, d in resultado_bipartita.red.nodes(data=True))},
        {"indicador": "aristas_autor_autor", "valor": pares_autor},
        {"indicador": "aristas_video_video", "valor": pares_video},
        {"indicador": "pesos_autor_autor_positivos", "valor": all(d["weight"] >= 1 for _, _, d in autores.edges(data=True))},
        {"indicador": "pesos_video_video_positivos", "valor": all(d["weight"] >= 1 for _, _, d in videos.edges(data=True))},
    ])
    return ResultadoProyecciones(autores, videos, metricas, distribuciones, componentes, nodos, validacion)


def visualizar_proyeccion(red: nx.Graph, ruta_salida: Path | None = None, semilla: int = 42) -> tuple[plt.Figure, plt.Axes]:
    """Dibuja la proyección completa y separa visualmente los aislados."""

    conectados = [nodo for nodo, grado in red.degree() if grado > 0]
    aislados = sorted(set(red.nodes).difference(conectados))
    subred = red.subgraph(conectados)
    posiciones = nx.spring_layout(subred, seed=semilla, weight="weight", iterations=300) if conectados else {}
    pesos = np.array([datos["weight"] for _, _, datos in subred.edges(data=True)], dtype=float)
    anchos = 0.25 + 0.55 * np.log1p(pesos) if len(pesos) else []
    color = "#2878B5" if red.graph.get("node_type") == "author" else "#E76F51"
    etiqueta = "Autores" if red.graph.get("node_type") == "author" else "Videos"

    figura, (eje, eje_aislados) = plt.subplots(1, 2, figsize=(18, 11), gridspec_kw={"width_ratios": [4.5, 1]})
    nx.draw_networkx_edges(subred, posiciones, ax=eje, edge_color="#667085", width=anchos, alpha=0.24)
    nx.draw_networkx_nodes(subred, posiciones, ax=eje, node_color=color, node_size=25, linewidths=0, alpha=0.85)
    eje.set_title(f"{etiqueta} con co-participación observada", fontsize=13, pad=12)
    eje.legend(handles=[Line2D([0], [0], marker="o", color="none", markerfacecolor=color, markersize=8, label=etiqueta)], loc="upper right", frameon=False)
    eje.set_axis_off()
    eje_aislados.set_title(f"Aislados observados\n({len(aislados):,})", fontsize=13, pad=12)
    if aislados:
        columnas = 14
        indices = np.arange(len(aislados))
        eje_aislados.scatter(indices % columnas, -(indices // columnas), s=30, color=color, edgecolors="white", linewidths=0.25, alpha=0.9)
        eje_aislados.set_aspect("equal")
    eje_aislados.set_axis_off()
    figura.suptitle(f"{red.graph.get('name', 'Proyección')} completa\n{red.number_of_nodes():,} nodos · {red.number_of_edges():,} aristas", fontsize=17, y=0.98)
    figura.text(0.5, 0.015, "Un aislado refleja ausencia de co-participación recuperada en la muestra, no ausencia de actividad real.", ha="center", va="bottom", fontsize=10, color="#475467")
    figura.tight_layout(rect=(0, 0.04, 1, 0.94))
    if ruta_salida is not None:
        Path(ruta_salida).parent.mkdir(parents=True, exist_ok=True)
        figura.savefig(ruta_salida, dpi=220, bbox_inches="tight", facecolor="white")
    return figura, eje


def visualizar_distribuciones(distribuciones: pd.DataFrame, ruta_salida: Path | None = None) -> tuple[plt.Figure, np.ndarray]:
    """Grafica la frecuencia de grado de las tres redes sin ocultar el grado cero."""

    nombres = ["bipartita_autor_video", "proyeccion_autor_autor", "proyeccion_video_video"]
    titulos = ["Bipartita autor–video", "Proyección autor–autor", "Proyección video–video"]
    figura, ejes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
    for eje, nombre, titulo in zip(ejes, nombres, titulos):
        tabla = distribuciones[distribuciones["red"] == nombre]
        eje.bar(tabla["grado"], tabla["nodos"], color="#4C78A8")
        eje.set(title=titulo, xlabel="Grado", ylabel="Número de nodos")
        eje.grid(axis="y", alpha=0.2)
    figura.suptitle("Distribuciones de grado (incluyen aislados)", fontsize=15, y=1.02)
    figura.tight_layout()
    if ruta_salida is not None:
        Path(ruta_salida).parent.mkdir(parents=True, exist_ok=True)
        figura.savefig(ruta_salida, dpi=220, bbox_inches="tight", facecolor="white")
    return figura, ejes


def _texto_resultados(resultado: ResultadoProyecciones) -> str:
    metricas = resultado.metricas.set_index("red")
    a = metricas.loc["proyeccion_autor_autor"]
    v = metricas.loc["proyeccion_video_video"]
    b = metricas.loc["bipartita_autor_video"]
    return f"""## Proyecciones y topología (ejercicios 5–6)

### Definiciones reproducibles

- **Proyección autor–autor:** dos autores se unen cuando comentaron en al menos un mismo video; `weight` es el número de videos distintos compartidos. Por ello, varios comentarios del mismo autor en el mismo video no aumentan el peso.
- **Proyección video–video:** dos videos se unen cuando recibieron comentarios de al menos un mismo autor; `weight` es el número de autores distintos compartidos.
- Se conservaron todos los nodos de cada partición: los videos sin comentarios recuperados permanecen como aislados en su proyección. Las redes son no dirigidas y los pesos se usan para el diseño de las figuras; las métricas de grado, densidad y transitividad se reportan sobre la estructura no ponderada para conservar interpretabilidad.
- **Cohesión** se operacionaliza como conectividad nodal global: el mínimo de nodos que habría que remover para desconectar una red ya conexa. Como las tres redes están fragmentadas, la cohesión global es 0; no significa que cada componente carezca de conexiones internas.

### Resultados estructurales

La bipartita contiene {int(b.nodos):,} nodos y {int(b.aristas):,} aristas, repartidos en {int(b.componentes):,} componentes; su componente mayor reúne {int(b.tamano_componente_mayor):,} nodos ({b.proporcion_componente_mayor:.1%}) y hay {int(b.aislados):,} aislados. La proyección de autores tiene {int(a.nodos):,} autores, {int(a.aristas):,} enlaces y {int(a.componentes):,} componentes; la mayor reúne {int(a.tamano_componente_mayor):,} autores ({a.proporcion_componente_mayor:.1%}). La proyección de videos tiene {int(v.nodos):,} videos, {int(v.aristas):,} enlaces y {int(v.componentes):,} componentes; la mayor reúne {int(v.tamano_componente_mayor):,} videos ({v.proporcion_componente_mayor:.1%}) y {int(v.aislados):,} videos quedan aislados en la muestra.

La densidad es baja en las tres redes (bipartita {b.densidad:.4f}, autores {a.densidad:.4f}, videos {v.densidad:.4f}), lo que junto con el predominio de grados bajos y los numerosos componentes describe participación fragmentada y concentrada en pocos espacios de co-participación. La transitividad es {a.transitividad:.4f} en autores y {v.transitividad:.4f} en videos: los triángulos de las proyecciones capturan grupos que coinciden en varios contenidos o autores, pero no prueban amistad, conversación directa ni afinidad. Aunque la cohesión global es 0 por fragmentación, la conectividad nodal de la componente mayor es {int(b.cohesion_componente_mayor)} (bipartita), {int(a.cohesion_componente_mayor)} (autores) y {int(v.cohesion_componente_mayor)} (videos).

### Interpretación y límites

La proyección autor–autor representa **audiencias que coincidieron en comentar**; la de video–video representa **contenidos con comentaristas compartidos**. Los enlaces muestran co-participación observada, no una relación social explícita ni exposición completa de las audiencias. En especial, un aislado solo indica que no hubo co-participación o comentarios recuperados bajo esta recolección; no permite concluir inactividad del video, su canal o sus espectadores. Los hallazgos describen los 406 comentarios y 293 videos disponibles, condicionados por las consultas, canales y momento de recolección.
"""


def exportar_proyecciones(resultado: ResultadoProyecciones, directorio: Path) -> dict[str, Path]:
    """Exporta redes, tablas, figuras y nota metodológica."""

    directorio = Path(directorio)
    tablas, figuras = directorio / "tablas", directorio / "figuras"
    tablas.mkdir(parents=True, exist_ok=True)
    figuras.mkdir(parents=True, exist_ok=True)
    rutas = {
        "metricas": tablas / "metricas_redes.csv",
        "grados": tablas / "distribucion_grados_redes.csv",
        "componentes": tablas / "componentes_redes.csv",
        "nodos_estructura": tablas / "nodos_estructura_redes.csv",
        "validacion": tablas / "validacion_proyecciones.csv",
        "red_autores": directorio / "proyeccion_autor_autor.gexf",
        "red_videos": directorio / "proyeccion_video_video.gexf",
        "figura_autores": figuras / "proyeccion_autor_autor_completa.png",
        "figura_videos": figuras / "proyeccion_video_video_completa.png",
        "figura_grados": figuras / "distribucion_grados_redes.png",
        "metodologia": directorio / "metodologia_proyecciones_topologia.md",
    }
    resultado.metricas.to_csv(rutas["metricas"], index=False, encoding="utf-8-sig")
    resultado.distribucion_grados.to_csv(rutas["grados"], index=False, encoding="utf-8-sig")
    resultado.componentes.to_csv(rutas["componentes"], index=False, encoding="utf-8-sig")
    resultado.nodos_estructura.to_csv(rutas["nodos_estructura"], index=False, encoding="utf-8-sig")
    resultado.validacion.to_csv(rutas["validacion"], index=False, encoding="utf-8-sig")
    nx.write_gexf(resultado.autores, rutas["red_autores"])
    nx.write_gexf(resultado.videos, rutas["red_videos"])
    figura, _ = visualizar_proyeccion(resultado.autores, rutas["figura_autores"])
    plt.close(figura)
    figura, _ = visualizar_proyeccion(resultado.videos, rutas["figura_videos"])
    plt.close(figura)
    figura, _ = visualizar_distribuciones(resultado.distribucion_grados, rutas["figura_grados"])
    plt.close(figura)
    rutas["metodologia"].write_text(_texto_resultados(resultado), encoding="utf-8")
    return rutas


def ejecutar(raiz: Path) -> ResultadoProyecciones:
    """Regenera las proyecciones y métricas desde los CSV originales."""

    raiz = Path(raiz).resolve()
    comentarios = pd.read_csv(raiz / "data" / "youtube_comments.csv", encoding="utf-8-sig")
    videos = pd.read_csv(raiz / "data" / "youtube_videos.csv", encoding="utf-8-sig")
    bipartita = construir_red_bipartita(comentarios, videos)
    resultado = analizar_proyecciones(bipartita)
    exportar_proyecciones(resultado, raiz / "resultados")
    return resultado


def main() -> None:
    analizador = argparse.ArgumentParser(description="Genera proyecciones y métricas de topología.")
    analizador.add_argument("--raiz", type=Path, default=Path(__file__).resolve().parents[1])
    argumentos = analizador.parse_args()
    resultado = ejecutar(argumentos.raiz)
    print(resultado.metricas.to_string(index=False))


if __name__ == "__main__":
    main()
