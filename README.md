# Laboratorio 6 — Red bipartita de YouTube

Este repositorio contiene un análisis reproducible de los datos recolectados de YouTube. La sección de red construye una red bipartita no dirigida entre autores y videos.

## Requisitos

- Python 3.11 o posterior
- Dependencias de `requirements.txt`

Instalación:

```bash
python -m pip install -r requirements.txt
```

## Ejecución

Desde la raíz del repositorio:

```bash
python -m src.red_bipartita
```

El script carga los CSV originales desde `data/`, valida los identificadores, agrupa cada pareja autor-video y regenera:

- `resultados/tablas/nodos_red_bipartita.csv`
- `resultados/tablas/aristas_red_bipartita.csv`
- `resultados/tablas/validacion_red_bipartita.csv`
- `resultados/red_bipartita.gexf`
- `resultados/figuras/red_bipartita_completa.png`

El notebook `notebooks/Laboratorio_6_Analisis_Redes_YouTube.ipynb` integra la misma función después de las secciones de carga, limpieza y exploración.

## Definición e interpretación

Un nodo de tipo `author` representa un `author_channel_id` y un nodo de tipo `video` representa un `video_id`. Existe una arista cuando el autor publicó al menos un comentario en el video; `weight` es el número de comentarios observados de esa pareja. La relación no indica amistad, conversación directa, aprobación ni exposición de toda la audiencia.

La red conserva todos los videos del archivo recolectado. Los videos aislados no tuvieron comentarios recuperados en esta muestra, lo cual no demuestra que carezcan de actividad en YouTube. Los nombres y handles se usan solo como etiquetas; los identificadores son las llaves estables.
