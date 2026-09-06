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
python -m src.proyecciones_topologia
```

El script carga los CSV originales desde `data/`, valida los identificadores, agrupa cada pareja autor-video y regenera:

- `resultados/tablas/nodos_red_bipartita.csv`
- `resultados/tablas/aristas_red_bipartita.csv`
- `resultados/tablas/validacion_red_bipartita.csv`
- `resultados/red_bipartita.gexf`
- `resultados/figuras/red_bipartita_completa.png`

El notebook `notebooks/Laboratorio_6_Analisis_Redes_YouTube.ipynb` integra la misma función después de las secciones de carga, limpieza y exploración.

## Entrega final — Proyecciones y topología (ejercicios 5–6)

La segunda orden regenera el análisis de proyecciones y topología desde los
CSV originales y la misma definición de red bipartita. Produce:

- `resultados/proyeccion_autor_autor.gexf` y `resultados/proyeccion_video_video.gexf`;
- `resultados/tablas/metricas_redes.csv`, con nodos, aristas, densidad, grado
  medio, componentes, aislados, periféricos, cohesión y transitividad;
- `resultados/tablas/distribucion_grados_redes.csv`,
  `componentes_redes.csv`, `nodos_estructura_redes.csv` y
  `validacion_proyecciones.csv`;
- las figuras completas de ambas proyecciones y de sus distribuciones de
  grado en `resultados/figuras/`;
- `resultados/metodologia_proyecciones_topologia.md`, listo para incorporar
  al informe.

La proyección autor–autor conecta dos autores que comentaron en un mismo
video; su peso es el número de videos distintos compartidos. La proyección
video–video conecta dos videos con al menos un autor comentarista en común;
su peso es el número de autores distintos compartidos. Las figuras y tablas
conservan nodos aislados. Ese aislamiento es solamente el de la participación
recuperada en esta muestra, no evidencia de ausencia de actividad real.

## Definición e interpretación

Un nodo de tipo `author` representa un `author_channel_id` y un nodo de tipo `video` representa un `video_id`. Existe una arista cuando el autor publicó al menos un comentario en el video; `weight` es el número de comentarios observados de esa pareja. La relación no indica amistad, conversación directa, aprobación ni exposición de toda la audiencia.

La red conserva todos los videos del archivo recolectado. Los videos aislados no tuvieron comentarios recuperados en esta muestra, lo cual no demuestra que carezcan de actividad en YouTube. Los nombres y handles se usan solo como etiquetas; los identificadores son las llaves estables.
