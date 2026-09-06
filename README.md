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
python -m src.comunidades_centralidad
python -m src.sentimiento_contenido
```

El flujo completo carga los CSV originales desde `data/`, valida los identificadores, agrupa cada pareja autor-video y regenera:

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

## Entrega final — Comunidades, centralidad y participantes puente (ejercicios 7–8)

La tercera orden regenera la detección modular de comunidades y el análisis
posicional de centralidad sobre las proyecciones y la bipartita. Produce:

- `resultados/tablas/comunidades_videos.csv`, resumiendo las comunidades
  temáticas, densidad interna, canales representados y tono discursivo;
- `resultados/tablas/centralidad_videos.csv` y
  `resultados/tablas/centralidad_autores.csv`, con grado no ponderado, grado
  ponderado, intermediación (*betweenness*), cercanía, PageRank y estado de corte;
- `resultados/tablas/ranking_puentes_articuladores.csv`, que sistematiza los 5
  videos bisagra y los 9 autores multividales que conectan las esferas temáticas;
- `resultados/tablas/validacion_comunidades.csv`, con chequeos automatizados de
  integridad;
- `resultados/figuras/comunidades_videos_completa.png` (núcleo y matriz de
  aislados) y `resultados/figuras/comunidades_videos_detalle.png` (anotación de
  canales, enlaces y autores puente);
- `resultados/metodologia_comunidades_centralidad.md`, listo para el informe.

Se utiliza la proyección video–video como red base para Louvain (semilla 42,
modularidad $Q = 0.4053$), identificando tres comunidades conexas sustantivas:
fiscalización política (C1), comunicación institucional/seguridad (C2) y
servicios cívicos/monopolios (C3). Exactamente 5 videos y 7 autores constituyen
puntos de articulación cuya remoción desconecta la circulación discursiva.

## Entrega final — Contenido, sentimiento, limitaciones y conclusiones (ejercicios 9–10)

La cuarta orden puntúa el sentimiento de cada comentario, lo resume por video,
canal, categoría y comunidad, y conecta el contenido con la estructura de la
red. Produce:

- `resultados/tablas/sentimiento_comentarios.csv`, el dataset con el puntaje de
  cada comentario y el detalle auditable (términos positivos y negativos
  detectados, emojis valorados y negaciones aplicadas);
- `resultados/tablas/sentimiento_resumen_general.csv`,
  `sentimiento_por_video.csv`, `sentimiento_por_canal.csv`,
  `sentimiento_por_categoria.csv` y `sentimiento_por_comunidad.csv`;
- `resultados/tablas/contenido_por_comunidad.csv`, con palabras y bigramas
  frecuentes junto al tono de cada comunidad;
- `resultados/tablas/afirmaciones_lenguaje_cauteloso.csv`, que separa
  descripción, asociación e inferencias que no pueden sostenerse;
- `resultados/tablas/validacion_sentimiento.csv`, con los chequeos automáticos;
- `resultados/figuras/sentimiento_general.png` y
  `resultados/figuras/sentimiento_comunidades.png`;
- `resultados/metodologia_sentimiento_contenido.md`, con la ficha del léxico,
  las limitaciones y las conclusiones integradas.

El sentimiento se calcula con un léxico afectivo en español implementado en
`src/sentimiento_contenido.py` (versión 1.0), sin dependencias externas ni
descargas: cada término, cada emoji y cada regla de negación, intensificación y
énfasis quedan auditables en el repositorio. La escala es `[-3, +3]` por término
y `[-1, +1]` por comentario; las categorías son positivo, negativo, neutral,
sin cobertura léxica y sin texto. Los comentarios sin ningún término reconocido
se reportan aparte y no se cuentan como neutrales, y las comparaciones con menos
de 10 comentarios se marcan como no comparables.

## Definición e interpretación

Un nodo de tipo `author` representa un `author_channel_id` y un nodo de tipo `video` representa un `video_id`. Existe una arista cuando el autor publicó al menos un comentario en el video; `weight` es el número de comentarios observados de esa pareja. La relación no indica amistad, conversación directa, aprobación ni exposición de toda la audiencia.

La red conserva todos los videos del archivo recolectado. Los videos aislados no tuvieron comentarios recuperados en esta muestra, lo cual no demuestra que carezcan de actividad en YouTube. Los nombres y handles se usan solo como etiquetas; los identificadores son las llaves estables.
