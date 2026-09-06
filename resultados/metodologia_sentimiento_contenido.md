# Contenido, sentimiento, limitaciones y conclusiones

## 1. Recurso léxico y decisiones de medición

| Elemento | Decisión |
|---|---|
| Herramienta | Léxico afectivo en español para el corpus (elaboración propia), versión 1.0 |
| Implementación | `src/sentimiento_contenido.py`, sin dependencias externas ni descargas |
| Escala por término | [-3, +3] por término |
| Escala por comentario | [-1, +1] tras normalización |
| Normalización | `puntaje / sqrt(puntaje^2 + 15)` |
| Categorías | positivo (>= 0.05), negativo (<= -0.05), neutral, sin cobertura léxica, sin texto |
| Negación | `no, ni, nunca, jamás, tampoco, nada, nadie, ningún, sin` invierten el signo (factor -0.74) en una ventana de 3 palabras |
| Intensificadores y atenuadores | `muy`, `demasiado`, `súper`, `bastante` amplifican; `poco`, `apenas`, `casi` reducen, con efecto decreciente según la distancia |
| Mayúsculas | una palabra evaluativa en mayúsculas sostenidas amplifica su valencia (x1.3) |
| Exclamaciones | hasta 3 signos suman 0.29 cada uno, conservando el signo del puntaje |
| Emojis | se puntúan en su forma Unicode y en la forma de código corto (`:hand-purple-blue-peace:`) que trae el archivo recolectado; banderas y emojis sin carga evaluativa valen cero |
| Textos vacíos | se conservan en la tabla con la etiqueta `sin texto` y no se cuentan como neutrales genuinos |
| Base textual | se puntúa `texto_original`; `texto_limpio` solo se usa para palabras y bigramas |

Se optó por un léxico propio y explícito, y no por un modelo preentrenado, por
tres razones reproducibles: el análisis debe correr sin descargas ni claves de
API, cada término y cada regla quedan auditables en el repositorio, y el
vocabulario se ajustó al registro observado en el corpus (denuncia política,
crónica policial y consumo). El costo de esa decisión es cobertura: un modelo
contextual capta ironía y giros que un léxico no reconoce.

**Sesgos conocidos del recurso:** no detecta ironía ni sarcasmo, frecuentes en
la crítica política; penaliza el vocabulario de denuncia aunque describa un
hecho y no una emoción; no cubre variantes dialectales guatemaltecas ausentes
del léxico; no lematiza, por lo que formas verbales no listadas quedan sin
puntuar; y trata cada comentario de forma aislada, sin el hilo ni el video al
que responde.

## 2. Sentimiento observado en el corpus

De los 406 comentarios evaluados, 133 resultan negativos (32.8%), 135 positivos (33.3%) y 1 neutrales (0.2%). Otros 137 comentarios (33.7%) no contienen ningún término ni emoji del léxico: se reportan como **sin cobertura léxica** y no se interpretan como neutralidad.

El sentimiento medio del corpus es -0.0094 y la mediana +0.0000. La cobertura léxica media es 66.3%; 56 comentarios aportan al menos un emoji con valencia y 22 activan la regla de negación.

### Videos con muestra suficiente (n >= 10)

| Video | Comentarios | Sentimiento medio | % negativos | % positivos |
|---|---|---|---|---|
| Qué rico come tu diputado | 161 | -0.1999 | 45.3% | 16.1% |
| La cooptación de Walter Mazariegos en la USAC | 50 | -0.0677 | 40.0% | 30.0% |
| EE.UU. envía a mexicanos deportados a Guatemala antes de su  | 25 | +0.0499 | 24.0% | 32.0% |
| Conferencia de Prensa del Gobierno de Guatemala. #LaRondaGt | 25 | +0.0931 | 20.0% | 36.0% |
| Internet: escoger el menos malo | 12 | +0.1129 | 33.3% | 50.0% |
| Inician los trabajos de recuperación del Puente Belice II. | 45 | +0.1348 | 26.7% | 55.6% |
| Capturan a presuntos delincuentes disfrazados de mujer señal | 14 | +0.1799 | 7.1% | 42.9% |
| Arroz con pollo a la MONOPOLIO | 16 | +0.2576 | 12.5% | 43.8% |
| Plan 2032 Ciudad de Guatemala | 25 | +0.4838 | 0.0% | 84.0% |

Los videos con menos de 10 comentarios permanecen en
`resultados/tablas/sentimiento_por_video.csv` marcados como no comparables.

### Canales con muestra suficiente

| Canal | Comentarios | Sentimiento medio | % negativos |
|---|---|---|---|
| Quorum | 256 | -0.1009 | 40.2% |
| Noticias Telemundo | 25 | +0.0499 | 24.0% |
| Gobierno de la República de Guatemala | 70 | +0.1199 | 24.3% |
| Noti7 | 14 | +0.1799 | 7.1% |
| Municipalidad de Guatemala | 25 | +0.4838 | 0.0% |

### Categorías

| Categoría | Comentarios | Sentimiento medio | Estado |
|---|---|---|---|
| News & Politics | 335 | -0.0389 | comparable |
| Entertainment | 70 | +0.1199 | comparable |
| Nonprofits & Activism | 1 | +0.8176 | muestra insuficiente |

Solo 2 categorías superan el umbral de muestra, de
modo que la diferencia entre categorías se describe pero no se usa como
evidencia de un patrón general.

## 3. Contenido y red: qué se dice en cada comunidad

- **C1 — Fiscalización política y coyuntura** (225 comentarios de 187 autores en 4 videos): sentimiento medio -0.1669, con 43.6% de comentarios negativos y 19.1% positivos. Palabras frecuentes: pueblo (54), dinero (28), diputados (28), estos (28), trabajo (24), diputado (22). Bigramas: nery rodas (5), estos diputados (5), pacto corruptos (5).
- **C2 — Esfera institucional y seguridad pública** (84 comentarios de 62 autores en 3 videos): sentimiento medio +0.1299, con 21.4% de comentarios negativos y 47.6% positivos. Palabras frecuentes: presidente (27), guatemala (16), país (12), años (11), arevalo (10), ni (9). Bigramas: presidente bernardo (9), bernardo arevalo (8), bla bla (4).
- **C3 — Servicios cívicos, movilidad y consumo** (34 comentarios de 29 autores en 3 videos): sentimiento medio +0.2170, con 20.6% de comentarios negativos y 50.0% positivos. Palabras frecuentes: más (12), excelente (10), empresas (8), solo (7), internet (7), ley (7). Bigramas: hand purple (3), purple blue (3), blue peace (3).

Los 63 comentarios publicados en videos sin co-comentación observada alcanzan un sentimiento medio de +0.2452. Se reportan aparte porque no pertenecen a ninguna comunidad de la red: su aislamiento proviene de la cobertura de la recolección.

La lectura conjunta es de asociación, no de causa: los videos de una misma
comunidad comparten al menos un comentarista y, además, comparten registro
temático. Nada en estos datos indica que la comunidad provoque el tono, ni que
los autores puente lo trasladen de un video a otro.

## 4. Limitaciones

1. **Cobertura parcial de comentarios.** Se recuperaron 406 comentarios
   principales para 19 videos de los 293 recolectados. No
   es el universo de comentarios de esos videos, ni incluye respuestas
   anidadas.
2. **Muestreo por consultas y canales.** Los videos provienen de búsquedas y
   canales específicos; la muestra refleja ese diseño de recolección y no la
   conversación de YouTube en Guatemala.
3. **Fechas relativas.** Las publicaciones se registran como texto relativo
   («hace 2 meses»), lo que impide fechar con precisión y ordenar series
   temporales.
4. **Conteos de un instante.** Vistas, likes y respuestas corresponden al
   momento de la descarga y siguen cambiando en la plataforma.
5. **Ausencia de respuestas explícitas entre autores.** `reply_count` cuenta
   respuestas recibidas por un comentario, pero no identifica quién responde a
   quién; la red no contiene interacción directa entre usuarios.
6. **Concentración en pocos videos.** Una porción mayoritaria de los
   comentarios proviene de un puñado de videos, y solo un grupo reducido de
   autores comenta en más de uno. Las métricas de red y de sentimiento dependen
   fuertemente de ese núcleo.
7. **Límites del léxico.** Ironía, sarcasmo, jerga local y negaciones complejas
   quedan fuera del alcance del recurso; el 33.7% de
   comentarios sin cobertura léxica documenta ese margen.
8. **Atribución indirecta del tema.** El sentimiento de una comunidad se
   construye con los comentarios de sus videos, no con textos escritos «dentro»
   de una comunidad: la comunidad es una propiedad de la red, no del texto.

## 5. Conclusiones integradas

1. **Participación (descripción).** La participación observada está muy
   concentrada: pocos videos reúnen la mayoría de los comentarios y la mayoría
   de los autores aparece una sola vez. Esto describe la muestra recolectada.
2. **Red (descripción).** La red bipartita autor–video es casi enteramente
   fragmentada: la conexión entre videos depende de un grupo pequeño de
   comentaristas que participaron en más de un video, y varios de ellos son
   puntos de articulación cuya remoción desconecta la componente.
3. **Comunidades (asociación).** Las comunidades detectadas en la proyección
   video–video agrupan contenidos con registro temático afín. La coincidencia
   entre estructura y tema es una asociación observada, no un mecanismo
   demostrado.
4. **Contenido (descripción).** El vocabulario dominante del corpus gira en
   torno a la fiscalización del gasto público, la crítica institucional y el
   consumo cotidiano, según las palabras y bigramas más frecuentes de cada
   comunidad.
5. **Sentimiento (descripción con cautela).** El tono no es uniforme y el promedio del corpus (-0.0094) no describe bien a ningún grupo: el registro negativo se concentra en la comunidad «Fiscalización política y coyuntura» (-0.1669) y en el video «Qué rico come tu diputado» (-0.1999), mientras que «Servicios cívicos, movilidad y consumo» resulta claramente más positiva (+0.2170). La medición captura el léxico escrito y no debe leerse como el estado emocional de las personas ni como su posición política.
6. **Alcance (inferencia acotada).** Ninguna de estas conclusiones se extiende
   a todos los usuarios de YouTube ni a la opinión pública guatemalteca. Son
   válidas para el conjunto recolectado, en el momento en que se descargó, y
   bajo las definiciones de arista y de peso declaradas en el informe.

La tabla `resultados/tablas/afirmaciones_lenguaje_cauteloso.csv` lista las
afirmaciones que requieren lenguaje cauteloso y las que no deben sostenerse con
estos datos.
