## Comunidades, centralidad y participantes puente (ejercicios 7–8)

### Justificación metodológica de la red seleccionada

Para la detección de comunidades se seleccionó de manera prioritaria la **proyección video–video**. Esta elección se fundamenta en que sus aristas modelan directamente **audiencias compartidas** (dos videos se enlazan si comparten al menos un autor comentarista común). Esta representación permite agrupar los contenidos en función de las trayectorias reales de atención y deliberación del público, habilitando una interpretación directa de afinidades editoriales, proximidad discursiva y solapamiento entre canales.

De forma complementaria, la **proyección autor–autor** y la **red bipartita** se utilizan para cuantificar el rol de intermediación de los comentaristas y verificar la existencia de puntos de corte nodales.

### Detección de comunidades y modularidad (Louvain)

Aplicando el algoritmo Louvain ponderado con semilla fija (`seed=42`) sobre la proyección de videos, se identifica una estructura de comunidades nítida con modularidad **Q = 0.4053**, lo que refleja una partición significativamente superior al azar.

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
  El video con mayor grado e intermediación es «Qué rico come tu diputado» (Canal: Quorum), con grado ponderado de 5, intermediación de 0.6389 en la componente conexa y PageRank de 0.1801. Junto a «Internet: escoger el menos malo» (intermediación 0.3889) y «Conferencia de Prensa del Gobierno» (intermediación 0.3889), constituyen los tres ejes neurálgicos de la circulación discursiva.
- **Puntos de articulación de corte en videos:**
  Se detectaron exactamente **5 videos que son puntos de articulación**: «Qué rico come tu diputado», «Internet: escoger el menos malo», «Conferencia de Prensa del Gobierno de Guatemala. #LaRondaGt», «La cooptación de Walter Mazariegos en la USAC» y «Inician los trabajos de recuperación del Puente Belice II.». La remoción de cualquiera de ellos fragmenta la componente conexa en componentes disjuntas. Por ejemplo, remover «Qué rico come tu diputado» aísla la cobertura de PrensaLibreOficial y segrega la rama institucional del Gobierno del debate crítico.

- **Autores centrales y puentes:**
  Entre los 332 autores de la muestra, únicamente **9 autores publicaron comentarios en más de un video**. De ellos, **7 autores actúan como puntos de articulación estrictos** en la componente gigante de la proyección autor–autor:
  1. `@virgiliogarcia3039` (Intermediación = 0.2390): es el puente inter-institucional cardinal de toda la red, al haber comentado simultáneamente en la Conferencia de Prensa del Gobierno y en «Qué rico come tu diputado» de Quorum.
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
