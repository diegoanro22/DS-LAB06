## Proyecciones y topología (ejercicios 5–6)

### Definiciones reproducibles

- **Proyección autor–autor:** dos autores se unen cuando comentaron en al menos un mismo video; `weight` es el número de videos distintos compartidos. Por ello, varios comentarios del mismo autor en el mismo video no aumentan el peso.
- **Proyección video–video:** dos videos se unen cuando recibieron comentarios de al menos un mismo autor; `weight` es el número de autores distintos compartidos.
- Se conservaron todos los nodos de cada partición: los videos sin comentarios recuperados permanecen como aislados en su proyección. Las redes son no dirigidas y los pesos se usan para el diseño de las figuras; las métricas de grado, densidad y transitividad se reportan sobre la estructura no ponderada para conservar interpretabilidad.
- **Cohesión** se operacionaliza como conectividad nodal global: el mínimo de nodos que habría que remover para desconectar una red ya conexa. Como las tres redes están fragmentadas, la cohesión global es 0; no significa que cada componente carezca de conexiones internas.

### Resultados estructurales

La bipartita contiene 625 nodos y 343 aristas, repartidos en 284 componentes; su componente mayor reúne 286 nodos (45.8%) y hay 274 aislados. La proyección de autores tiene 332 autores, 10,732 enlaces y 10 componentes; la mayor reúne 276 autores (83.1%). La proyección de videos tiene 293 videos, 11 enlaces y 284 componentes; la mayor reúne 10 videos (3.4%) y 283 videos quedan aislados en la muestra.

La densidad es baja en las tres redes (bipartita 0.0018, autores 0.1953, videos 0.0003), lo que junto con el predominio de grados bajos y los numerosos componentes describe participación fragmentada y concentrada en pocos espacios de co-participación. La transitividad es 0.9840 en autores y 0.3158 en videos: los triángulos de las proyecciones capturan grupos que coinciden en varios contenidos o autores, pero no prueban amistad, conversación directa ni afinidad. Aunque la cohesión global es 0 por fragmentación, la conectividad nodal de la componente mayor es 1 (bipartita), 1 (autores) y 1 (videos).

### Interpretación y límites

La proyección autor–autor representa **audiencias que coincidieron en comentar**; la de video–video representa **contenidos con comentaristas compartidos**. Los enlaces muestran co-participación observada, no una relación social explícita ni exposición completa de las audiencias. En especial, un aislado solo indica que no hubo co-participación o comentarios recuperados bajo esta recolección; no permite concluir inactividad del video, su canal o sus espectadores. Los hallazgos describen los 406 comentarios y 293 videos disponibles, condicionados por las consultas, canales y momento de recolección.
