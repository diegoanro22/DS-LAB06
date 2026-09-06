"""Análisis de contenido, sentimiento y conclusiones integradas.

Este módulo puntúa el sentimiento de cada comentario con un léxico afectivo en
español documentado en el propio código, resume la polaridad por video, canal,
categoría y comunidad detectada en la red, y conecta el contenido (palabras y
bigramas) con la estructura de co-participación.

El puntaje describe el texto observado en esta muestra: mide el vocabulario
evaluativo que aparece escrito, no la intención, la emoción real ni la postura
política de quien comenta.
"""

from __future__ import annotations

import argparse
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.comunidades_centralidad import (
    ResultadoComunidades,
    analizar_comunidades_y_centralidad,
)
from src.proyecciones_topologia import analizar_proyecciones
from src.red_bipartita import construir_red_bipartita

# ---------------------------------------------------------------------------
# Ficha del recurso léxico
# ---------------------------------------------------------------------------

NOMBRE_LEXICO = "Léxico afectivo en español para el corpus (elaboración propia)"
VERSION_LEXICO = "1.0"
ESCALA_TERMINOS = "[-3, +3] por término"
ESCALA_COMENTARIO = "[-1, +1] tras normalización"
UMBRAL_POLARIDAD = 0.05
MINIMO_MUESTRA = 10  # comentarios necesarios para reportar una comparación
ALFA_NORMALIZACION = 15.0
FACTOR_NEGACION = -0.74
FACTOR_MAYUSCULAS = 1.30
INCREMENTO_EXCLAMACION = 0.29
MAXIMO_EXCLAMACIONES = 3
MAXIMO_REPETICIONES_EMOJI = 3
VENTANA_MODIFICADORES = 3

# Valencia por término. Los pesos están en [-3, +3] y solo cubren vocabulario
# evaluativo: sustantivos temáticos como «diputado», «gobierno» o «internet»
# quedan fuera para no confundir el tema con la polaridad.
LEXICO_ES: dict[str, float] = {
    # --- Negativo fuerte: acusación de delito o corrupción ---
    "corrupto": -3.0, "corrupta": -3.0, "corruptos": -3.0, "corruptas": -3.0,
    "corrupcion": -3.0, "corrupción": -3.0, "corruptela": -3.0,
    "ladron": -3.0, "ladrón": -3.0, "ladrones": -3.0, "rata": -3.0, "ratas": -3.0,
    "robar": -3.0, "roban": -3.0, "robando": -3.0, "robo": -3.0, "robos": -3.0,
    "robaron": -3.0, "mafia": -3.0, "mafioso": -3.0, "mafiosos": -3.0, "narco": -3.0,
    "fraude": -3.0, "estafa": -3.0, "estafan": -3.0, "estafador": -3.0,
    "saqueo": -3.0, "saquean": -3.0, "impunidad": -3.0, "soborno": -3.0, "coima": -3.0,
    "delincuente": -2.5, "delincuentes": -2.5, "criminal": -2.5, "criminales": -2.5,
    "crimen": -2.5, "asalto": -2.5, "asaltan": -2.5, "extorsion": -2.5, "extorsión": -2.5,
    "traidor": -2.5, "traidores": -2.5, "traicion": -2.5, "traición": -2.5,
    "sinverguenza": -2.5, "sinvergüenza": -2.5, "sinverguenzas": -2.5,
    "asco": -2.5, "asqueroso": -2.5, "repugnante": -2.5, "pesimo": -2.5, "pésimo": -2.5,
    "violencia": -2.5, "violento": -2.5, "abuso": -2.5, "abusan": -2.5, "abusivo": -2.5,
    # --- Negativo medio: crítica, carencia y malestar ---
    "verguenza": -2.0, "vergüenza": -2.0, "vergonzoso": -2.0, "descarado": -2.0,
    "cinico": -2.0, "cínico": -2.0, "cinismo": -2.0, "hipocrita": -2.0, "hipócrita": -2.0,
    "mentira": -2.0, "mentiras": -2.0, "mentiroso": -2.0, "mienten": -2.0, "miente": -2.0,
    "injusticia": -2.0, "injusto": -2.0, "impune": -2.0, "arbitrario": -2.0,
    "hambre": -2.0, "pobreza": -2.0, "miseria": -2.0, "desnutricion": -2.0,
    "desnutrición": -2.0, "sufriendo": -2.0, "sufre": -2.0, "sufren": -2.0,
    "sufrimiento": -2.0, "muriendo": -2.0, "muriendose": -2.0, "muriéndose": -2.0,
    "muerte": -2.0, "muertos": -2.0, "lamentable": -2.0, "indignante": -2.0,
    "indignacion": -2.0, "indignación": -2.0, "basura": -2.0, "porqueria": -2.0,
    "porquería": -2.0, "caos": -2.0, "desastre": -2.0,
    "malo": -2.0, "mala": -2.0, "malos": -2.0, "malas": -2.0, "peor": -2.0, "peores": -2.0,
    "terrible": -2.0, "horrible": -2.0, "fatal": -2.0, "grave": -1.5,
    "enojo": -2.0, "enojado": -2.0, "molesto": -1.5, "molesta": -1.5, "harto": -1.5,
    "triste": -2.0, "tristeza": -2.0, "pena": -1.5, "lastima": -1.5, "lástima": -1.5,
    "burla": -1.5, "burlan": -1.5, "engaño": -2.0, "engañan": -2.0,
    "fracaso": -2.0, "fracasado": -2.0, "incapaz": -2.0, "inutil": -2.0, "inútil": -2.0,
    "ignorante": -1.5, "mediocre": -1.5, "excusas": -1.5, "excusa": -1.5,
    # --- Negativo leve: reclamo y dificultad ---
    "problema": -1.0, "problemas": -1.0, "queja": -1.0, "reclamo": -1.0, "reclama": -1.0,
    "dificil": -1.0, "difícil": -1.0, "caro": -1.0, "carisimo": -1.5, "carísimo": -1.5,
    "lento": -1.0, "falla": -1.0, "fallas": -1.0, "atraso": -1.0, "atrasado": -1.0,
    "duda": -0.5, "dudo": -1.0, "preocupa": -1.0, "preocupante": -1.5, "miedo": -1.5,
    "nefasto": -2.0, "abandono": -1.5, "olvidado": -1.0, "sucio": -1.5, "peligroso": -1.5,
    "caros": -1.0, "caras": -1.0, "aburrido": -1.5, "ridiculo": -2.0, "ridículo": -2.0,
    "payaso": -2.0, "payasos": -2.0, "vago": -1.5, "vagos": -1.5, "flojo": -1.0,
    "colmo": -1.5, "barbaridad": -2.0, "atrocidad": -2.5, "cruel": -2.5, "macabro": -2.5,
    "usurpador": -2.5, "dictadura": -2.5, "dictador": -2.5, "tirano": -2.5,
    "secuestrador": -3.0, "secuestro": -3.0, "asesino": -3.0, "asesinato": -3.0,
    "robarle": -3.0, "robarse": -3.0, "roba": -3.0, "robaba": -3.0, "robado": -3.0,
    "sinverguenzada": -2.5, "sinvergüenzada": -2.5, "ilegal": -2.0, "ilegalidad": -2.0,
    "atacan": -1.5, "ataque": -1.5, "amenaza": -2.0, "cinicos": -2.0, "cínicos": -2.0,
    "engañar": -2.0, "manipulan": -2.0, "manipulacion": -2.0, "manipulación": -2.0,
    "desigualdad": -1.5, "abusos": -2.5, "prepotente": -2.0,
    # --- Positivo fuerte ---
    "excelente": 3.0, "excelentes": 3.0, "maravilloso": 3.0, "maravillosa": 3.0,
    "genial": 3.0, "espectacular": 3.0, "extraordinario": 3.0, "amor": 3.0,
    "felicidades": 2.5, "felicitaciones": 2.5, "enhorabuena": 2.5, "orgullo": 2.5,
    "orgulloso": 2.5, "orgullosa": 2.5, "honesto": 2.5, "honesta": 2.5, "honestidad": 2.5,
    "admirable": 2.5, "increible": 2.0, "increíble": 2.0, "buenisima": 2.5,
    "buenísima": 2.5, "buenisimo": 2.5, "buenísimo": 2.5, "feliz": 2.5, "felicidad": 2.5,
    "encanta": 2.5,
    # --- Positivo medio ---
    "gracias": 2.0, "agradezco": 2.0, "agradecido": 2.0, "agradecemos": 2.0,
    "bueno": 2.0, "buena": 2.0, "buenos": 2.0, "buenas": 2.0, "buen": 2.0,
    "bendiciones": 2.0, "bendicion": 2.0, "bendición": 2.0, "esperanza": 2.0,
    "transparencia": 2.0, "transparente": 2.0, "solidaridad": 2.0, "paz": 2.0,
    "exito": 2.0, "éxito": 2.0, "exitoso": 2.0, "logro": 2.0, "logros": 2.0,
    "viva": 2.0, "aplausos": 2.0, "celebro": 2.0, "disfrute": 2.0, "disfruto": 2.0,
    "hermoso": 2.0, "hermosa": 2.0, "lindo": 2.0, "linda": 2.0, "bonito": 2.0,
    "bonita": 2.0, "felicito": 2.0,
    # --- Positivo leve ---
    "bien": 1.5, "mejor": 1.5, "mejores": 1.5, "mejora": 1.5, "mejorar": 1.0,
    "gusta": 1.5, "gustó": 1.5, "gusto": 1.5, "apoyo": 1.5, "apoyar": 1.5, "apoya": 1.5,
    "justicia": 1.5, "justo": 1.5, "respeto": 1.5, "respetuoso": 1.5, "digno": 1.5,
    "interesante": 1.5, "util": 1.5, "útil": 1.5, "correcto": 1.5, "acuerdo": 1.0,
    "adelante": 1.5, "animo": 1.5, "ánimo": 1.5, "esperemos": 0.5, "ojala": 0.5,
    "ojalá": 0.5, "comparto": 1.0, "aprender": 1.0, "aprendi": 1.0, "aprendí": 1.0,
    "saludos": 1.0, "amable": 1.5, "calidad": 1.5, "tranquilo": 1.0,
    "jaja": 1.0, "jajaja": 1.0, "jajajaja": 1.0, "jeje": 1.0, "risa": 1.0,
    "gratis": 1.0, "entretenido": 1.5, "veracidad": 1.5, "claridad": 1.5,
    "sabiduria": 1.5, "sabiduría": 1.5, "esperamos": 0.5, "apoyamos": 1.5,
}

# Modificadores gramaticales aplicados sobre el término evaluativo siguiente.
INTENSIFICADORES: dict[str, float] = {
    "muy": 1.35, "mucho": 1.30, "mucha": 1.30, "muchos": 1.25, "muchas": 1.25,
    "muchisimo": 1.45, "muchísimo": 1.45, "super": 1.40, "súper": 1.40,
    "demasiado": 1.35, "bastante": 1.20, "tan": 1.25, "tanto": 1.20, "tanta": 1.20,
    "totalmente": 1.30, "completamente": 1.30, "realmente": 1.25, "absolutamente": 1.35,
    "extremadamente": 1.50, "sumamente": 1.40, "verdaderamente": 1.25,
}

ATENUADORES: dict[str, float] = {
    "poco": 0.60, "poca": 0.60, "apenas": 0.70, "casi": 0.80, "medio": 0.80,
    "algo": 0.80, "ligeramente": 0.70, "levemente": 0.70, "relativamente": 0.80,
}

NEGACIONES: frozenset[str] = frozenset({
    "no", "ni", "nunca", "jamas", "jamás", "tampoco", "nada", "nadie",
    "ningun", "ningún", "ninguna", "ninguno", "sin",
})

# Emojis y códigos cortos. El archivo recolectado guarda algunos emojis como
# texto (por ejemplo «:hand-purple-blue-peace:»), por lo que se puntúan ambas
# formas. Las banderas y los emojis sin carga evaluativa clara valen cero, y la
# risa recibe una valencia baja porque en este corpus suele marcar burla.
LEXICO_EMOJIS: dict[str, float] = {
    "😂": 1.0, "🤣": 1.0, "😅": 1.0, "😊": 2.0, "😁": 2.0, "😃": 2.0, "😄": 2.0,
    "😸": 1.5, "🥳": 2.5, "🎉": 2.5, "👏": 2.5, "👍": 2.0, "🙌": 2.0, "👌": 1.5,
    "❤": 3.0, "💙": 2.5, "💕": 2.5, "🥰": 3.0, "😍": 3.0, "🙏": 1.5,
    "💯": 2.0, "🔥": 1.5, "🥹": 1.0, "💪": 1.5, "✅": 1.0, "🤝": 1.5,
    "😡": -3.0, "🤬": -3.0, "😠": -2.5, "🤮": -3.0, "🤢": -2.5, "💩": -3.0,
    "👹": -2.5, "😢": -2.0, "😭": -2.0, "😞": -2.0, "😔": -1.5, "😤": -2.0,
    "🙄": -1.5, "😒": -1.5, "👎": -2.5, "💔": -2.5, "😷": -1.0,
}

LEXICO_CODIGOS: dict[str, float] = {
    ":hand-purple-blue-peace:": 1.5,
    ":face-blue-smiling:": 2.0,
}

PATRON_EMOJI = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF]"
)
PATRON_CODIGO = re.compile(r":[a-z0-9]+(?:-[a-z0-9]+)+:")
PATRON_PALABRA = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+")

# Mismas stopwords que la limpieza de texto de la sección de preprocesamiento.
STOPWORDS_ES: frozenset[str] = frozenset({
    "a", "al", "algo", "ante", "antes", "aqui", "asi", "aunque", "bajo", "bien", "cada",
    "casi", "como", "con", "contra", "cual", "cuando", "de", "del", "desde", "donde",
    "dos", "e", "el", "ella", "ellos", "en", "entre", "era", "es", "esa", "ese", "eso",
    "esta", "este", "esto", "fue", "ha", "han", "hasta", "hay", "la", "las", "le", "les",
    "lo", "los", "mas", "me", "mi", "muy", "no", "nos", "o", "os", "para", "pero", "poco",
    "por", "porque", "que", "quien", "se", "ser", "si", "sin", "sobre", "son", "su", "sus",
    "tambien", "te", "tiene", "todo", "tu", "un", "una", "uno", "unos", "ustedes", "va",
    "ya", "y", "yo",
})

COLORES_SENTIMIENTO = {
    "positivo": "#1B9E77",
    "neutral": "#94A3B8",
    "negativo": "#D95F02",
    "sin cobertura léxica": "#CBD5E1",
    "sin texto": "#E2E8F0",
}


@dataclass(frozen=True)
class ResultadoSentimiento:
    """Tablas del análisis de contenido, sentimiento y conclusiones."""

    comentarios: pd.DataFrame
    resumen_general: pd.DataFrame
    por_video: pd.DataFrame
    por_canal: pd.DataFrame
    por_categoria: pd.DataFrame
    por_comunidad: pd.DataFrame
    contenido_comunidad: pd.DataFrame
    afirmaciones: pd.DataFrame
    validacion: pd.DataFrame


# ---------------------------------------------------------------------------
# Puntuación de un comentario
# ---------------------------------------------------------------------------


def limpiar_texto(valor: object) -> str:
    """Normaliza el texto para frecuencias: minúsculas, sin URL, signos ni stopwords."""

    texto = "" if pd.isna(valor) else str(valor).lower()
    texto = PATRON_CODIGO.sub(" ", texto)
    texto = re.sub(r"https?://\S+|www\.\S+", " ", texto)
    texto = re.sub(r"([#@])([\wáéíóúüñ]+)", r" \2 ", texto)
    texto = re.sub(r"[^\wáéíóúüñ\s]", " ", texto, flags=re.IGNORECASE)
    tokens = [
        token for token in texto.split()
        if not token.isdigit() and token not in STOPWORDS_ES and len(token) > 1
    ]
    return " ".join(tokens)


def _factor_modificadores(palabras: list[str], posicion: int) -> tuple[float, bool]:
    """Devuelve el factor de intensidad y si hay negación en la ventana previa.

    Se revisan hasta ``VENTANA_MODIFICADORES`` palabras anteriores. Los
    intensificadores y atenuadores pierden efecto conforme se alejan del término
    evaluativo; la negación invierte el signo una sola vez y detiene la ventana.
    """

    factor = 1.0
    for distancia in range(1, VENTANA_MODIFICADORES + 1):
        indice = posicion - distancia
        if indice < 0:
            break
        previa = palabras[indice]
        if previa in NEGACIONES:
            return factor, True
        atenuacion = 1.0 - 0.25 * (distancia - 1)
        if previa in INTENSIFICADORES:
            factor *= 1.0 + (INTENSIFICADORES[previa] - 1.0) * atenuacion
        elif previa in ATENUADORES:
            factor *= 1.0 - (1.0 - ATENUADORES[previa]) * atenuacion
    return factor, False


def puntuar_texto(valor: object) -> dict[str, object]:
    """Puntúa un comentario y devuelve el detalle auditable de esa puntuación.

    El puntaje bruto suma la valencia de los términos y emojis reconocidos tras
    aplicar negación, intensificadores, mayúsculas sostenidas y énfasis por
    signos de exclamación. Después se normaliza a ``[-1, +1]`` mediante
    ``puntaje / sqrt(puntaje^2 + ALFA_NORMALIZACION)``, de modo que un
    comentario largo no supere automáticamente a uno corto.
    """

    texto = "" if pd.isna(valor) else str(valor)
    vacio = not texto.strip()

    codigos = PATRON_CODIGO.findall(texto.lower())
    emojis = PATRON_EMOJI.findall(texto)
    palabras_originales = PATRON_PALABRA.findall(texto)
    palabras = [palabra.lower() for palabra in palabras_originales]

    puntaje = 0.0
    positivos: list[str] = []
    negativos: list[str] = []
    negaciones = 0

    for posicion, palabra in enumerate(palabras):
        base = LEXICO_ES.get(palabra)
        if base is None:
            continue
        valor_termino = base
        original = palabras_originales[posicion]
        if len(original) > 2 and original.isupper():
            valor_termino *= FACTOR_MAYUSCULAS
        factor, negado = _factor_modificadores(palabras, posicion)
        valor_termino *= factor
        if negado:
            valor_termino *= FACTOR_NEGACION
            negaciones += 1
        puntaje += valor_termino
        (positivos if valor_termino > 0 else negativos).append(palabra)

    # Un mismo emoji repetido enfatiza, pero no multiplica el puntaje sin
    # límite: cada símbolo aporta como máximo MAXIMO_REPETICIONES_EMOJI veces.
    emojis_valorados = [emoji for emoji in emojis if LEXICO_EMOJIS.get(emoji)]
    codigos_valorados = [codigo for codigo in codigos if codigo in LEXICO_CODIGOS]
    for simbolo, repeticiones in Counter(emojis_valorados).items():
        puntaje += LEXICO_EMOJIS[simbolo] * min(repeticiones, MAXIMO_REPETICIONES_EMOJI)
    for codigo, repeticiones in Counter(codigos_valorados).items():
        puntaje += LEXICO_CODIGOS[codigo] * min(repeticiones, MAXIMO_REPETICIONES_EMOJI)

    exclamaciones = min(texto.count("!"), MAXIMO_EXCLAMACIONES)
    if exclamaciones and puntaje != 0:
        puntaje += math.copysign(INCREMENTO_EXCLAMACION * exclamaciones, puntaje)

    compuesto = puntaje / math.sqrt(puntaje**2 + ALFA_NORMALIZACION) if puntaje else 0.0
    compuesto = float(np.clip(compuesto, -1.0, 1.0))

    cubierto = bool(positivos or negativos or emojis_valorados or codigos_valorados)
    if vacio:
        etiqueta = "sin texto"
    elif not cubierto:
        etiqueta = "sin cobertura léxica"
    elif compuesto >= UMBRAL_POLARIDAD:
        etiqueta = "positivo"
    elif compuesto <= -UMBRAL_POLARIDAD:
        etiqueta = "negativo"
    else:
        etiqueta = "neutral"

    return {
        "puntaje_bruto": round(puntaje, 4),
        "sentimiento": round(compuesto, 4),
        "categoria_sentimiento": etiqueta,
        "terminos_lexico": len(positivos) + len(negativos),
        "terminos_positivos": " | ".join(positivos),
        "terminos_negativos": " | ".join(negativos),
        "emojis_valorados": "".join(emojis_valorados) + "".join(codigos_valorados),
        "negaciones_aplicadas": negaciones,
        "texto_vacio": vacio,
        "cobertura_lexica": cubierto,
    }


def calcular_sentimiento(
    comentarios_df: pd.DataFrame,
    videos_df: pd.DataFrame,
) -> pd.DataFrame:
    """Puntúa todos los comentarios y adjunta el contexto del video comentado."""

    base = comentarios_df.copy()
    base["video_id"] = base["video_id"].astype(str).str.strip()
    if "texto_original" not in base.columns:
        base["texto_original"] = base["text"].fillna("")
    if "texto_limpio" not in base.columns:
        base["texto_limpio"] = base["texto_original"].map(limpiar_texto)

    puntajes = pd.DataFrame(
        list(base["texto_original"].map(puntuar_texto)), index=base.index
    )
    base = pd.concat([base.drop(columns=puntajes.columns, errors="ignore"), puntajes], axis=1)

    contexto = videos_df.copy()
    contexto["video_id"] = contexto["video_id"].astype(str).str.strip()
    columnas_contexto = ["video_id", "title", "channel_name", "category", "view_count"]
    contexto = contexto[[c for c in columnas_contexto if c in contexto.columns]]
    base = base.merge(
        contexto, on="video_id", how="left", validate="many_to_one", suffixes=("", "_video")
    )

    titulo = base["title"] if "title" in base.columns else base["video_title"]
    base["titulo_video"] = titulo.fillna(base.get("video_title", "Sin título"))
    canal = base["channel_name_video"] if "channel_name_video" in base.columns else base["channel_name"]
    base["canal"] = canal.fillna("Sin canal")
    base["categoria_video"] = base["category"].fillna("Sin categoría")

    columnas = [
        "comment_id", "video_id", "author_channel_id", "author_handle",
        "titulo_video", "canal", "categoria_video",
        "texto_original", "texto_limpio",
        "puntaje_bruto", "sentimiento", "categoria_sentimiento", "terminos_lexico",
        "terminos_positivos", "terminos_negativos", "emojis_valorados",
        "negaciones_aplicadas", "texto_vacio", "cobertura_lexica",
    ]
    return base[[c for c in columnas if c in base.columns]]


# ---------------------------------------------------------------------------
# Resúmenes agregados
# ---------------------------------------------------------------------------


def _resumir(df: pd.DataFrame, clave: str, etiqueta: str) -> pd.DataFrame:
    """Resume el sentimiento por una llave de agrupación.

    ``muestra_suficiente`` marca los grupos con al menos ``MINIMO_MUESTRA``
    comentarios. Los grupos por debajo del umbral se conservan en la tabla, pero
    quedan etiquetados para que no se comparen como si fueran equivalentes.
    """

    grupos = df.groupby(clave, dropna=False)
    resumen = pd.DataFrame({
        "comentarios": grupos.size(),
        "autores_unicos": grupos["author_channel_id"].nunique(),
        "sentimiento_medio": grupos["sentimiento"].mean().round(4),
        "sentimiento_mediano": grupos["sentimiento"].median().round(4),
        "desviacion": grupos["sentimiento"].std().fillna(0.0).round(4),
        "positivos": grupos["categoria_sentimiento"].apply(
            lambda serie: int((serie == "positivo").sum())
        ),
        "negativos": grupos["categoria_sentimiento"].apply(
            lambda serie: int((serie == "negativo").sum())
        ),
        "neutrales_o_sin_cobertura": grupos["categoria_sentimiento"].apply(
            lambda serie: int(serie.isin(
                ["neutral", "sin cobertura léxica", "sin texto"]
            ).sum())
        ),
        "cobertura_lexica": grupos["cobertura_lexica"].mean().round(4),
    }).reset_index().rename(columns={clave: etiqueta})

    total = resumen["comentarios"]
    resumen["porcentaje_positivos"] = (100 * resumen["positivos"] / total).round(2)
    resumen["porcentaje_negativos"] = (100 * resumen["negativos"] / total).round(2)
    resumen["muestra_suficiente"] = total >= MINIMO_MUESTRA
    resumen["nota_comparacion"] = np.where(
        resumen["muestra_suficiente"],
        "comparable",
        f"muestra insuficiente (< {MINIMO_MUESTRA} comentarios): no comparar",
    )
    return resumen.sort_values(
        ["comentarios", "sentimiento_medio"], ascending=False
    ).reset_index(drop=True)


def _frecuencias(textos: pd.Series, cantidad: int = 10) -> tuple[str, str]:
    """Devuelve las palabras y bigramas más frecuentes de un conjunto de textos."""

    palabras: Counter = Counter()
    bigramas: Counter = Counter()
    for texto in textos.fillna(""):
        tokens = texto.split()
        palabras.update(tokens)
        bigramas.update(f"{a} {b}" for a, b in zip(tokens, tokens[1:]))
    top_palabras = ", ".join(f"{p} ({n})" for p, n in palabras.most_common(cantidad))
    top_bigramas = ", ".join(f"{b} ({n})" for b, n in bigramas.most_common(cantidad))
    return top_palabras, top_bigramas


def resumir_contenido_por_comunidad(
    df: pd.DataFrame,
    comunidades_resumen: pd.DataFrame,
) -> pd.DataFrame:
    """Cruza el contenido textual con las comunidades detectadas en la red.

    Cada comentario hereda la comunidad del video donde fue publicado. Es una
    atribución indirecta: describe de qué se habla en videos que comparten
    audiencia, no una conversación entre los autores.
    """

    nombres = dict(zip(
        comunidades_resumen["comunidad_id"], comunidades_resumen["nombre_comunidad"]
    ))
    filas = []
    for comunidad, grupo in df.groupby("comunidad_id"):
        top_palabras, top_bigramas = _frecuencias(grupo["texto_limpio"])
        filas.append({
            "comunidad_id": int(comunidad),
            "nombre_comunidad": nombres.get(
                comunidad, "Videos sin co-comentación observada"
            ),
            "videos_con_comentarios": grupo["video_id"].nunique(),
            "comentarios": len(grupo),
            "autores_unicos": grupo["author_channel_id"].nunique(),
            "sentimiento_medio": round(grupo["sentimiento"].mean(), 4),
            "sentimiento_mediano": round(grupo["sentimiento"].median(), 4),
            "porcentaje_positivos": round(
                100 * (grupo["categoria_sentimiento"] == "positivo").mean(), 2
            ),
            "porcentaje_negativos": round(
                100 * (grupo["categoria_sentimiento"] == "negativo").mean(), 2
            ),
            "cobertura_lexica": round(grupo["cobertura_lexica"].mean(), 4),
            "palabras_frecuentes": top_palabras,
            "bigramas_frecuentes": top_bigramas,
            "muestra_suficiente": len(grupo) >= MINIMO_MUESTRA,
        })
    return pd.DataFrame(filas).sort_values("comunidad_id").reset_index(drop=True)


def _tabla_afirmaciones(
    contenido_comunidad: pd.DataFrame,
    por_categoria: pd.DataFrame,
    resumen_general: pd.DataFrame,
) -> pd.DataFrame:
    """Clasifica las afirmaciones del informe en descripción, asociación o inferencia."""

    comparables = int(contenido_comunidad["muestra_suficiente"].sum())
    categorias_comparables = int(por_categoria["muestra_suficiente"].sum())
    general = resumen_general.iloc[0]
    return pd.DataFrame([
        {
            "afirmacion": (
                f"La muestra reúne {int(general['negativos'])} comentarios negativos y "
                f"{int(general['positivos'])} positivos según el léxico aplicado."
            ),
            "tipo": "descripción",
            "alcance": "Válida para los comentarios recolectados en esta muestra.",
            "lenguaje_requerido": "Directo: se observa en la muestra.",
        },
        {
            "afirmacion": "El corpus completo tiene un tono predominantemente negativo.",
            "tipo": "inferencia no sostenible",
            "alcance": (
                f"El promedio del corpus es {float(general['sentimiento_medio']):+.4f}: "
                "el tono negativo se concentra en videos concretos, no en el conjunto."
            ),
            "lenguaje_requerido": "Evitar: precisar en qué videos se observa.",
        },
        {
            "afirmacion": "Los grupos de videos con audiencia compartida difieren en tono.",
            "tipo": "asociación",
            "alcance": f"Solo entre los {comparables} grupos con muestra suficiente.",
            "lenguaje_requerido": "Cauteloso: se asocia, no se demuestra causa.",
        },
        {
            "afirmacion": "La categoría del video determina el sentimiento de su audiencia.",
            "tipo": "inferencia no sostenible",
            "alcance": f"Solo {categorias_comparables} categorías superan el umbral de muestra.",
            "lenguaje_requerido": "Evitar: el diseño no permite afirmar causalidad.",
        },
        {
            "afirmacion": "Los autores puente trasladan un tono común entre comunidades.",
            "tipo": "inferencia no sostenible",
            "alcance": "Los autores multividales aportan muy pocos comentarios cada uno.",
            "lenguaje_requerido": "Evitar o presentar como hipótesis a verificar.",
        },
        {
            "afirmacion": "Los videos sin comentarios recuperados carecen de participación.",
            "tipo": "inferencia no sostenible",
            "alcance": "El aislamiento es de la recolección, no de la plataforma.",
            "lenguaje_requerido": "Evitar: describirlo como cobertura parcial.",
        },
        {
            "afirmacion": "El sentimiento medido refleja la emoción real de quien comenta.",
            "tipo": "inferencia no sostenible",
            "alcance": "El léxico mide vocabulario escrito, no estados internos.",
            "lenguaje_requerido": "Evitar: hablar del tono del texto observado.",
        },
    ])


def analizar_contenido_y_sentimiento(
    comentarios_df: pd.DataFrame,
    videos_df: pd.DataFrame,
    resultado_comunidades: ResultadoComunidades,
) -> ResultadoSentimiento:
    """Ejecuta el análisis de sentimiento y su conexión con la estructura de red."""

    df = calcular_sentimiento(comentarios_df, videos_df)
    asignacion = resultado_comunidades.asignacion_comunidades
    df["comunidad_id"] = df["video_id"].map(
        lambda video: asignacion.get(f"video::{video}", 0)
    )

    resumen_general = pd.DataFrame([{
        "comentarios_evaluados": len(df),
        "sentimiento_medio": round(df["sentimiento"].mean(), 4),
        "sentimiento_mediano": round(df["sentimiento"].median(), 4),
        "positivos": int((df["categoria_sentimiento"] == "positivo").sum()),
        "neutrales": int((df["categoria_sentimiento"] == "neutral").sum()),
        "negativos": int((df["categoria_sentimiento"] == "negativo").sum()),
        "sin_cobertura_lexica": int(
            (df["categoria_sentimiento"] == "sin cobertura léxica").sum()
        ),
        "sin_texto": int((df["categoria_sentimiento"] == "sin texto").sum()),
        "cobertura_lexica": round(df["cobertura_lexica"].mean(), 4),
        "comentarios_con_emoji_valorado": int(df["emojis_valorados"].str.len().gt(0).sum()),
        "comentarios_con_negacion": int(df["negaciones_aplicadas"].gt(0).sum()),
    }])

    por_video = _resumir(df, "titulo_video", "video")
    por_canal = _resumir(df, "canal", "canal")
    por_categoria = _resumir(df, "categoria_video", "categoria")
    por_comunidad = _resumir(df, "comunidad_id", "comunidad_id")
    contenido_comunidad = resumir_contenido_por_comunidad(
        df, resultado_comunidades.comunidades_resumen
    )
    afirmaciones = _tabla_afirmaciones(contenido_comunidad, por_categoria, resumen_general)

    validacion = pd.DataFrame([
        {"indicador": "comentarios_puntuados",
         "valor": len(df) == len(comentarios_df)},
        {"indicador": "todos_con_video_identificado",
         "valor": bool(df["titulo_video"].notna().all())},
        {"indicador": "sentimiento_dentro_de_escala",
         "valor": bool(df["sentimiento"].between(-1, 1).all())},
        {"indicador": "categorias_suman_total",
         "valor": int(df["categoria_sentimiento"].value_counts().sum()) == len(df)},
        {"indicador": "comentarios_asignados_a_comunidad",
         "valor": bool(df["comunidad_id"].notna().all())},
        {"indicador": "comunidades_cubiertas_en_contenido",
         "valor": set(contenido_comunidad["comunidad_id"])
         == set(df["comunidad_id"].unique())},
        {"indicador": "grupos_pequenos_marcados",
         "valor": bool((~por_video["muestra_suficiente"]).any())},
    ])

    return ResultadoSentimiento(
        comentarios=df,
        resumen_general=resumen_general,
        por_video=por_video,
        por_canal=por_canal,
        por_categoria=por_categoria,
        por_comunidad=por_comunidad,
        contenido_comunidad=contenido_comunidad,
        afirmaciones=afirmaciones,
        validacion=validacion,
    )


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------


def visualizar_sentimiento_general(
    resultado: ResultadoSentimiento,
    ruta_salida: Path | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Distribución del sentimiento y comparación por canal, categoría y video."""

    df = resultado.comentarios
    figura, ejes = plt.subplots(2, 2, figsize=(16, 10))

    ejes[0, 0].hist(df["sentimiento"], bins=30, color="#4C78A8", edgecolor="white")
    ejes[0, 0].axvline(0, color="#0F172A", linewidth=1, linestyle="--")
    ejes[0, 0].set(
        title=f"Distribución del sentimiento por comentario (n = {len(df):,})",
        xlabel="Puntaje normalizado [-1, +1]",
        ylabel="Comentarios",
    )

    conteos = df["categoria_sentimiento"].value_counts()
    ejes[0, 1].bar(
        conteos.index,
        conteos.values,
        color=[COLORES_SENTIMIENTO.get(c, "#94A3B8") for c in conteos.index],
    )
    for indice, valor in enumerate(conteos.values):
        ejes[0, 1].text(
            indice, valor, f"{valor}\n({100 * valor / len(df):.1f}%)",
            ha="center", va="bottom", fontsize=9,
        )
    ejes[0, 1].set(title="Comentarios por categoría de sentimiento", ylabel="Comentarios")
    ejes[0, 1].tick_params(axis="x", rotation=20)
    ejes[0, 1].margins(y=0.18)

    canales = resultado.por_canal.sort_values("sentimiento_medio")
    colores_canal = [
        "#1B9E77" if suficiente else "#CBD5E1" for suficiente in canales["muestra_suficiente"]
    ]
    ejes[1, 0].barh(
        [f"{c} (n={n})" for c, n in zip(canales["canal"], canales["comentarios"])],
        canales["sentimiento_medio"],
        color=colores_canal,
    )
    ejes[1, 0].axvline(0, color="#0F172A", linewidth=1)
    ejes[1, 0].set(
        title=f"Sentimiento medio por canal (gris: n < {MINIMO_MUESTRA}, no comparable)",
        xlabel="Sentimiento medio",
    )

    videos = resultado.por_video[resultado.por_video["muestra_suficiente"]].sort_values(
        "sentimiento_medio"
    )
    ejes[1, 1].barh(
        [f"{t[:45]} (n={n})" for t, n in zip(videos["video"], videos["comentarios"])],
        videos["sentimiento_medio"],
        color="#7570B3",
    )
    ejes[1, 1].axvline(0, color="#0F172A", linewidth=1)
    ejes[1, 1].set(
        title=f"Sentimiento medio por video con n >= {MINIMO_MUESTRA}",
        xlabel="Sentimiento medio",
    )

    figura.suptitle(
        "Sentimiento de los comentarios recolectados — léxico en español, escala [-1, +1]",
        fontsize=14,
    )
    figura.tight_layout()
    if ruta_salida is not None:
        figura.savefig(ruta_salida, dpi=200, bbox_inches="tight")
    return figura, ejes


def visualizar_sentimiento_comunidades(
    resultado: ResultadoSentimiento,
    ruta_salida: Path | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Conecta el contenido con la red: tono y vocabulario de cada comunidad."""

    contenido = resultado.contenido_comunidad.copy()
    etiquetas = [
        f"C{fila.comunidad_id}: {fila.nombre_comunidad[:34]}\n(n={fila.comentarios})"
        if fila.comunidad_id > 0
        else f"Sin co-comentación\n(n={fila.comentarios})"
        for fila in contenido.itertuples()
    ]

    figura, ejes = plt.subplots(1, 3, figsize=(19, 7), gridspec_kw={"width_ratios": [1, 1, 1.3]})

    colores = [
        "#D95F02" if valor < 0 else "#1B9E77" for valor in contenido["sentimiento_medio"]
    ]
    ejes[0].barh(etiquetas, contenido["sentimiento_medio"], color=colores)
    ejes[0].axvline(0, color="#0F172A", linewidth=1)
    ejes[0].set(title="Sentimiento medio por comunidad de videos", xlabel="Sentimiento medio")

    positivos = contenido["porcentaje_positivos"]
    negativos = contenido["porcentaje_negativos"]
    restantes = 100 - positivos - negativos
    ejes[1].barh(etiquetas, negativos, color="#D95F02", label="negativos")
    ejes[1].barh(etiquetas, restantes, left=negativos, color="#94A3B8",
                 label="neutrales o sin cobertura")
    ejes[1].barh(etiquetas, positivos, left=negativos + restantes, color="#1B9E77",
                 label="positivos")
    ejes[1].set(title="Composición del tono (%)", xlabel="Porcentaje de comentarios")
    ejes[1].legend(loc="lower right", fontsize=9)

    ejes[2].set_axis_off()
    ejes[2].set_title("Vocabulario más frecuente por comunidad", fontsize=12)
    posicion = 0.97
    for fila in contenido.itertuples():
        nombre = (
            f"C{fila.comunidad_id} — {fila.nombre_comunidad}"
            if fila.comunidad_id > 0
            else "Videos sin co-comentación observada"
        )
        palabras = ", ".join(fila.palabras_frecuentes.split(", ")[:6])
        bigramas = ", ".join(fila.bigramas_frecuentes.split(", ")[:3])
        ejes[2].text(0, posicion, nombre, fontsize=10, fontweight="bold", va="top")
        ejes[2].text(0, posicion - 0.05, f"palabras: {palabras}", fontsize=9, va="top", wrap=True)
        ejes[2].text(0, posicion - 0.10, f"bigramas: {bigramas}", fontsize=9, va="top", wrap=True)
        posicion -= 0.24

    figura.suptitle(
        "Contenido y sentimiento por comunidad detectada en la proyección video–video",
        fontsize=14,
    )
    figura.tight_layout()
    if ruta_salida is not None:
        figura.savefig(ruta_salida, dpi=200, bbox_inches="tight")
    return figura, ejes


# ---------------------------------------------------------------------------
# Documento metodológico, limitaciones y conclusiones
# ---------------------------------------------------------------------------


def _texto_resultados(resultado: ResultadoSentimiento) -> str:
    """Redacta la memoria metodológica, las limitaciones y las conclusiones."""

    general = resultado.resumen_general.iloc[0]
    contenido = resultado.contenido_comunidad
    conectadas = contenido[contenido["comunidad_id"] > 0]
    aislados = contenido[contenido["comunidad_id"] == 0]
    videos_comparables = resultado.por_video[resultado.por_video["muestra_suficiente"]]
    canales_comparables = resultado.por_canal[resultado.por_canal["muestra_suficiente"]]
    categorias_comparables = resultado.por_categoria[
        resultado.por_categoria["muestra_suficiente"]
    ]

    total = int(general["comentarios_evaluados"])
    negativos = int(general["negativos"])
    positivos = int(general["positivos"])
    neutrales = int(general["neutrales"])
    sin_cobertura = int(general["sin_cobertura_lexica"])

    lineas_comunidad = []
    for fila in conectadas.itertuples():
        lineas_comunidad.append(
            f"- **C{fila.comunidad_id} — {fila.nombre_comunidad}** "
            f"({fila.comentarios} comentarios de {fila.autores_unicos} autores en "
            f"{fila.videos_con_comentarios} videos): sentimiento medio "
            f"{fila.sentimiento_medio:+.4f}, con {fila.porcentaje_negativos:.1f}% de "
            f"comentarios negativos y {fila.porcentaje_positivos:.1f}% positivos. "
            f"Palabras frecuentes: {', '.join(fila.palabras_frecuentes.split(', ')[:6])}. "
            f"Bigramas: {', '.join(fila.bigramas_frecuentes.split(', ')[:3])}."
        )
    bloque_comunidades = "\n".join(lineas_comunidad)

    lineas_videos = []
    for fila in videos_comparables.sort_values("sentimiento_medio").itertuples():
        lineas_videos.append(
            f"| {fila.video[:60]} | {fila.comentarios} | {fila.sentimiento_medio:+.4f} | "
            f"{fila.porcentaje_negativos:.1f}% | {fila.porcentaje_positivos:.1f}% |"
        )
    tabla_videos = "\n".join(lineas_videos)

    lineas_canales = []
    for fila in canales_comparables.sort_values("sentimiento_medio").itertuples():
        lineas_canales.append(
            f"| {fila.canal} | {fila.comentarios} | {fila.sentimiento_medio:+.4f} | "
            f"{fila.porcentaje_negativos:.1f}% |"
        )
    tabla_canales = "\n".join(lineas_canales)

    lineas_categorias = []
    for fila in resultado.por_categoria.itertuples():
        nota = "comparable" if fila.muestra_suficiente else "muestra insuficiente"
        lineas_categorias.append(
            f"| {fila.categoria} | {fila.comentarios} | {fila.sentimiento_medio:+.4f} | {nota} |"
        )
    tabla_categorias = "\n".join(lineas_categorias)

    porcentaje_negativos = f"{100 * negativos / total:.1f}%"
    porcentaje_positivos = f"{100 * positivos / total:.1f}%"
    porcentaje_neutrales = f"{100 * neutrales / total:.1f}%"
    porcentaje_sin_cobertura = f"{100 * sin_cobertura / total:.1f}%"
    porcentaje_cobertura = f"{100 * float(general['cobertura_lexica']):.1f}%"
    media_general = f"{float(general['sentimiento_medio']):+.4f}"
    mediana_general = f"{float(general['sentimiento_mediano']):+.4f}"

    videos_con_comentarios = int(contenido["videos_con_comentarios"].sum())
    orden_comunidades = conectadas.sort_values("sentimiento_medio")
    comunidad_negativa = orden_comunidades.iloc[0]
    comunidad_positiva = orden_comunidades.iloc[-1]
    video_negativo = videos_comparables.sort_values("sentimiento_medio").iloc[0]

    fila_aislados = aislados.iloc[0] if len(aislados) else None
    texto_aislados = (
        f"Los {int(fila_aislados['comentarios'])} comentarios publicados en videos sin "
        f"co-comentación observada alcanzan un sentimiento medio de "
        f"{float(fila_aislados['sentimiento_medio']):+.4f}. Se reportan aparte porque no "
        "pertenecen a ninguna comunidad de la red: su aislamiento proviene de la "
        "cobertura de la recolección."
        if fila_aislados is not None
        else "Todos los comentarios pertenecen a videos con co-comentación observada."
    )

    return f"""# Contenido, sentimiento, limitaciones y conclusiones

## 1. Recurso léxico y decisiones de medición

| Elemento | Decisión |
|---|---|
| Herramienta | {NOMBRE_LEXICO}, versión {VERSION_LEXICO} |
| Implementación | `src/sentimiento_contenido.py`, sin dependencias externas ni descargas |
| Escala por término | {ESCALA_TERMINOS} |
| Escala por comentario | {ESCALA_COMENTARIO} |
| Normalización | `puntaje / sqrt(puntaje^2 + {ALFA_NORMALIZACION:.0f})` |
| Categorías | positivo (>= {UMBRAL_POLARIDAD}), negativo (<= -{UMBRAL_POLARIDAD}), neutral, sin cobertura léxica, sin texto |
| Negación | `no, ni, nunca, jamás, tampoco, nada, nadie, ningún, sin` invierten el signo (factor {FACTOR_NEGACION}) en una ventana de {VENTANA_MODIFICADORES} palabras |
| Intensificadores y atenuadores | `muy`, `demasiado`, `súper`, `bastante` amplifican; `poco`, `apenas`, `casi` reducen, con efecto decreciente según la distancia |
| Mayúsculas | una palabra evaluativa en mayúsculas sostenidas amplifica su valencia (x{FACTOR_MAYUSCULAS}) |
| Exclamaciones | hasta {MAXIMO_EXCLAMACIONES} signos suman {INCREMENTO_EXCLAMACION} cada uno, conservando el signo del puntaje |
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

De los {total} comentarios evaluados, {negativos} resultan negativos ({porcentaje_negativos}), {positivos} positivos ({porcentaje_positivos}) y {neutrales} neutrales ({porcentaje_neutrales}). Otros {sin_cobertura} comentarios ({porcentaje_sin_cobertura}) no contienen ningún término ni emoji del léxico: se reportan como **sin cobertura léxica** y no se interpretan como neutralidad.

El sentimiento medio del corpus es {media_general} y la mediana {mediana_general}. La cobertura léxica media es {porcentaje_cobertura}; {int(general['comentarios_con_emoji_valorado'])} comentarios aportan al menos un emoji con valencia y {int(general['comentarios_con_negacion'])} activan la regla de negación.

### Videos con muestra suficiente (n >= {MINIMO_MUESTRA})

| Video | Comentarios | Sentimiento medio | % negativos | % positivos |
|---|---|---|---|---|
{tabla_videos}

Los videos con menos de {MINIMO_MUESTRA} comentarios permanecen en
`resultados/tablas/sentimiento_por_video.csv` marcados como no comparables.

### Canales con muestra suficiente

| Canal | Comentarios | Sentimiento medio | % negativos |
|---|---|---|---|
{tabla_canales}

### Categorías

| Categoría | Comentarios | Sentimiento medio | Estado |
|---|---|---|---|
{tabla_categorias}

Solo {len(categorias_comparables)} categorías superan el umbral de muestra, de
modo que la diferencia entre categorías se describe pero no se usa como
evidencia de un patrón general.

## 3. Contenido y red: qué se dice en cada comunidad

{bloque_comunidades}

{texto_aislados}

La lectura conjunta es de asociación, no de causa: los videos de una misma
comunidad comparten al menos un comentarista y, además, comparten registro
temático. Nada en estos datos indica que la comunidad provoque el tono, ni que
los autores puente lo trasladen de un video a otro.

## 4. Limitaciones

1. **Cobertura parcial de comentarios.** Se recuperaron {total} comentarios
   principales para {videos_con_comentarios} videos de los 293 recolectados. No
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
   quedan fuera del alcance del recurso; el {porcentaje_sin_cobertura} de
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
5. **Sentimiento (descripción con cautela).** El tono no es uniforme y el promedio del corpus ({media_general}) no describe bien a ningún grupo: el registro negativo se concentra en la comunidad «{comunidad_negativa['nombre_comunidad']}» ({float(comunidad_negativa['sentimiento_medio']):+.4f}) y en el video «{video_negativo.video[:55]}» ({video_negativo.sentimiento_medio:+.4f}), mientras que «{comunidad_positiva['nombre_comunidad']}» resulta claramente más positiva ({float(comunidad_positiva['sentimiento_medio']):+.4f}). La medición captura el léxico escrito y no debe leerse como el estado emocional de las personas ni como su posición política.
6. **Alcance (inferencia acotada).** Ninguna de estas conclusiones se extiende
   a todos los usuarios de YouTube ni a la opinión pública guatemalteca. Son
   válidas para el conjunto recolectado, en el momento en que se descargó, y
   bajo las definiciones de arista y de peso declaradas en el informe.

La tabla `resultados/tablas/afirmaciones_lenguaje_cauteloso.csv` lista las
afirmaciones que requieren lenguaje cauteloso y las que no deben sostenerse con
estos datos.
"""


# ---------------------------------------------------------------------------
# Exportación y ejecución
# ---------------------------------------------------------------------------


def exportar_resultados(
    resultado: ResultadoSentimiento,
    directorio: Path,
) -> dict[str, Path]:
    """Exporta tablas, figuras y memoria metodológica de los ejercicios 9 y 10."""

    directorio = Path(directorio)
    tablas = directorio / "tablas"
    figuras = directorio / "figuras"
    tablas.mkdir(parents=True, exist_ok=True)
    figuras.mkdir(parents=True, exist_ok=True)

    rutas = {
        "comentarios": tablas / "sentimiento_comentarios.csv",
        "resumen_general": tablas / "sentimiento_resumen_general.csv",
        "por_video": tablas / "sentimiento_por_video.csv",
        "por_canal": tablas / "sentimiento_por_canal.csv",
        "por_categoria": tablas / "sentimiento_por_categoria.csv",
        "por_comunidad": tablas / "sentimiento_por_comunidad.csv",
        "contenido_comunidad": tablas / "contenido_por_comunidad.csv",
        "afirmaciones": tablas / "afirmaciones_lenguaje_cauteloso.csv",
        "validacion": tablas / "validacion_sentimiento.csv",
        "figura_general": figuras / "sentimiento_general.png",
        "figura_comunidades": figuras / "sentimiento_comunidades.png",
        "metodologia": directorio / "metodologia_sentimiento_contenido.md",
    }

    resultado.comentarios.to_csv(rutas["comentarios"], index=False, encoding="utf-8-sig")
    resultado.resumen_general.to_csv(rutas["resumen_general"], index=False, encoding="utf-8-sig")
    resultado.por_video.to_csv(rutas["por_video"], index=False, encoding="utf-8-sig")
    resultado.por_canal.to_csv(rutas["por_canal"], index=False, encoding="utf-8-sig")
    resultado.por_categoria.to_csv(rutas["por_categoria"], index=False, encoding="utf-8-sig")
    resultado.por_comunidad.to_csv(rutas["por_comunidad"], index=False, encoding="utf-8-sig")
    resultado.contenido_comunidad.to_csv(
        rutas["contenido_comunidad"], index=False, encoding="utf-8-sig"
    )
    resultado.afirmaciones.to_csv(rutas["afirmaciones"], index=False, encoding="utf-8-sig")
    resultado.validacion.to_csv(rutas["validacion"], index=False, encoding="utf-8-sig")

    figura_general, _ = visualizar_sentimiento_general(resultado, rutas["figura_general"])
    plt.close(figura_general)
    figura_comunidades, _ = visualizar_sentimiento_comunidades(
        resultado, rutas["figura_comunidades"]
    )
    plt.close(figura_comunidades)

    rutas["metodologia"].write_text(_texto_resultados(resultado), encoding="utf-8")
    return rutas


def ejecutar(raiz: Path) -> ResultadoSentimiento:
    """Regenera el análisis de contenido y sentimiento desde los CSV originales."""

    raiz = Path(raiz).resolve()
    comentarios = pd.read_csv(raiz / "data" / "youtube_comments.csv", encoding="utf-8-sig")
    videos = pd.read_csv(raiz / "data" / "youtube_videos.csv", encoding="utf-8-sig")

    bipartita = construir_red_bipartita(comentarios, videos)
    proyecciones = analizar_proyecciones(bipartita)
    comunidades = analizar_comunidades_y_centralidad(
        bipartita, proyecciones, comentarios, semilla=42
    )

    resultado = analizar_contenido_y_sentimiento(comentarios, videos, comunidades)
    exportar_resultados(resultado, raiz / "resultados")
    return resultado


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sentimiento por comentario, contenido por comunidad y conclusiones."
    )
    parser.add_argument(
        "--raiz",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Ruta base del repositorio",
    )
    argumentos = parser.parse_args()
    resultado = ejecutar(argumentos.raiz)

    print("=== RESUMEN GENERAL DE SENTIMIENTO ===")
    print(resultado.resumen_general.to_string(index=False))
    print("\n=== SENTIMIENTO POR VIDEO (MUESTRA SUFICIENTE) ===")
    print(
        resultado.por_video[resultado.por_video["muestra_suficiente"]][
            ["video", "comentarios", "sentimiento_medio",
             "porcentaje_negativos", "porcentaje_positivos"]
        ].to_string(index=False)
    )
    print("\n=== CONTENIDO Y SENTIMIENTO POR COMUNIDAD ===")
    print(
        resultado.contenido_comunidad[
            ["comunidad_id", "nombre_comunidad", "comentarios", "autores_unicos",
             "sentimiento_medio", "porcentaje_negativos", "muestra_suficiente"]
        ].to_string(index=False)
    )
    print("\n=== VALIDACIÓN DE CONSISTENCIA ===")
    print(resultado.validacion.to_string(index=False))


if __name__ == "__main__":
    main()
