"""
Ilustración del paso 1 de /modelo-original: un comité de modelos y un paciente.

Usa el lenguaje de la portada (assets/portada.html): la misma proyección
(``P(x, y, z) = (x·HW, y·HW − z·ZH)``, con baldosas de 32 px y 34 px por unidad
de altura), baldosas en damero, mostrador en bloque con cara frontal y
superior, personas de cabeza redonda y torso, y la paleta del proyecto. Los
colores no están en el SVG: son clases que ``assets/style.css`` define con
variables, así que la ilustración cambia sola entre modo claro y oscuro,
como la portada.

Es estática. Solo tiene una entrada de unos 250 ms por elemento (CSS, una vez,
sin bucle) que ``prefers-reduced-motion`` desactiva. Los pesos de las barras
son ilustrativos: no representan datos reales.
"""

from __future__ import annotations

HW, ZH = 32, 34   # baldosa y altura por unidad, como en la portada
ANCHO, ALTO = 640, 320


def P(x: float, y: float, z: float = 0.0) -> tuple[float, float]:
    """Proyección de la portada: vista casi cenital con leve volumen."""
    return x * HW, y * HW - z * ZH


def _persona(x: float, y: float, clase_camisa: str, entra: int, atras: bool = True) -> str:
    """Cabeza, torso y, si no está tapada por el mostrador, las piernas."""
    cx, cy = P(x, y)
    piernas = "" if atras else (f'<rect class="il-pantalon" x="{cx - 9:.1f}" y="{cy + 8:.1f}" width="7" height="20" rx="3"/>'
                                f'<rect class="il-pantalon" x="{cx + 2:.1f}" y="{cy + 8:.1f}" width="7" height="20" rx="3"/>')
    return (f'<g class="il-entra" style="--i:{entra}">{piernas}'
            f'<rect class="{clase_camisa}" x="{cx - 14:.1f}" y="{cy - 14:.1f}" width="28" height="28" rx="12"/>'
            f'<circle class="il-piel" cx="{cx:.1f}" cy="{cy - 26:.1f}" r="11"/>'
            f'<path class="il-pelo" d="M{cx - 11:.1f} {cy - 28:.1f}a11 11 0 0 1 22 0z"/></g>')


def svg_comite(modelos: list[str] | None = None) -> str:
    """El SVG del comité (una sola línea: se incrusta en un Markdown con HTML)."""
    modelos = modelos or ["Árbol", "Random Forest", "XGBoost", "SVM"]
    pesos = [0.16, 0.26, 0.46, 0.12][:len(modelos)]            # ilustrativos
    xs = [2.6, 5.4, 8.2, 11.0][:len(modelos)]
    partes = [f'<svg class="il-comite" viewBox="0 0 {ANCHO} {ALTO}" role="img" '
              f'aria-label="Ilustración: un comité de cuatro modelos sentados tras un mostrador y un paciente que llega; '
              f'las barras del mostrador indican cuánto se confía en cada modelo para ese paciente.">',
              '<defs><pattern id="il-piso" width="64" height="64" patternUnits="userSpaceOnUse">'
              '<rect class="il-piso-a" width="64" height="64"/>'
              '<rect class="il-piso-b" width="32" height="32"/><rect class="il-piso-b" x="32" y="32" width="32" height="32"/>'
              '</pattern></defs>',
              f'<rect width="{ANCHO}" height="{ALTO}" rx="14" fill="url(#il-piso)"/>']
    # Personas del comité, detrás del mostrador.
    for i, x in enumerate(xs):
        partes.append(_persona(x, 3.25, "il-camisa-enfermera", i))
    # Mostrador: cara superior y frontal (la lateral es degenerada en esta proyección).
    x0, y0 = P(0.9, 3.806, 0.7)
    x1, y1 = P(12.4, 4.619, 0.7)
    _, yf = P(0, 4.619, 0)
    partes.append(f'<g class="il-entra" style="--i:{len(xs)}">'
                  f'<rect class="il-mesa-sup" x="{x0:.1f}" y="{y0:.1f}" width="{x1 - x0:.1f}" height="{y1 - y0:.1f}" rx="4"/>'
                  f'<rect class="il-mesa-fre" x="{x0:.1f}" y="{y1:.1f}" width="{x1 - x0:.1f}" height="{yf - y1:.1f}" rx="4"/></g>')
    # Barra de confianza de cada modelo sobre el mostrador y su placa en la cara frontal.
    for i, (x, nombre, w) in enumerate(zip(xs, modelos, pesos)):
        cx, _ = P(x, 0)
        largo = 76
        partes.append(f'<g class="il-entra" style="--i:{len(xs) + 1 + i}">'
                      f'<rect class="il-pista" x="{cx - largo / 2:.1f}" y="{y0 + 6:.1f}" width="{largo}" height="10" rx="5"/>'
                      f'<rect class="il-barra" x="{cx - largo / 2:.1f}" y="{y0 + 6:.1f}" width="{largo * w * 1.9:.1f}" height="10" rx="5"/>'
                      f'<rect class="il-placa" x="{cx - 44:.1f}" y="{y1 + 4:.1f}" width="88" height="19" rx="9"/>'
                      f'<text class="il-placa-texto" x="{cx:.1f}" y="{y1 + 17.5:.1f}" text-anchor="middle">{nombre}</text></g>')
    # Región de competencia alrededor del paciente: vecinos con y sin readmisión.
    px, py = P(16.6, 6.6)
    import math
    anillo = []
    for j in range(12):
        ang = 2 * math.pi * j / 12 + 0.25
        r = 62 + (9 if j % 3 == 0 else 0)
        cls = "il-vecino-1" if j % 2 else "il-vecino-0"
        anillo.append(f'<circle class="{cls}" cx="{px + r * math.cos(ang):.1f}" cy="{py + 0.62 * r * math.sin(ang):.1f}" r="5"/>')
    n = len(xs) * 2 + 1
    partes.append(f'<g class="il-entra" style="--i:{n}"><ellipse class="il-region" cx="{px:.1f}" cy="{py:.1f}" rx="78" ry="52"/>'
                  + "".join(anillo) + '</g>')
    # Camino del paciente al comité (discontinuo) y el paciente.
    partes.append(f'<path class="il-camino il-entra" style="--i:{n + 1}" d="M{px - 18:.1f} {py - 30:.1f}C{px - 90:.1f} {py - 70:.1f} 300 150 262 {yf + 8:.1f}"/>')
    partes.append(f'<g class="il-llega" style="--i:{n + 2}">' + _persona(16.6, 6.6, "il-camisa-paciente", 0, atras=False) + '</g>')
    # Rótulos.
    partes.append(f'<g class="il-entra" style="--i:{n + 3}">'
                  '<text class="il-rotulo" x="40" y="20">Comité de modelos</text>'
                  '<text class="il-rotulo-suave" x="40" y="36">cada barra: cuánto se confía en él</text>'
                  f'<text class="il-rotulo" x="{px:.1f}" y="{py + 82:.1f}" text-anchor="middle">Paciente que llega</text>'
                  f'<text class="il-rotulo-suave" x="{px:.1f}" y="{py + 99:.1f}" text-anchor="middle">y sus pacientes parecidos</text></g>')
    partes.append('</svg>')
    return "".join(partes)
