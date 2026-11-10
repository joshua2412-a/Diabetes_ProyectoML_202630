"""Pruebas del formato de cifras del dashboard (src/tablero/formato.py)."""

from src.tablero.formato import lista_y, miles, num, p_valor, pct


def test_coma_decimal_y_punto_de_millar():
    assert num(1234.567, 2) == "1.234,57"
    assert miles(79473) == "79.473"
    assert pct(11.392) == "11,4 %"


def test_signo_y_nulos():
    assert num(-0.0023, 4, signo=True) == "-0,0023"
    assert num(0.5, 1, signo=True) == "+0,5"
    assert num(float("nan")) == "–"


def test_p_valor_y_listas():
    assert p_valor(0.00002) == "p < 0,0001"
    assert p_valor(0.0312) == "p = 0,0312"
    assert lista_y(["A", "B", "C"]) == "A, B y C"
    assert lista_y(["A"]) == "A"
