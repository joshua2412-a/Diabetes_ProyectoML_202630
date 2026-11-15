/* ==========================================================================
   Interacción común a todos los gráficos del dashboard.

   1. Panel flotante: sustituye al cuadro de Plotly por un panel con el
      estilo del tablero que sigue al cursor con un suavizado (interpolación
      en cada fotograma), aparece y desaparece con una transición corta.
      El contenido es el MISMO texto que Plotly arma con cada hovertemplate:
      se lee de su capa de hover (que queda invisible), así que no hay que
      duplicar textos en Python.
   2. Foco: al pasar sobre una barra o un punto, ese elemento se ilumina y
      el resto de su gráfico se atenúa un poco.
   3. Cifras que cuentan: los números grandes (KPI) suben desde cero cuando
      aparecen, conservando su formato (coma decimal, punto de millar, «%»).
      Solo al llegar a una página: si la cifra cambia porque el usuario movió
      un control (deslizador, selector, teclado), se muestra de inmediato.
      Contar desde cero en cada movimiento del simulador era ruido.

   Todo respeta «reducir movimiento»: sin suavizado ni conteo en ese caso.
   ========================================================================== */
(function () {
  "use strict";
  var reducido = window.matchMedia && matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* ---------- 1. Panel flotante ---------------------------------------- */
  var panel = document.createElement("div");
  panel.className = "tip-flotante";
  panel.setAttribute("role", "tooltip");
  panel.setAttribute("aria-hidden", "true");
  document.addEventListener("DOMContentLoaded", function () { document.body.appendChild(panel); });
  if (document.body) document.body.appendChild(panel);

  var objetivoX = 0, objetivoY = 0, x = 0, y = 0, visible = false, fotograma = null, ultimoMouse = null;

  function animar() {
    var k = reducido ? 1 : 0.24;                 // fracción del camino que recorre en cada fotograma
    x += (objetivoX - x) * k;
    y += (objetivoY - y) * k;
    panel.style.transform = "translate3d(" + x.toFixed(1) + "px," + y.toFixed(1) + "px,0)";
    if (Math.abs(objetivoX - x) > 0.3 || Math.abs(objetivoY - y) > 0.3) fotograma = requestAnimationFrame(animar);
    else fotograma = null;
  }

  function colocar(cx, cy, saltar) {
    var w = panel.offsetWidth, h = panel.offsetHeight, m = 16;
    var nx = cx + m, ny = cy + m;
    if (nx + w > window.innerWidth - 8) nx = cx - w - m;      // no salirse por la derecha
    if (ny + h > window.innerHeight - 8) ny = cy - h - m;     // ni por abajo
    objetivoX = Math.max(8, nx); objetivoY = Math.max(8, ny);
    if (saltar) { x = objetivoX; y = objetivoY; }
    if (!fotograma) fotograma = requestAnimationFrame(animar);
  }

  document.addEventListener("mousemove", function (e) {
    ultimoMouse = e;
    if (visible) colocar(e.clientX, e.clientY, false);
  }, { passive: true });

  // Color del elemento señalado (la caja de Plotly es blanca para todos).
  function colorDe(p) {
    var fd = p.fullData || p.data || {};
    var m = fd.marker || {}, c = m.color;
    if (Array.isArray(c)) c = c[p.pointNumber];
    if (typeof c !== "string") c = (fd.line && fd.line.color) || null;
    return typeof c === "string" ? c : null;
  }

  // Texto de la capa de hover de Plotly → bloques del panel.
  function leerHover(gd, puntos) {
    var bloques = [];
    gd.querySelectorAll(".hoverlayer .hovertext").forEach(function (g, i) {
      var color = puntos && puntos[i] ? colorDe(puntos[i]) : null;
      var lineas = [];
      var texto = g.querySelector("text.nums");
      if (texto) {
        var tspans = texto.querySelectorAll("tspan.line");
        if (tspans.length) tspans.forEach(function (t) { lineas.push(t.innerHTML); });
        else lineas.push(texto.innerHTML);
      }
      var extra = g.querySelector("text.name");
      if (extra && extra.textContent.trim()) lineas.push('<span class="tip-extra">' + extra.innerHTML + "</span>");
      if (lineas.length) bloques.push({ color: color, lineas: lineas });
    });
    return bloques;
  }

  function pintarPanel(bloques) {
    // tspan con font-weight bold → <strong>; el resto, texto plano.
    panel.innerHTML = bloques.map(function (b) {
      var cuerpo = b.lineas.map(function (l, i) {
        var html = l.replace(/<tspan[^>]*font-weight:\s*bold[^>]*>(.*?)<\/tspan>/g, "<strong>$1</strong>")
                    .replace(/<\/?tspan[^>]*>/g, "");
        return '<div class="tip-linea' + (i === 0 ? " tip-primera" : "") + '">' + html + "</div>";
      }).join("");
      var marca = b.color && b.color !== "none" ? '<span class="tip-marca" style="background:' + b.color + '"></span>' : "";
      return '<div class="tip-bloque">' + marca + '<div class="tip-cuerpo">' + cuerpo + "</div></div>";
    }).join("");
  }

  /* ---------- 2. Foco en el elemento bajo el cursor -------------------- */
  function trazaDom(gd, indice) {
    var trazas = gd.querySelectorAll(".trace");
    for (var i = 0; i < trazas.length; i++) {
      var d = trazas[i].__data__;
      if (d && d[0] && d[0].trace && d[0].trace.index === indice) return trazas[i];
    }
    return null;
  }

  function enfocar(gd, puntos) {
    quitarFoco(gd);
    var alguno = false;
    (puntos || []).forEach(function (p) {
      var tipo = p.data && p.data.type;
      if (tipo !== "bar" && tipo !== "scatter" && tipo !== "scattergl") return;
      var t = trazaDom(gd, p.curveNumber);
      if (!t) return;
      var elementos = t.querySelectorAll(".point");
      var el = elementos[p.pointNumber];
      if (el) { el.classList.add("punto-activo"); alguno = true; }
    });
    if (alguno) gd.classList.add("con-foco");
  }

  function quitarFoco(gd) {
    gd.classList.remove("con-foco");
    gd.querySelectorAll(".punto-activo").forEach(function (e) { e.classList.remove("punto-activo"); });
  }

  /* ---------- Conexión con cada gráfico -------------------------------- */
  function conectar(gd) {
    if (gd.__interaccion || typeof gd.on !== "function") return;
    gd.__interaccion = true;
    gd.classList.add("tip-propio");
    gd.on("plotly_hover", function (ev) {
      // La capa de hover se dibuja en el mismo ciclo: se lee en el siguiente fotograma.
      requestAnimationFrame(function () {
        var bloques = leerHover(gd, ev.points);
        if (!bloques.length) return;
        pintarPanel(bloques);
        var e = ev.event || ultimoMouse;
        var aparecer = !visible;
        visible = true;
        panel.classList.add("visible");
        if (e) colocar(e.clientX, e.clientY, aparecer);
        enfocar(gd, ev.points);
      });
    });
    gd.on("plotly_unhover", function () {
      visible = false;
      panel.classList.remove("visible");
      quitarFoco(gd);
    });
  }

  /* ---------- 3. Cifras que cuentan ------------------------------------ */
  // Última vez que el usuario tocó un control. Los enlaces de navegación no
  // cuentan: al llegar a una página nueva las cifras sí deben contar.
  var ultimoControl = -Infinity, VENTANA_CONTROL = 1500;
  function marcarControl(e) {
    var t = e.target;
    if (t && t.closest && t.closest("a[href], .nav-link")) return;
    ultimoControl = performance.now();
  }
  document.addEventListener("pointerdown", marcarControl, true);
  document.addEventListener("keydown", marcarControl, true);

  var FORMATO = /^([\d.]+)(,\d+)?(\s?%)?$/;   // 11,4 % · 79.473 · 4,1

  function contar(el) {
    // Mientras cuenta, su propio texto cambia en cada fotograma: sin esta
    // guarda, el observador lo tomaría por un valor nuevo y volvería a empezar.
    if (reducido || el.__animando || el.__contado === el.textContent) return;
    var original = el.textContent.trim();
    if (performance.now() - ultimoControl < VENTANA_CONTROL) { el.__contado = original; return; }
    var m = original.match(FORMATO);
    if (!m) { el.__contado = original; return; }
    var decimales = m[2] ? m[2].length - 1 : 0;
    var valor = parseFloat(m[1].replace(/\./g, "") + (m[2] ? "." + m[2].slice(1) : ""));
    var sufijo = m[3] || "";
    var miles = m[1].indexOf(".") >= 0 || valor >= 1000;
    var inicio = null, duracion = 600;
    el.__contado = original;
    el.__animando = true;
    function formato(v) {
      var partes = v.toFixed(decimales).split(".");
      var entero = miles ? partes[0].replace(/\B(?=(\d{3})+(?!\d))/g, ".") : partes[0];
      return entero + (decimales ? "," + partes[1] : "") + sufijo;
    }
    function paso(ts) {
      if (inicio === null) inicio = ts;
      var t = Math.min(1, (ts - inicio) / duracion);
      var suave = 1 - Math.pow(1 - t, 3);
      el.textContent = t < 1 ? formato(valor * suave) : original;
      if (t < 1) requestAnimationFrame(paso);
      else el.__animando = false;
    }
    requestAnimationFrame(paso);
  }

  /* ---------- Observador: conecta lo que Dash monta después ------------ */
  function revisar(raiz) {
    (raiz.querySelectorAll ? raiz : document).querySelectorAll(".js-plotly-plot").forEach(conectar);
    (raiz.querySelectorAll ? raiz : document).querySelectorAll(".kpi-valor, .kpi-mini-valor, .gluco-numero, .cifra-grande:not(.sin-conteo)").forEach(contar);
  }
  var pendiente = false;
  new MutationObserver(function () {
    if (pendiente) return;
    pendiente = true;
    requestAnimationFrame(function () { pendiente = false; revisar(document); });
  }).observe(document.documentElement, { childList: true, subtree: true });
})();
