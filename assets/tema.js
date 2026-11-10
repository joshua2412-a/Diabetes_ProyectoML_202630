/* Botón claro / oscuro. El atributo inicial lo pone el <head> de app.py.
   Sigue al sistema mientras el usuario no haya elegido. */
(function () {
  "use strict";
  function aplicar(tema, guardar) {
    var raiz = document.documentElement;
    raiz.classList.add("tema-cambiando");          // sin transiciones durante el cambio
    raiz.setAttribute("data-tema", tema);
    if (guardar) { try { localStorage.setItem("tema", tema); } catch (e) {} }
    var b = document.getElementById("boton-tema");
    if (b) b.setAttribute("aria-pressed", tema === "oscuro" ? "true" : "false");
    requestAnimationFrame(function () { requestAnimationFrame(function () {
      raiz.classList.remove("tema-cambiando");
    }); });
  }
  document.addEventListener("click", function (e) {
    if (!e.target.closest || !e.target.closest("#boton-tema")) return;
    var actual = document.documentElement.getAttribute("data-tema");
    aplicar(actual === "oscuro" ? "claro" : "oscuro", true);
  });
  // Al cambiar de página Dash usa history.pushState y conserva el scroll de
  // la página anterior: la nueva se abriría a media altura, sin su cabecera.
  var empujar = history.pushState;
  history.pushState = function () {
    var r = empujar.apply(this, arguments);
    window.scrollTo({ top: 0, behavior: "instant" });
    return r;
  };
  if (window.matchMedia) {
    matchMedia("(prefers-color-scheme: dark)").addEventListener("change", function (e) {
      var elegido = null; try { elegido = localStorage.getItem("tema"); } catch (x) {}
      if (!elegido) aplicar(e.matches ? "oscuro" : "claro", false);
    });
  }
})();
