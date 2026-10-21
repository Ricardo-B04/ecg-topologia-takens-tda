"""Primer ejercicio: un latido MIT-BIH normal y su embedding de Takens en 3D."""

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_CSV = ROOT / "data" / "mitbih_train.csv"
DEFAULT_HTML = ROOT / "01_takens_ecg.html"


def leer_latido(path: Path, fila: int) -> list[float]:
    if fila < 0:
        raise ValueError("El índice de fila debe ser no negativo.")
    with path.open(newline="", encoding="utf-8") as archivo:
        lector = csv.reader(archivo)
        for indice, valores in enumerate(lector):
            if indice == fila:
                if len(valores) != 188:
                    raise ValueError(f"Se esperaban 188 columnas; se encontraron {len(valores)}.")
                numeros = [float(valor) for valor in valores]
                if numeros[-1] != 0:
                    raise ValueError(f"La fila {fila} es clase {numeros[-1]:g}, no clase 0.")
                return numeros[:-1]
    raise IndexError(f"El archivo no contiene la fila {fila}.")


def takens_3d(senal: list[float], retardo: int = 3) -> list[list[float]]:
    if retardo < 1 or len(senal) <= 2 * retardo:
        raise ValueError("La señal es demasiado corta para este retardo.")
    return [[senal[i], senal[i + retardo], senal[i + 2 * retardo]]
            for i in range(len(senal) - 2 * retardo)]


def generar_html(senal: list[float], puntos: list[list[float]], destino: Path) -> None:
    datos = json.dumps({"senal": senal, "puntos": puntos}, separators=(",", ":"))
    html = """<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ECG normal · Takens 3D</title>
  <script src="https://cdn.plot.ly/plotly-4.1.2.min.js" charset="utf-8"></script>
  <style>
    :root { color-scheme: light; font-family: system-ui, sans-serif; }
    body { max-width: 1180px; margin: 0 auto; padding: 28px 20px; color: #172033; background: #f7f9fc; }
    h1 { font-size: 1.5rem; margin: 0 0 6px; }
    p { margin: 0 0 18px; color: #526174; }
    .panel { background: white; border: 1px solid #dbe3ed; border-radius: 14px; padding: 16px; margin-bottom: 16px; }
    .plot { width: 100%; height: 340px; }
    #takens { height: 570px; }
    .caption { font-size: .9rem; color: #536176; line-height: 1.45; }
    @media (max-width: 700px) { #takens { height: 440px; } }
  </style>
</head>
<body>
  <h1>Un latido normal: señal 1D y reconstrucción 3D</h1>
  <p>MIT-BIH, clase 0 · dimensión 3 · retardo 3 muestras · 187 muestras → 181 puntos</p>
  <section class="panel"><div id="senal" class="plot"></div></section>
  <section class="panel"><div id="takens" class="plot"></div>
    <div class="caption">Arrastra para rotar, usa la rueda para acercar y coloca el cursor sobre la trayectoria para ver el índice temporal. La trayectoria de un latido aislado no tiene que cerrar un ciclo.</div>
  </section>
  <script>
    const datos = __PAYLOAD__;
    const valores = datos.senal;
    const puntos = datos.puntos;
    const indices = puntos.map((_, i) => i);
    const x = puntos.map(p => p[0]);
    const y = puntos.map(p => p[1]);
    const z = puntos.map(p => p[2]);
    const config = {responsive: true, displaylogo: false, scrollZoom: true};
    const base = {paper_bgcolor: '#fff', plot_bgcolor: '#fff', margin: {l: 60, r: 25, t: 35, b: 50}, font: {color: '#26364a'}};

    Plotly.newPlot('senal', [{type: 'scatter', mode: 'lines', x: valores.map((_, i) => i), y: valores,
      line: {color: '#2456a6', width: 2.5}, hovertemplate: 'muestra %{x}<br>amplitud %{y:.3f}<extra></extra>'}],
      {...base, title: {text: 'Latido original'}, xaxis: {title: 'muestra'}, yaxis: {title: 'amplitud normalizada'}}, config);

    Plotly.newPlot('takens', [
      {type: 'scatter3d', mode: 'lines', x, y, z, customdata: indices,
       line: {color: indices, colorscale: 'Viridis', width: 6},
       hovertemplate: 'i = %{customdata}<br>x[i] = %{x:.3f}<br>x[i+3] = %{y:.3f}<br>x[i+6] = %{z:.3f}<extra></extra>',
       showlegend: false},
      {type: 'scatter3d', mode: 'markers', x: [x[0], x.at(-1)], y: [y[0], y.at(-1)], z: [z[0], z.at(-1)],
       text: ['inicio', 'fin'], marker: {size: 5, color: ['#12a874', '#e05b50']},
       hovertemplate: '%{text}<extra></extra>', showlegend: false}
    ], {...base, title: {text: 'Embedding de Takens'}, margin: {l: 0, r: 0, t: 40, b: 0},
        scene: {xaxis: {title: 'x[i]'}, yaxis: {title: 'x[i+3]'}, zaxis: {title: 'x[i+6]'}, aspectmode: 'cube'}}, config);
  </script>
</body>
</html>
"""
    destino.write_text(html.replace("__PAYLOAD__", datos), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="CSV MIT-BIH sin encabezado")
    parser.add_argument("--fila", type=int, default=29, help="índice de fila, comenzando en 0")
    parser.add_argument("--salida", type=Path, default=DEFAULT_HTML, help="archivo HTML de salida")
    args = parser.parse_args()
    senal = leer_latido(args.csv, args.fila)
    puntos = takens_3d(senal)
    generar_html(senal, puntos, args.salida)
    print(f"Clase 0: {len(senal)} muestras → {len(puntos)} puntos; HTML: {args.salida}")


if __name__ == "__main__":
    main()
