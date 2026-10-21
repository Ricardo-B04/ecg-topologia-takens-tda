"""Evalúa dos modelos ya entrenados ante desfases con ceros al inicio."""

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
import plotly.graph_objects as go
from gtda.time_series import SingleTakensEmbedding
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


ROOT = Path(__file__).resolve().parent
CLASES = (0, 2)


def desplazar_derecha(X: np.ndarray, muestras: int) -> np.ndarray:
    """Añade ceros al inicio y recorta la misma cantidad al final."""
    if not 0 <= muestras < X.shape[1]:
        raise ValueError("El desfase debe estar entre 0 y 186 muestras")
    desplazadas = np.zeros_like(X)
    if muestras == 0:
        desplazadas[:] = X
    else:
        desplazadas[:, muestras:] = X[:, :-muestras]
    return desplazadas


def vector_topologico(X, vr, pi):
    takens = SingleTakensEmbedding(parameters_type="fixed", time_delay=3,
                                  dimension=3, stride=1)
    nubes = np.stack([takens.fit_transform(senal) for senal in X])
    diagramas = vr.transform(nubes)
    return pi.transform(diagramas).reshape(len(X), -1)


def metricas(y, pred, pred_base):
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "f1_macro": float(f1_score(y, pred, labels=CLASES, average="macro")),
        "acuerdo_con_prediccion_original": float(np.mean(pred == pred_base)),
        "matriz_confusion": confusion_matrix(y, pred, labels=CLASES).tolist(),
    }


def grafico(resultados, salida):
    figura = go.Figure()
    for clave, nombre, color in (
        ("crudo", "ECG crudo", "#275dad"),
        ("topologico", "Imagen de persistencia", "#e07132"),
    ):
        figura.add_trace(go.Scatter(
            x=[r["desfase_muestras"] for r in resultados],
            y=[100 * r[clave]["accuracy"] for r in resultados],
            mode="lines+markers", name=nombre,
            line={"color": color, "width": 3}, marker={"size": 9},
            hovertemplate="Desfase: %{x} muestras<br>Exactitud: %{y:.1f}%<extra>%{fullData.name}</extra>",
        ))
    figura.update_layout(
        title="Sensibilidad a desfases con ceros al inicio",
        xaxis_title="Desfase hacia la derecha (muestras)",
        yaxis_title="Exactitud en 600 latidos (%)",
        yaxis={"range": [0, 100]},
        template="plotly_white", hovermode="x unified",
    )
    figura.write_html(salida, include_plotlyjs=True, full_html=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path,
                        default=ROOT / "resultados/03_benchmark")
    parser.add_argument("--desfases", type=int, nargs="+", default=[0, 1, 3, 6, 12, 24],
                        help="Número de muestras desplazadas a la derecha")
    parser.add_argument("--salida", type=Path,
                        default=ROOT / "resultados/04_invarianza_temporal")
    args = parser.parse_args()
    if len(set(args.desfases)) != len(args.desfases) or 0 not in args.desfases:
        parser.error("--desfases debe incluir 0 y no tener valores repetidos")
    if any(not 0 <= k < 187 for k in args.desfases):
        parser.error("Cada desfase debe estar entre 0 y 186")
    desfases = sorted(args.desfases)

    with np.load(args.benchmark / "prueba.npz") as datos:
        X = datos["X_crudo"]
        y = datos["y"]
        filas = datos["filas_origen"]
        X_topo_base = datos["X_topologico"]
    if X.shape != (600, 187) or X_topo_base.shape != (600, 800):
        raise ValueError("El benchmark esperado debe contener 600 latidos y 800 variables topológicas")
    modelo_crudo = joblib.load(args.benchmark / "modelo_crudo.joblib")
    modelo_topologico = joblib.load(args.benchmark / "modelo_topologico.joblib")
    vr = joblib.load(args.benchmark / "vietoris_rips.joblib")
    pi = joblib.load(args.benchmark / "persistence_image.joblib")
    benchmark = json.loads((args.benchmark / "metricas.json").read_text(encoding="utf-8"))

    base_crudo = modelo_crudo.predict(X)
    base_topologico = modelo_topologico.predict(X_topo_base)
    if abs(accuracy_score(y, base_crudo) - benchmark["crudo"]["accuracy"]) > 1e-12:
        raise AssertionError("El modelo crudo no reproduce el benchmark")
    if abs(accuracy_score(y, base_topologico) - benchmark["topologico"]["accuracy"]) > 1e-12:
        raise AssertionError("El modelo topológico no reproduce el benchmark")

    resultados = []
    predicciones = []
    for k in desfases:
        X_desplazado = desplazar_derecha(X, k)
        X_topologico = X_topo_base if k == 0 else vector_topologico(X_desplazado, vr, pi)
        if not np.isfinite(X_topologico).all():
            raise ValueError(f"Valores topológicos no finitos con desfase {k}")
        pred_crudo = modelo_crudo.predict(X_desplazado)
        pred_topologico = modelo_topologico.predict(X_topologico)
        fila = {
            "desfase_muestras": k,
            "crudo": metricas(y, pred_crudo, base_crudo),
            "topologico": metricas(y, pred_topologico, base_topologico),
        }
        resultados.append(fila)
        predicciones.extend(
            (k, int(f), int(real), int(pc), int(pt))
            for f, real, pc, pt in zip(filas, y, pred_crudo, pred_topologico)
        )
        print(f"{k:2d} muestras: crudo={fila['crudo']['accuracy']:.3f}; "
              f"topológico={fila['topologico']['accuracy']:.3f}", flush=True)

    args.salida.mkdir(parents=True, exist_ok=True)
    metadatos = {
        "protocolo": "desfase a la derecha con ceros al inicio y recorte al final",
        "numero_latidos": int(len(y)),
        "clases_orden_matriz": list(CLASES),
        "desfases_muestras": desfases,
        "modelos_reentrenados": False,
        "resultados": resultados,
    }
    (args.salida / "metricas.json").write_text(
        json.dumps(metadatos, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (args.salida / "comparacion.csv").open("w", newline="", encoding="utf-8") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(["desfase_muestras", "accuracy_crudo", "f1_macro_crudo",
                          "acuerdo_base_crudo", "accuracy_topologico", "f1_macro_topologico",
                          "acuerdo_base_topologico"])
        for r in resultados:
            escritor.writerow([r["desfase_muestras"], r["crudo"]["accuracy"],
                              r["crudo"]["f1_macro"],
                              r["crudo"]["acuerdo_con_prediccion_original"],
                              r["topologico"]["accuracy"],
                              r["topologico"]["f1_macro"],
                              r["topologico"]["acuerdo_con_prediccion_original"]])
    with (args.salida / "predicciones.csv").open("w", newline="", encoding="utf-8") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(["desfase_muestras", "fila_prueba", "clase_real",
                          "pred_crudo", "pred_topologico"])
        escritor.writerows(predicciones)
    grafico(resultados, args.salida / "exactitud_desfase.html")
    print(f"Guardado en {args.salida.resolve()}")


if __name__ == "__main__":
    main()
