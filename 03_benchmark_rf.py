"""Compara Random Forest con ECG crudo e imágenes de persistencia H0/H1."""

import argparse
import csv
import importlib
import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from gtda.diagrams import PersistenceImage
from gtda.homology import VietorisRipsPersistence
from gtda.time_series import SingleTakensEmbedding
from sklearn import __version__ as sklearn_version
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score)


ROOT = Path(__file__).resolve().parent
CLASES = (0, 2)


def proyectar_takens(X):
    takens = SingleTakensEmbedding(parameters_type="fixed", time_delay=3,
                                  dimension=3, stride=1)
    nubes = np.stack([takens.fit_transform(senal) for senal in X])
    if nubes.shape[1:] != (181, 3):
        raise AssertionError(f"Forma inesperada de Takens: {nubes.shape}")
    return nubes


def evaluar(y, pred):
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1_macro": float(f1_score(y, pred, labels=CLASES, average="macro")),
        "matriz_confusion": confusion_matrix(y, pred, labels=CLASES).tolist(),
        "reporte": classification_report(y, pred, labels=CLASES,
                                         target_names=["normal", "ventricular"],
                                         output_dict=True, zero_division=0),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entrenamiento", type=Path,
                        default=ROOT / "resultados/02_vectorizacion/representaciones.npz")
    parser.add_argument("--prueba", type=Path, default=ROOT / "data/mitbih_test.csv")
    parser.add_argument("--por-clase", type=int, default=300)
    parser.add_argument("--semilla-prueba", type=int, default=43)
    parser.add_argument("--arboles", type=int, default=300)
    parser.add_argument("--trabajadores", type=int, default=2)
    parser.add_argument("--salida", type=Path, default=ROOT / "resultados/03_benchmark")
    args = parser.parse_args()
    for path in (args.entrenamiento, args.prueba):
        if not path.is_file():
            parser.error(f"Falta el archivo: {path}")
    if min(args.por_clase, args.arboles, args.trabajadores) < 1:
        parser.error("--por-clase, --arboles y --trabajadores deben ser positivos")

    with np.load(args.entrenamiento) as datos:
        X_train = datos["X_crudo"]
        y_train = datos["y"]
        nubes_train = datos["nubes"]
        diagramas_train = datos["diagramas"]
        X_topologico_anterior = datos["X_topologico"]
    if X_train.shape[1] != 187 or nubes_train.shape[1:] != (181, 3):
        raise ValueError("Las representaciones de entrenamiento no usan 187 muestras y Takens 3D")
    if set(np.unique(y_train)) != set(CLASES):
        raise ValueError(f"Se esperaban clases {CLASES} en entrenamiento")

    # El muestreador ya usado en el paso 2 lee el CSV sin cargarlo entero.
    muestreador = importlib.import_module("02_vectorizacion_topologica").muestra_balanceada
    X_test, y_test, filas_test, vistos = muestreador(
        args.prueba, args.por_clase, args.semilla_prueba
    )
    print(f"Prueba: {len(X_test)} latidos ({dict(Counter(y_test.tolist()))}); "
          f"archivo: {dict(vistos)}", flush=True)

    nubes_test = proyectar_takens(X_test)
    vr = VietorisRipsPersistence(homology_dimensions=(0, 1),
                                 collapse_edges=True, n_jobs=args.trabajadores)
    vr.fit(nubes_train)
    diagramas_test = vr.transform(nubes_test)
    pi = PersistenceImage(sigma=0.05, n_bins=20, weight_function=np.asarray,
                          n_jobs=args.trabajadores).fit(diagramas_train)
    X_topo_train = pi.transform(diagramas_train).reshape(len(X_train), -1)
    X_topo_test = pi.transform(diagramas_test).reshape(len(X_test), -1)
    if X_topo_train.shape[1] != 800 or X_topo_test.shape[1] != 800:
        raise AssertionError("La imagen topológica debe tener 800 variables")
    diferencia_previa = float(np.max(np.abs(X_topo_train - X_topologico_anterior)))
    if diferencia_previa > 1e-3:
        raise ValueError(f"La nueva transformación no coincide con el paso 2: {diferencia_previa}")
    if not np.isfinite(X_topo_test).all():
        raise ValueError("La prueba topológica contiene valores no finitos")

    modelo_crudo = RandomForestClassifier(n_estimators=args.arboles,
                                          random_state=42, n_jobs=args.trabajadores)
    modelo_topologico = RandomForestClassifier(n_estimators=args.arboles,
                                               random_state=42, n_jobs=args.trabajadores)
    modelo_crudo.fit(X_train, y_train)
    modelo_topologico.fit(X_topo_train, y_train)
    pred_crudo = modelo_crudo.predict(X_test)
    pred_topologico = modelo_topologico.predict(X_topo_test)
    prob_crudo = modelo_crudo.predict_proba(X_test)[:, list(modelo_crudo.classes_).index(2)]
    prob_topologico = modelo_topologico.predict_proba(X_topo_test)[:, list(modelo_topologico.classes_).index(2)]

    args.salida.mkdir(parents=True, exist_ok=True)
    metricas = {
        "clases_orden_matriz": list(CLASES),
        "n_entrenamiento": int(len(y_train)),
        "n_prueba": int(len(y_test)),
        "entrenamiento_por_clase": dict(Counter(map(int, y_train))),
        "prueba_por_clase": dict(Counter(map(int, y_test))),
        "archivo_entrenamiento": str(args.entrenamiento.resolve()),
        "archivo_prueba": str(args.prueba.resolve()),
        "semilla_prueba": args.semilla_prueba,
        "random_forest": {"n_estimators": args.arboles, "random_state": 42},
        "sklearn": sklearn_version,
        "diferencia_maxima_imagen_paso_2": diferencia_previa,
        "crudo": evaluar(y_test, pred_crudo),
        "topologico": evaluar(y_test, pred_topologico),
    }
    (args.salida / "metricas.json").write_text(
        json.dumps(metricas, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (args.salida / "predicciones.csv").open("w", newline="", encoding="utf-8") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(["fila_prueba", "clase_real", "pred_crudo", "prob_ventricular_crudo",
                          "pred_topologico", "prob_ventricular_topologico"])
        for fila in zip(filas_test, y_test, pred_crudo, prob_crudo,
                        pred_topologico, prob_topologico):
            escritor.writerow(fila)
    np.savez_compressed(args.salida / "prueba.npz", X_crudo=X_test, y=y_test,
                        filas_origen=filas_test, X_topologico=X_topo_test.astype(np.float32))
    joblib.dump(modelo_crudo, args.salida / "modelo_crudo.joblib")
    joblib.dump(modelo_topologico, args.salida / "modelo_topologico.joblib")
    joblib.dump(vr, args.salida / "vietoris_rips.joblib")
    joblib.dump(pi, args.salida / "persistence_image.joblib")
    for nombre in ("crudo", "topologico"):
        m = metricas[nombre]
        print(f"{nombre}: accuracy={m['accuracy']:.3f}, "
              f"balanced_accuracy={m['balanced_accuracy']:.3f}, "
              f"F1 macro={m['f1_macro']:.3f}; matriz={m['matriz_confusion']}")
    print(f"Guardado en {args.salida.resolve()}")


if __name__ == "__main__":
    main()
