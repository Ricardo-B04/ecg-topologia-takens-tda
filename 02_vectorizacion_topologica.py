"""Muestra balanceada MIT-BIH y vectores topológicos para el benchmark."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
CLASES = (0, 2)  # N: normal; V: latido ectópico ventricular


def muestra_balanceada(csv_path: Path, por_clase: int, semilla: int):
    """Reservoir sampling: lee el CSV una vez sin cargarlo completo en RAM."""
    if por_clase < 1:
        raise ValueError("--por-clase debe ser positivo")
    rng = np.random.default_rng(semilla)
    reservorios = {clase: [] for clase in CLASES}
    vistos = Counter()

    with csv_path.open(newline="", encoding="utf-8") as archivo:
        for fila_origen, fila in enumerate(csv.reader(archivo)):
            if len(fila) != 188:
                raise ValueError(f"Fila {fila_origen}: se esperaban 188 columnas, hay {len(fila)}")
            try:
                clase_num = float(fila[-1])
            except ValueError as exc:
                raise ValueError("El CSV MIT-BIH debe leerse sin encabezado") from exc
            if not clase_num.is_integer():
                raise ValueError(f"Fila {fila_origen}: etiqueta no entera {clase_num}")
            clase = int(clase_num)
            if clase not in CLASES:
                continue
            vistos[clase] += 1
            senal = np.asarray(fila[:-1], dtype=np.float32)
            if not np.isfinite(senal).all():
                raise ValueError(f"Fila {fila_origen}: señal no finita")
            elemento = (fila_origen, senal, clase)
            deposito = reservorios[clase]
            if len(deposito) < por_clase:
                deposito.append(elemento)
            else:
                reemplazar = rng.integers(vistos[clase])
                if reemplazar < por_clase:
                    deposito[reemplazar] = elemento

    insuficientes = {c: vistos[c] for c in CLASES if vistos[c] < por_clase}
    if insuficientes:
        raise ValueError(f"No hay {por_clase} filas por clase: {insuficientes}")
    elementos = sum((reservorios[c] for c in CLASES), [])
    elementos = [elementos[i] for i in rng.permutation(len(elementos))]
    filas = np.asarray([e[0] for e in elementos], dtype=np.int64)
    X = np.stack([e[1] for e in elementos])
    y = np.asarray([e[2] for e in elementos], dtype=np.uint8)
    return X, y, filas, vistos


def vectorizar(X: np.ndarray, trabajadores: int):
    from gtda.diagrams import PersistenceImage
    from gtda.homology import VietorisRipsPersistence
    from gtda.time_series import SingleTakensEmbedding

    takens = SingleTakensEmbedding(
        parameters_type="fixed", time_delay=3, dimension=3, stride=1
    )
    nubes = np.stack([takens.fit_transform(senal) for senal in X])
    if nubes.shape[1:] != (181, 3):
        raise AssertionError(f"Forma inesperada de las nubes: {nubes.shape}")

    vr = VietorisRipsPersistence(
        homology_dimensions=(0, 1), collapse_edges=True, n_jobs=trabajadores
    )
    diagramas = vr.fit_transform(nubes)
    imagen = PersistenceImage(
        sigma=0.05, n_bins=20, weight_function=lambda persistencia: persistencia,
        n_jobs=trabajadores,
    ).fit_transform(diagramas)
    X_topologico = imagen.reshape(len(X), -1)
    return nubes, diagramas, imagen, X_topologico


def resumen_diagramas(diagramas: np.ndarray, y: np.ndarray, filas: np.ndarray, path: Path):
    nacimiento = diagramas[:, :, 0]
    muerte = diagramas[:, :, 1]
    es_h1 = (diagramas[:, :, 2] == 1) & (muerte > nacimiento)
    persistencias = np.where(es_h1, muerte - nacimiento, 0)
    with path.open("w", newline="", encoding="utf-8") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(["indice_muestra", "fila_origen", "clase", "n_h1", "persistencia_h1_max"])
        for i in range(len(y)):
            escritor.writerow([i, int(filas[i]), int(y[i]), int(es_h1[i].sum()),
                              float(persistencias[i].max())])


def distancia_bottleneck(diagramas: np.ndarray, y: np.ndarray, por_clase: int,
                         trabajadores: int, salida: Path):
    from gtda.diagrams import PairwiseDistance

    indices = np.concatenate([np.flatnonzero(y == clase)[:por_clase] for clase in CLASES])
    # delta=0 da la distancia exacta; infinito combina H0 y H1 por su máximo.
    D = PairwiseDistance(metric="bottleneck", metric_params={"delta": 0.0},
                         order=np.inf, n_jobs=trabajadores).fit_transform(diagramas[indices])
    np.savez_compressed(salida, distancias=D.astype(np.float32),
                        indices_muestra=indices, clases=y[indices])
    return D.shape


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=ROOT / "data" / "mitbih_train.csv")
    parser.add_argument("--por-clase", type=int, default=300)
    parser.add_argument("--semilla", type=int, default=42)
    parser.add_argument("--trabajadores", type=int, default=2)
    parser.add_argument("--bottleneck-por-clase", type=int, default=0,
                        help="0 omite la matriz; N calcula una matriz exacta de 2N latidos")
    parser.add_argument("--salida", type=Path, default=ROOT / "resultados" / "02_vectorizacion")
    args = parser.parse_args()
    if not args.csv.is_file():
        parser.error(f"Falta el CSV: {args.csv}")
    if args.trabajadores < 1:
        parser.error("--trabajadores debe ser positivo")
    if not 0 <= args.bottleneck_por_clase <= args.por_clase:
        parser.error("--bottleneck-por-clase debe estar entre 0 y --por-clase")

    try:
        import gtda
    except ImportError as exc:
        parser.error("Falta giotto-tda. Instala requirements-tda.txt")
        raise exc

    X, y, filas, vistos = muestra_balanceada(args.csv, args.por_clase, args.semilla)
    print(f"Muestra: {len(X)} latidos ({Counter(y.tolist())}); origen: {dict(vistos)}", flush=True)
    nubes, diagramas, imagen, X_topologico = vectorizar(X, args.trabajadores)
    args.salida.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.salida / "representaciones.npz",
                        X_crudo=X, y=y, filas_origen=filas,
                        nubes=nubes.astype(np.float32),
                        diagramas=diagramas.astype(np.float32),
                        imagenes=imagen.astype(np.float32),
                        X_topologico=X_topologico.astype(np.float32))
    resumen_diagramas(diagramas, y, filas, args.salida / "resumen_h1.csv")

    metadatos = {
        "fuente": str(args.csv.resolve()), "clases": {"0": "N (normal)", "2": "V (ectópico ventricular)"},
        "semilla": args.semilla, "por_clase": args.por_clase,
        "takens": {"dimension": 3, "retardo": 3, "stride": 1},
        "homologia": [0, 1], "persistence_image": {"sigma": 0.05, "n_bins": 20,
                                               "peso": "persistencia"},
        "formas": {"X_crudo": list(X.shape), "nubes": list(nubes.shape),
                   "diagramas": list(diagramas.shape), "imagenes": list(imagen.shape),
                   "X_topologico": list(X_topologico.shape)},
        "giotto_tda": gtda.__version__,
    }
    if args.bottleneck_por_clase:
        forma = distancia_bottleneck(diagramas, y, args.bottleneck_por_clase,
                                    args.trabajadores, args.salida / "bottleneck.npz")
        metadatos["bottleneck"] = {"por_clase": args.bottleneck_por_clase,
                                    "forma": list(forma), "delta": 0, "combinacion": "max(H0,H1)"}
    (args.salida / "metadatos.json").write_text(json.dumps(metadatos, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Nubes {nubes.shape}; diagramas {diagramas.shape}; imágenes {imagen.shape}", flush=True)
    print(f"Guardado en {args.salida.resolve()}")


if __name__ == "__main__":
    main()
