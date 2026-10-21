"""Verifica la integridad de los dos CSV MIT-BIH usados en el experimento."""

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ARCHIVOS = {
    "mitbih_train.csv": "ae4c80504ad1d0be5c3f1b3c6ef02851e4d1b05d05805fce2a73374f083a700e",
    "mitbih_test.csv": "f3fdb3fd318275e653fac082b617d8c1ef09fca619a0606dc22c863451c77333",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1024 * 1024), b""):
            digest.update(bloque)
    return digest.hexdigest()


def main():
    errores = []
    for nombre, esperado in ARCHIVOS.items():
        path = ROOT / "data" / nombre
        if not path.is_file():
            errores.append(f"Falta {path}")
            continue
        encontrado = sha256(path)
        if encontrado != esperado:
            errores.append(f"{nombre}: SHA-256 distinto ({encontrado})")
        else:
            print(f"OK {nombre}: {encontrado}")
    if errores:
        raise SystemExit("\n".join(errores))


if __name__ == "__main__":
    main()
