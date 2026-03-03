# check_missing_annotations.py
from pathlib import Path


def main():
    # ============================
    # CONFIGURE AQUI
    # ============================
    ROOT_DIR = r"../annotations/"
    OUTPUT_FILE = Path("outputs/missing_annotations.dtxt")
    # ============================

    root = Path(ROOT_DIR).resolve()

    if not root.exists() or not root.is_dir():
        raise ValueError(f"Root inválido: {root}")

    missing = []

    # Varre recursivamente
    for img_path in root.rglob("*"):
        if not img_path.is_file():
            continue

        if img_path.suffix.lower() not in (".jpg", ".jpeg"):
            continue

        expected_json = img_path.with_suffix(".json")

        if not expected_json.exists():
            missing.append(str(img_path))

    # Cria pasta outputs se não existir
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Salva resultado
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        if not missing:
            f.write("Todos os JPG possuem JSON correspondente.\n")
        else:
            f.write(f"{len(missing)} imagens sem JSON correspondente:\n\n")
            for path in missing:
                f.write(path + "\n")

    print(f"Processo finalizado. Resultado salvo em: {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()