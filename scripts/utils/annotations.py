import re
import pandas as pd
from pathlib import Path

def get_missing_video_annotation_data_info(txt_path: str = "outputs/missing_annotations.dtxt") -> pd.DataFrame:
    """
    Lê um arquivo .txt contendo caminhos de imagens e retorna um DataFrame
    com combinações únicas de:
    SUBJECT | ACTIVITY | ROUTINE | CAMERA
    """
    
    pattern = re.compile(
        r"subject(\d+)[\\/]+activity(\d+)[\\/]+routine(\d+)_cam(\d+)",
        re.IGNORECASE
    )

    records = []

    with open(txt_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            match = pattern.search(line)
            if match:
                subject, activity, routine, camera = map(int, match.groups())
                records.append({
                    "SUBJECT": subject,
                    "ACTIVITY": activity,
                    "ROUTINE": routine,
                    "CAMERA": camera
                })

    df = pd.DataFrame(records).drop_duplicates().sort_values(
        by=["SUBJECT", "ACTIVITY", "ROUTINE", "CAMERA"]
    ).reset_index(drop=True)

    return df

def generate_report_missing_annotations(root_dir = r"../annotations/", output_dir = r"outputs/"):

    output_file = Path(f"{output_dir}missing_annotations.dtxt")

    root = Path(root_dir).resolve()

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
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Salva resultado
    with output_file.open("w", encoding="utf-8") as f:
        if not missing:
            f.write("Todos os JPG possuem JSON correspondente.\n")
        else:
            f.write(f"{len(missing)} imagens sem JSON correspondente:\n\n")
            for path in missing:
                f.write(path + "\n")

    print(f"Processo finalizado. Resultado salvo em: {output_file.resolve()}")
    
from pathlib import Path

def generate_report_missing_images(root_dir = r"../annotations/", output_dir = r"outputs/"):

    output_file = Path(f"{output_dir}missing_images.dtxt")

    root = Path(root_dir).resolve()

    if not root.exists() or not root.is_dir():
        raise ValueError(f"Root inválido: {root}")

    missing = []

    # Varre recursivamente
    for json_path in root.rglob("*"):
        if not json_path.is_file():
            continue

        if json_path.suffix.lower() != ".json":
            continue

        # Verifica JPG e JPEG
        expected_jpg = json_path.with_suffix(".jpg")
        expected_jpeg = json_path.with_suffix(".jpeg")

        if not expected_jpg.exists() and not expected_jpeg.exists():
            missing.append(str(json_path))

    # Cria pasta outputs se não existir
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Salva resultado
    with output_file.open("w", encoding="utf-8") as f:
        if not missing:
            f.write("Todos os JSON possuem JPG correspondente.\n")
        else:
            f.write(f"{len(missing)} JSON sem JPG correspondente:\n\n")
            for path in missing:
                f.write(path + "\n")

    print(f"Processo finalizado. Resultado salvo em: {output_file.resolve()}")
