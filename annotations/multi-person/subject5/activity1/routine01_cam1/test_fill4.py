import json
import cv2
import numpy as np
from pathlib import Path

# -------------------------
# IO / máscaras
# -------------------------
def load_mask_from_xanylabeling(json_path: str, labels=None):
    """
    JSON estilo LabelMe/XAnyLabeling:
      imagePath, imageHeight, imageWidth, shapes[{label, shape_type='polygon', points}]
    """
    p = Path(json_path)
    data = json.loads(p.read_text(encoding="utf-8"))
    H, W = int(data["imageHeight"]), int(data["imageWidth"])
    mask = np.zeros((H, W), dtype=np.uint8)

    for shp in data.get("shapes", []):
        if shp.get("shape_type") != "polygon":
            continue
        lbl = shp.get("label")
        if labels is not None and lbl not in labels:
            continue
        pts = np.array(shp["points"], dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)

    return mask, data.get("imagePath")

def bbox_from_mask(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

def make_band_above_roi(roi_mask, band_px=10):
    """
    Faixa acima do topo da ROI (para amostrar cor da camisa).
    """
    bb = bbox_from_mask(roi_mask)
    band = np.zeros_like(roi_mask)
    if bb is None:
        return band
    x0, y0, x1, _ = bb
    yA0 = max(0, y0 - band_px)
    yA1 = y0
    band[yA0:yA1, x0:x1 + 1] = 255
    return band

def local_window_around_roi(roi_mask, x_pad=25, down_px=25, up_px=10):
    """
    Janela local para impedir que o "componente camisa" vire a camiseta inteira.
    """
    h, w = roi_mask.shape
    bb = bbox_from_mask(roi_mask)
    win = np.zeros_like(roi_mask)
    if bb is None:
        return win
    x0, y0, x1, y1 = bb
    X0 = max(0, x0 - x_pad)
    X1 = min(w, x1 + 1 + x_pad)
    Y0 = max(0, y0 - up_px)
    Y1 = min(h, y1 + 1 + down_px)
    win[Y0:Y1, X0:X1] = 255
    return win

# -------------------------
# Cor / threshold em Lab
# -------------------------
def median_lab(img_bgr, mask):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        med = np.median(lab.reshape(-1, 3), axis=0)
    else:
        med = np.median(lab[ys, xs, :], axis=0)
    return med  # (L,a,b)

def color_threshold_from_sample(img_bgr, sample_mask, tol_L=20, tol_a=12, tol_b=12):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L0, a0, b0 = median_lab(img_bgr, sample_mask)

    L = lab[:, :, 0].astype(np.int16)
    a = lab[:, :, 1].astype(np.int16)
    b = lab[:, :, 2].astype(np.int16)

    m = (np.abs(L - int(L0)) <= tol_L) & (np.abs(a - int(a0)) <= tol_a) & (np.abs(b - int(b0)) <= tol_b)
    return (m.astype(np.uint8) * 255)

# -------------------------
# Connected components
# -------------------------
def pick_component_above_roi(components, roi_bb):
    """
    Escolhe o componente que está acima (ou tocando) o topo da ROI e mais próximo.
    """
    labels_map, stats, centroids = components
    x0, y0, x1, _ = roi_bb

    best_id = None
    best_score = 1e18

    for cid in range(1, stats.shape[0]):
        x, y, w, h, area = stats[cid]
        # se o componente está totalmente abaixo do topo da ROI, ignore
        if y >= y0:
            continue

        # distância vertical até o topo da ROI
        dist_v = max(0, y0 - (y + h))
        # penaliza distância horizontal do centroide à ROI
        cx = centroids[cid][0]
        dist_h = 0.0
        if cx < x0:
            dist_h = x0 - cx
        elif cx > x1:
            dist_h = cx - x1

        score = dist_v + 0.25 * dist_h + (0.00001 * max(0, 5000 - area))  # leve viés contra componentes minúsculos
        if score < best_score:
            best_score = score
            best_id = cid

    return best_id

# -------------------------
# Preenchimento + gradiente de "barra"
# -------------------------
def median_color_bgr(img_bgr, mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return (255, 255, 255)
    pix = img_bgr[ys, xs, :]
    med = np.median(pix, axis=0).astype(np.uint8)
    return (int(med[0]), int(med[1]), int(med[2]))

def fill_target_with_color(img_bgr, target_mask, fill_bgr, feather_sigma=1.6):
    """
    Só altera pixels dentro de target_mask; fora fica idêntico.
    Faz um feather leve (gauss) só para amaciar bordas.
    """
    out = img_bgr.copy()

    # alpha suave (0..1) só na região do target
    roi = (target_mask > 0).astype(np.float32)
    alpha = cv2.GaussianBlur(roi, (0, 0), feather_sigma)
    alpha = np.clip(alpha, 0.0, 1.0)[..., None]

    fill_img = np.zeros_like(out, dtype=np.float32)
    fill_img[:, :] = np.array(fill_bgr, dtype=np.float32)

    out_f = out.astype(np.float32)
    out_f = out_f * (1.0 - alpha) + fill_img * alpha
    return np.clip(out_f, 0, 255).astype(np.uint8)

def darken_bottom_edge(img_bgr, target_mask, band_px=12, strength=0.55):
    """
    Escurece uma faixa perto da borda INFERIOR do target_mask (gradiente),
    simulando "fim da camisa".
    """
    h, w = target_mask.shape
    ys, xs = np.where(target_mask > 0)
    if len(xs) == 0:
        return img_bgr

    bottom = int(ys.max())
    y0 = max(0, bottom - band_px + 1)
    y1 = bottom + 1

    # alpha 0..strength na faixa, só dentro da máscara
    alpha = np.zeros((h, w), dtype=np.float32)
    denom = max(1, (y1 - y0 - 1))
    for y in range(y0, y1):
        t = (y - y0) / denom
        alpha[y, :] = t * strength

    alpha[target_mask == 0] = 0.0

    out = img_bgr.astype(np.float32)
    out *= (1.0 - alpha[..., None])
    return np.clip(out, 0, 255).astype(np.uint8)

# -------------------------
# Main
# -------------------------
def process_one(
    json_path,
    image_path=None,
    labels=None,
    # LCC / cor
    band_px=12, tol_L=22, tol_a=14, tol_b=14,
    open_k=3, close_k=9,
    # limitar quanto da camisa entra
    x_pad=28, up_px=12, down_px=35,
    # amostrar cor (evitar cinza)
    offset_px=4, sample_h=10,
    # visual
    feather_sigma=1.6,
    bottom_band_px=14, bottom_strength=0.55,
    out_path=None
):
    jp = Path(json_path)
    roi_mask, rel = load_mask_from_xanylabeling(json_path, labels=labels)

    if image_path is None:
        if rel is None:
            raise ValueError("JSON não tem imagePath; passe image_path manualmente.")
        image_path = (jp.parent / rel).as_posix()

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Não consegui abrir: {image_path}")

    roi_bb = bbox_from_mask(roi_mask)
    if roi_bb is None:
        raise RuntimeError("ROI vazia (máscara do JSON sem pixels).")

    # 1) faixa acima da ROI para "cor da camisa"
    band = make_band_above_roi(roi_mask, band_px=band_px)

    # 2) threshold por cor (camisa provável)
    shirt_prob = color_threshold_from_sample(img, band, tol_L=tol_L, tol_a=tol_a, tol_b=tol_b)

    # 3) morfologia para reduzir ruído e unir camisa (sem exagerar)
    if open_k > 0:
        shirt_prob = cv2.morphologyEx(shirt_prob, cv2.MORPH_OPEN, np.ones((open_k, open_k), np.uint8), iterations=1)
    if close_k > 0:
        shirt_prob = cv2.morphologyEx(shirt_prob, cv2.MORPH_CLOSE, np.ones((close_k, close_k), np.uint8), iterations=1)

    # 4) connected components
    n, labels_map, stats, centroids = cv2.connectedComponentsWithStats((shirt_prob > 0).astype(np.uint8), connectivity=8)
    comp_id = pick_component_above_roi((labels_map, stats, centroids), roi_bb)

    # 5) extrai componente de camisa, mas LIMITA localmente pra não pintar tudo
    win = local_window_around_roi(roi_mask, x_pad=x_pad, down_px=down_px, up_px=up_px)

    if comp_id is None:
        shirt_local = cv2.bitwise_and(shirt_prob, win)  # fallback: usa máscara de camisa provável
    else:
        shirt_comp = (labels_map == comp_id).astype(np.uint8) * 255
        shirt_local = cv2.bitwise_and(shirt_comp, win)

    # 6) máscara final que será modificada (ROI + camisa local)
    target = cv2.bitwise_or(roi_mask, shirt_local)
    # 6.1) adiciona faixa acima para “quebrar” a transição
    top_band = add_band_above_roi(roi_mask, pixels=12, x_pad=10)

    # opcional: só adiciona onde parece camisa (pra não pegar fundo)
    top_band = cv2.bitwise_and(top_band, shirt_prob)

    target_fill = cv2.bitwise_or(target, top_band)

    # 7) amostra cor: faixa logo acima da ROI, RESTRINGIDA ao shirt_local (evita puxar calça/fundo)
    x0, y0, x1, _ = roi_bb
    y1s = max(0, y0 - offset_px)
    y0s = max(0, y1s - sample_h)

    sample = np.zeros_like(roi_mask)
    sample[y0s:y1s, x0:x1 + 1] = 255
    sample = cv2.bitwise_and(sample, shirt_local)

    # fallback se sample ficar vazio: usa 'band' acima da ROI mas intersecta com shirt_prob
    if np.count_nonzero(sample) < 20:
        sample = cv2.bitwise_and(band, shirt_prob)

    # máscara para o blur “invadir” a camisa um pouco (mata a listra)
    blur_mask = expand_mask_up(target_fill, pixels=25)

    # em vez de cor sólida: usa blur do próprio frame (mantém gradiente)
    out = fill_target_with_blur(img, blur_mask, ksize=41, feather_sigma=4.0)

    # opcional: mantém seu “fim de camisa”
    out = darken_bottom_edge(out, target_fill, band_px=bottom_band_px, strength=bottom_strength)

    # salva
    if out_path is None:
        out_path = jp.with_suffix("").as_posix() + "_lcc_local_grad.jpg"

    cv2.imwrite(out_path, out)
    print("OK ->", out_path)
    return out_path

def blur_target_region_feather(img_bgr, target_mask, ksize=5, feather_sigma=2.0):
    """
    Blur apenas na região target_mask com borda suave (feather).
    """

    # imagem borrada
    blurred = cv2.GaussianBlur(img_bgr, (ksize, ksize), 0)

    # máscara -> alpha 0..1
    alpha = (target_mask > 0).astype(np.float32)

    # feather na borda
    alpha = cv2.GaussianBlur(alpha, (0,0), feather_sigma)

    alpha = np.clip(alpha, 0, 1)
    alpha = alpha[...,None]

    # mistura
    out = img_bgr.astype(np.float32)*(1-alpha) + blurred.astype(np.float32)*alpha

    return np.clip(out,0,255).astype(np.uint8)

def expand_mask_up(mask, pixels=8):
    """
    Expande máscara apenas para cima
    """
    kernel = np.ones((pixels, 1), np.uint8)

    # anchor no fundo do kernel faz crescer para cima
    expanded = cv2.dilate(mask, kernel, anchor=(0, pixels-1))

    return expanded

def blur_target_region(img_bgr, target_mask, ksize=3):
    """
    Aplica Gaussian blur apenas dentro da região target_mask
    """
    blurred = cv2.GaussianBlur(img_bgr, (ksize, ksize), 0)

    m = (target_mask > 0)  # <- 2D (H,W)

    out = img_bgr.copy()
    out[m] = blurred[m]    # funciona porque out[m] vira (N,3)

    return out

def fill_target_with_blur(img_bgr, target_mask, ksize=41, feather_sigma=3.0):
    """
    Em vez de pintar com cor sólida, preenche com o próprio frame borrado.
    Isso mantém iluminação/gradiente e reduz muito a listra.
    """
    blurred = cv2.GaussianBlur(img_bgr, (ksize, ksize), 0)

    alpha = (target_mask > 0).astype(np.float32)
    alpha = cv2.GaussianBlur(alpha, (0, 0), feather_sigma)
    alpha = np.clip(alpha, 0, 1)[..., None]

    out = img_bgr.astype(np.float32) * (1 - alpha) + blurred.astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)

def add_band_above_roi(roi_mask, pixels=10, x_pad=6):
    """
    Cria uma faixa (retângulo) acima do topo da ROI:
    - pixels: altura da faixa
    - x_pad: expande um pouco pros lados
    """
    h, w = roi_mask.shape
    bb = bbox_from_mask(roi_mask)
    band = np.zeros_like(roi_mask)
    if bb is None:
        return band

    x0, y0, x1, _ = bb
    X0 = max(0, x0 - x_pad)
    X1 = min(w, x1 + 1 + x_pad)

    Y0 = max(0, y0 - pixels)
    Y1 = y0

    band[Y0:Y1, X0:X1] = 255
    return band

if __name__ == "__main__":
    # Ajuste o json_path para o seu arquivo real
    process_one(
        json_path="frame_crop_cam1_000210.json",
        labels=None,

        # LCC/cor
        band_px=12, tol_L=22, tol_a=14, tol_b=14,
        open_k=3, close_k=9,

        # limitar quanto pinta
        x_pad=28, up_px=12, down_px=35,

        # amostragem de cor (evitar cinza)
        offset_px=4, sample_h=10,

        # aparência
        feather_sigma=1.6,
        bottom_band_px=14,
        bottom_strength=0.55
    )