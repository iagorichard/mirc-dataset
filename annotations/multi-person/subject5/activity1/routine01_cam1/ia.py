import json
import cv2
import numpy as np
from pathlib import Path
import torch
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline

def mask_from_labelme_polygon(json_path: str, labels=None) -> tuple[np.ndarray, dict]:
    """
    Lê JSON estilo LabelMe/XAnyLabeling:
      - imageHeight, imageWidth
      - shapes[{label, shape_type='polygon', points:[[x,y],...]}]
    Retorna máscara 0/255 (uint8) e metadados.
    """
    p = Path(json_path)
    data = json.loads(p.read_text(encoding="utf-8"))

    H = int(data["imageHeight"])
    W = int(data["imageWidth"])
    mask = np.zeros((H, W), dtype=np.uint8)

    for shp in data.get("shapes", []):
        if shp.get("shape_type") != "polygon":
            continue
        lbl = shp.get("label")
        if labels is not None and lbl not in labels:
            continue
        pts = np.array(shp["points"], dtype=np.int32)  # (N,2)
        cv2.fillPoly(mask, [pts], 255)

    meta = {
        "imagePath": data.get("imagePath"),
        "H": H,
        "W": W,
    }
    return mask, meta

def generate_mask(json_path: str, out_mask_path: str | None = None, labels=None):
    mask, meta = mask_from_labelme_polygon(json_path, labels=labels)
    jp = Path(json_path)

    if out_mask_path is None:
        out_mask_path = jp.with_suffix("").as_posix() + "_mask.png"

    cv2.imwrite(out_mask_path, mask)
    print("OK ->", out_mask_path)
    print("imagePath no JSON:", meta["imagePath"])



MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-inpainting"

def bbox_from_mask(mask_np: np.ndarray):
    ys, xs = np.where(mask_np > 0)
    if len(xs) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return x0, y0, x1, y1

def expand_bbox(bb, W, H, pad=40):
    x0,y0,x1,y1 = bb
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(W-1, x1 + pad)
    y1 = min(H-1, y1 + pad)
    return x0,y0,x1,y1

def resize_to_max(pil_img, max_side=512, resample=Image.BICUBIC):
    w, h = pil_img.size
    scale = min(max_side / max(w, h), 1.0)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    if (nw, nh) == (w, h):
        return pil_img
    return pil_img.resize((nw, nh), resample=resample)

def proccess(
    frame_path="frame_crop_cam1_000300.jpg",
    mask_path="frame_crop_cam1_000300_mask.png",
    out_path="frame_crop_cam1_000300_inpaint.png",
    prompt="extend the white t-shirt naturally, same fabric texture and lighting",
    negative_prompt="second shirt, undershirt, seam, stripe, artifact, blurry, distorted",
    max_side=512,
    pad=50,
    steps=28,
    guidance=5.5,
    seed=1234
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionInpaintPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype)
    pipe = pipe.to(device)

    # Otimizações pra pouca VRAM (ajuda bastante em 4GB). :contentReference[oaicite:3]{index=3}
    pipe.enable_vae_tiling()
    pipe.enable_attention_slicing()

    # Carrega frame e máscara
    frame = Image.open(frame_path).convert("RGB")
    mask  = Image.open(mask_path).convert("L")  # branco = inpaint

    W, H = frame.size
    mask_np = np.array(mask)
    bb = bbox_from_mask(mask_np)
    if bb is None:
        raise RuntimeError("Máscara vazia (não há pixels brancos).")

    x0,y0,x1,y1 = expand_bbox(bb, W, H, pad=pad)

    # Recorte local
    frame_crop = frame.crop((x0, y0, x1+1, y1+1))
    mask_crop  = mask.crop((x0, y0, x1+1, y1+1))

    # Redimensiona recorte pra caber na VRAM
    frame_in = resize_to_max(frame_crop, max_side=max_side, resample=Image.BICUBIC)
    mask_in  = resize_to_max(mask_crop,  max_side=max_side, resample=Image.NEAREST)

    # Garante máscara binária limpa
    mask_in = mask_in.point(lambda p: 255 if p > 127 else 0)

    g = torch.Generator(device=device).manual_seed(seed)

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=frame_in,
        mask_image=mask_in,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=g,
    ).images[0]

    # Volta ao tamanho do recorte original e cola no frame
    result_up = result.resize(frame_crop.size, Image.BICUBIC)

    out = frame.copy()
    out.paste(result_up, (x0, y0))

    out.save(out_path)
    print("OK ->", out_path)

if __name__ == "__main__":
    # Exemplo:
    generate_mask("frame_crop_cam1_000300.json", labels=None)  # ou labels=["seu_label"]
    proccess("frame_crop_cam1_000300.jpg")