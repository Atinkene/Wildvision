"""
WildVision API — FastAPI backend
Détection d'animaux sauvages via YOLOv8m et RF-DETR
Supports : image upload, video upload, webcam streaming (WebSocket)
"""

import os, io, cv2, json, time, base64, tempfile, asyncio
from pathlib import Path
from typing import Optional
import numpy as np
from PIL import Image

from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, Query, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# ── Chemins modèles ───────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent.parent
YOLO_BEST   = BASE_DIR / "models" / "yolov8m_best.pt"
RFDETR_BEST = BASE_DIR / "models" / "rfdetr_best.pth"

# ── 15 classes (Jaguar retiré — absent du dataset d'entraînement) ─────────────
NAMES = [
    'Brown-bear', 'Cheetah', 'Crocodile', 'Elephant', 'Fox', 'Giraffe',
    'Hedgehog', 'Human', 'Leopard', 'Lion', 'Lynx',
    'Ostrich', 'Rhinoceros', 'Tiger', 'Zebra'
]
NC = len(NAMES)  # 15

import random; random.seed(42)
CLASS_COLORS = {n: tuple(random.randint(80, 230) for _ in range(3)) for n in NAMES}

# ── Chargement des modèles ────────────────────────────────────────────────────
print("⏳ Chargement des modèles...")
yolo_model   = None
rfdetr_model = None

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️  Device détecté : {DEVICE.upper()}")

# ── YOLOv8m ───────────────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO
    if YOLO_BEST.exists():
        yolo_model = YOLO(str(YOLO_BEST))
        print(f"✅ YOLOv8m chargé (device={DEVICE})")
    else:
        yolo_model = YOLO("yolov8m.pt")
        print("⚠️  YOLOv8m — poids fine-tuned introuvables, fallback COCO")
except Exception as e:
    print(f"❌ YOLOv8m non disponible : {e}")

# ── RF-DETR — chargement checkpoint fine-tuné EMA ────────────────────────────
try:
    from rfdetr import RFDETRBase

    if RFDETR_BEST.exists():
        # Instancier avec le bon nombre de classes
        rfdetr_model = RFDETRBase(num_classes=NC, device=DEVICE)

        # Charger le checkpoint EMA
        ckpt = torch.load(str(RFDETR_BEST), map_location=DEVICE)
        print(f"[RF-DETR] Clés checkpoint : {list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}")

        # Le checkpoint Lightning contient 'state_dict' avec préfixe 'model.'
        raw_sd = ckpt.get("state_dict") or ckpt.get("model") or ckpt

        # Nettoyer le préfixe 'model.' ajouté par Lightning
        state_dict = {
            k.replace("model.", "", 1): v
            for k, v in raw_sd.items()
            if not k.startswith("criterion") and not k.startswith("postprocess")
        }
        print(f"[RF-DETR] {len(state_dict)} clés après nettoyage préfixe")

        # Trouver le vrai module PyTorch (descendre dans ModelContext)
        inner = rfdetr_model.model
        while hasattr(inner, "model") and not hasattr(inner, "load_state_dict"):
            inner = inner.model

        if hasattr(inner, "load_state_dict"):
            missing, unexpected = inner.load_state_dict(state_dict, strict=False)
            print(f"[RF-DETR] missing={len(missing)}, unexpected={len(unexpected)}")
            inner.eval()
            print(f"✅ RF-DETR EMA chargé depuis {RFDETR_BEST.name} (device={DEVICE})")
        else:
            # Dernier recours : utiliser get_model() si disponible
            inner2 = rfdetr_model.get_model() if hasattr(rfdetr_model, "get_model") else None
            if inner2 and hasattr(inner2, "load_state_dict"):
                missing, unexpected = inner2.load_state_dict(state_dict, strict=False)
                print(f"[RF-DETR via get_model()] missing={len(missing)}, unexpected={len(unexpected)}")
                inner2.eval()
                print(f"✅ RF-DETR EMA chargé (get_model) depuis {RFDETR_BEST.name}")
            else:
                print(f"⚠️  Impossible de charger les poids — structure : {type(inner)}, {dir(inner)}")
                # RF-DETR reste chargé avec poids COCO (15 classes réinitialisées)
    else:
        print(f"⚠️  RF-DETR — checkpoint introuvable : {RFDETR_BEST}")
        rfdetr_model = None

except Exception as e:
    import traceback
    print(f"❌ RF-DETR non disponible : {e}")
    traceback.print_exc()
    rfdetr_model = None

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="WildVision API",
    description="Détection d'animaux sauvages — YOLOv8m vs RF-DETR",
    version="1.0.0"
)

app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

static_dir = BASE_DIR / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# ── Téléchargement + chargement des modèles au startup ───────────────────────
@app.on_event("startup")
async def startup_event():
    global yolo_model, rfdetr_model

    # 1. Télécharger les modèles manquants
    from app.download_models import ensure_models
    ensure_models()

    # 2. Charger YOLOv8m
    try:
        from ultralytics import YOLO
        if YOLO_BEST.exists():
            yolo_model = YOLO(str(YOLO_BEST))
            print(f"✅ YOLOv8m chargé (device={DEVICE})")
        else:
            yolo_model = YOLO("yolov8m.pt")
            print("⚠️  YOLOv8m — poids fine-tuned introuvables, fallback COCO")
    except Exception as e:
        print(f"❌ YOLOv8m non disponible : {e}")

    # 3. Charger RF-DETR
    try:
        from rfdetr import RFDETRBase
        if RFDETR_BEST.exists():
            rfdetr_model = RFDETRBase(num_classes=NC, device=DEVICE)
            ckpt = torch.load(str(RFDETR_BEST), map_location=DEVICE)
            print(f"[RF-DETR] Clés checkpoint : {list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}")
            raw_sd = ckpt.get("state_dict") or ckpt.get("model") or ckpt
            state_dict = {
                k.replace("model.", "", 1): v
                for k, v in raw_sd.items()
                if not k.startswith("criterion") and not k.startswith("postprocess")
            }
            inner = rfdetr_model.model
            while hasattr(inner, "model") and not hasattr(inner, "load_state_dict"):
                inner = inner.model
            if hasattr(inner, "load_state_dict"):
                missing, unexpected = inner.load_state_dict(state_dict, strict=False)
                print(f"[RF-DETR] missing={len(missing)}, unexpected={len(unexpected)}")
                inner.eval()
            print(f"✅ RF-DETR chargé (device={DEVICE})")
        else:
            print(f"⚠️  RF-DETR — checkpoint introuvable : {RFDETR_BEST}")
    except Exception as e:
        import traceback
        print(f"❌ RF-DETR non disponible : {e}")
        traceback.print_exc()

# ── Utilitaires ───────────────────────────────────────────────────────────────
def predict_yolo(img_bgr: np.ndarray, conf: float = 0.25) -> dict:
    if yolo_model is None:
        return {"error": "YOLOv8m non disponible"}
    t0 = time.time()
    results = yolo_model.predict(img_bgr, verbose=False, conf=conf, device=DEVICE)[0]
    elapsed = (time.time() - t0) * 1000
    detections = []
    for box in results.boxes:
        cls_id = int(box.cls.item())
        class_name = NAMES[cls_id] if cls_id < NC else f"[ID:{cls_id}]"
        detections.append({
            "bbox":       [round(v, 1) for v in box.xyxy[0].tolist()],
            "class_id":   cls_id,
            "class_name": class_name,
            "confidence": round(float(box.conf.item()), 3),
            "color":      CLASS_COLORS.get(class_name, (200, 200, 200))
        })
    return {"model": "yolov8m", "detections": detections,
            "ms": round(elapsed, 1), "fps": round(1000 / elapsed, 1) if elapsed > 0 else 0}


def predict_rfdetr(img_bgr: np.ndarray, conf: float = 0.25) -> dict:
    if rfdetr_model is None:
        return {"error": "RF-DETR non disponible"}
    t0 = time.time()
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    d = rfdetr_model.predict(pil, threshold=conf)
    elapsed = (time.time() - t0) * 1000
    detections = []
    if d and len(d) > 0:
        for box, cls_id, score in zip(d.xyxy.tolist(), d.class_id.tolist(), d.confidence.tolist()):
            cls_id = int(cls_id)
            # Checkpoint fine-tuné → IDs 0-based alignés sur NAMES
            class_name = NAMES[cls_id] if cls_id < NC else f"[ID:{cls_id}]"
            detections.append({
                "bbox":       [round(v, 1) for v in box],
                "class_id":   cls_id,
                "class_name": class_name,
                "confidence": round(float(score), 3),
                "color":      CLASS_COLORS.get(class_name, (200, 200, 200))
            })
    return {"model": "rfdetr", "detections": detections,
            "ms": round(elapsed, 1), "fps": round(1000 / elapsed, 1) if elapsed > 0 else 0}


def draw_detections(img_bgr: np.ndarray, detections: list) -> np.ndarray:
    img = img_bgr.copy()
    for det in detections:
        if "bbox" not in det: continue
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        col = tuple(det.get("color", (200, 200, 200)))
        label = f"{det['class_name']} {det['confidence']:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 6, y1), col, -1)
        cv2.putText(img, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img


def encode_image_b64(img_bgr: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode()


def run_model(model_name: str, img_bgr: np.ndarray, conf: float) -> dict:
    if model_name == "yolo":
        return predict_yolo(img_bgr, conf)
    elif model_name == "rfdetr":
        return predict_rfdetr(img_bgr, conf)
    elif model_name == "both":
        return {"yolo": predict_yolo(img_bgr, conf), "rfdetr": predict_rfdetr(img_bgr, conf)}
    raise HTTPException(400, "model doit être 'yolo', 'rfdetr' ou 'both'")

# ── Routes REST ───────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    html = static_dir / "index.html"
    return html.read_text() if html.exists() else "<h1>WildVision API</h1><p>Voir /docs</p>"


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "models": {"yolov8m": yolo_model is not None, "rfdetr": rfdetr_model is not None},
        "classes": NAMES
    }


@app.post("/predict/image")
async def predict_image(
    file:      UploadFile = File(...),
    model:     str   = Query("both", description="yolo | rfdetr | both"),
    conf:      float = Query(0.25, ge=0.01, le=1.0),
    annotated: bool  = Query(True, description="Retourner l'image annotée en base64")
):
    """Détection sur une image uploadée (JPG, PNG, WEBP)."""
    data = await file.read()
    arr  = np.frombuffer(data, np.uint8)
    img  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Image invalide")

    result = run_model(model, img, conf)

    if annotated:
        dets = result.get("detections", []) if model != "both" else \
               result.get("yolo", {}).get("detections", [])
        ann = draw_detections(img, dets)
        result["annotated_image"] = encode_image_b64(ann)
        result["image_size"] = {"w": img.shape[1], "h": img.shape[0]}

    return JSONResponse(result)


@app.post("/predict/video")
async def predict_video(
    file:    UploadFile = File(...),
    model:   str   = Query("yolo"),
    conf:    float = Query(0.25),
    every_n: int   = Query(2, description="Traiter 1 frame sur N")
):
    """Détection sur une vidéo uploadée — retourne JSON avec stats par frame."""
    data = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(data); tmp_path = tmp.name

    cap     = cv2.VideoCapture(tmp_path)
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 25
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames_result, frame_idx = [], 0
    t_start = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        if frame_idx % every_n == 0:
            r = run_model(model, frame, conf)
            frames_result.append({
                "frame": frame_idx,
                "time_s": round(frame_idx / fps_src, 2),
                "detections": r.get("detections", []),
                "ms": r.get("ms", 0)
            })
        frame_idx += 1

    cap.release()
    os.unlink(tmp_path)

    from collections import Counter
    all_dets = [d["class_name"] for f in frames_result for d in f["detections"]]
    return JSONResponse({
        "model": model,
        "video_fps": round(fps_src, 1),
        "total_frames": total,
        "processed_frames": len(frames_result),
        "processing_time_s": round(time.time() - t_start, 2),
        "class_counts": dict(Counter(all_dets)),
        "frames": frames_result
    })


@app.post("/predict/frame")
async def predict_frame(
    file:  UploadFile = File(...),
    model: str   = Query("yolo"),
    conf:  float = Query(0.25)
):
    """Détection sur une frame webcam (JPEG) — optimisé latence."""
    data = await file.read()
    arr  = np.frombuffer(data, np.uint8)
    img  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Frame invalide")

    result = run_model(model, img, conf)
    ann    = draw_detections(img, result.get("detections", []))
    result["annotated_image"] = encode_image_b64(ann)
    return JSONResponse(result)


# ── WebSocket — streaming webcam ──────────────────────────────────────────────
@app.websocket("/ws/webcam")
async def webcam_stream(websocket: WebSocket):
    """
    WebSocket pour détection en temps réel depuis webcam.
    Client envoie : JSON { frame: "<base64 JPEG>", model: "yolo", conf: 0.25 }
    Serveur répond : JSON { detections: [...], annotated_image: "<base64>", ms: float }
    """
    await websocket.accept()
    print("📡 WebSocket connecté")
    try:
        while True:
            msg     = await websocket.receive_text()
            payload = json.loads(msg)

            b64 = payload.get("frame", "")
            if "," in b64: b64 = b64.split(",")[1]
            img_bytes = base64.b64decode(b64)
            arr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

            if img is None:
                await websocket.send_json({"error": "frame invalide"})
                continue

            model_name = payload.get("model", "yolo")
            conf       = float(payload.get("conf", 0.25))

            result = run_model(model_name, img, conf)
            dets   = result.get("detections", [])
            ann    = draw_detections(img, dets)

            await websocket.send_json({
                "detections":     dets,
                "annotated_image": encode_image_b64(ann),
                "ms":    result.get("ms", 0),
                "fps":   result.get("fps", 0),
                "model": model_name
            })

    except WebSocketDisconnect:
        print("📡 WebSocket déconnecté")
    except Exception as e:
        print(f"❌ WebSocket erreur : {e}")
        await websocket.close()

# ── Lancement ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=8000, reload=False)