"""
Téléchargement automatique des modèles depuis HuggingFace Hub au démarrage.
"""

import os
from pathlib import Path
from huggingface_hub import hf_hub_download

# ── Config HuggingFace ────────────────────────────────────────────────────────
HF_REPO_ID = "TON_USERNAME/wildvision-models"   # ← à remplacer
HF_TOKEN   = os.getenv("HF_TOKEN")              # variable d'env Render

MODELS_DIR = Path(__file__).parent.parent / "models"
FILENAMES  = ["yolov8m_best.pt", "rfdetr_best.pth"]


def ensure_models():
    """Vérifie et télécharge les modèles manquants depuis HuggingFace."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for filename in FILENAMES:
        dest = MODELS_DIR / filename
        if dest.exists():
            print(f"✔  {filename} déjà présent ({dest.stat().st_size / 1e6:.1f} MB)")
        else:
            print(f"⏬ Téléchargement de {filename} depuis HuggingFace...")
            try:
                hf_hub_download(
                    repo_id=HF_REPO_ID,
                    filename=filename,
                    local_dir=str(MODELS_DIR),
                    token=HF_TOKEN
                )
                print(f"✅ {filename} téléchargé ({dest.stat().st_size / 1e6:.1f} MB)")
            except Exception as e:
                print(f"❌ Échec téléchargement {filename} : {e}")