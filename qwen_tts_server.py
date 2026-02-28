#!/usr/bin/env python3
"""
Qwen3-TTS Custom Voice Service - FastAPI (H100 Ready)
JSON-only API version
"""

import os
import sys
import logging
import threading
import tempfile
import shutil
import warnings
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

import torch
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

warnings.filterwarnings("ignore")

# ============================================================
# 0. GPU 设置 
# ============================================================
# 指定使用 GPU 6
target_gpu = "6"

# 注意：必须在 import torch 或加载模型之前设置此环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = target_gpu

print(f"[GPU SELECT] Force using physical GPU {target_gpu}")

# ============================================================
# 0.1 固定随机种子
# ============================================================
import random
import numpy as np
import torch
import os

def set_all_seeds(seed=42):
    """
    设置随机种子以确保可复现
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_all_seeds(42)


# ============================================================
# 1. 依赖
# ============================================================
try:
    from qwen_tts import Qwen3TTSModel
except ImportError:
    print("❌ Cannot import qwen_tts, please pip install qwen-tts")
    sys.exit(1)

# ============================================================
# 2. 配置
# ============================================================
SERVER_PORT = 6001

class TTSConfig:
    model_path = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    dtype = torch.bfloat16
    attn_impl = "flash_attention_2"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ============================================================
# 3. Service
# ============================================================
class QwenCustomVoiceService:
    def __init__(self):
        logger.info("Loading Qwen3-TTS CustomVoice model...")
        self.model = Qwen3TTSModel.from_pretrained(
            TTSConfig.model_path,
            device_map="auto",
            dtype=TTSConfig.dtype,
            attn_implementation=TTSConfig.attn_impl,
        )
        logger.info("Model loaded successfully.")

    def infer(
        self,
        text: str,
        language: str,
        speaker: str,
        instruct: Optional[str] = None
    ) -> str:
        temp_dir = tempfile.mkdtemp()
        try:
            logger.info(
                f"CustomVoice | speaker={speaker}, language={language}, "
                f"text_len={len(text)}"
            )

            wavs, sr = self.model.generate_custom_voice(
                text=text,
                language=language,
                speaker=speaker,
                instruct=instruct or ""
            )

            output_path = os.path.join(
                temp_dir,
                f"qwen_custom_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
            )
            sf.write(output_path, wavs[0], sr)

            final_path = tempfile.NamedTemporaryFile(
                delete=False, suffix=".wav"
            ).name
            shutil.move(output_path, final_path)

            return final_path

        except Exception as e:
            logger.error("Inference error", exc_info=True)
            raise e
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

# ============================================================
# 4. FastAPI
# ============================================================
gpu_lock = threading.Lock()
service: QwenCustomVoiceService | None = None

class CustomVoiceRequest(BaseModel):
    text: str
    speaker: str
    language: str = "Auto"
    instruct: Optional[str] = ""

@asynccontextmanager
async def lifespan(_: FastAPI):
    global service
    service = QwenCustomVoiceService()
    yield
    logger.info("Shutting down Qwen3-TTS CustomVoice service")

app = FastAPI(
    title="Qwen3-TTS Custom Voice API (JSON Only)",
    lifespan=lifespan
)

@app.post("/api/custom_voice")
def custom_voice_endpoint(payload: CustomVoiceRequest):
    """
    Custom Voice TTS (JSON only)
    """

    if not gpu_lock.acquire(blocking=True):
        raise HTTPException(503, "Server is busy")

    try:
        output_audio_path = service.infer(
            text=payload.text,
            language=payload.language,
            speaker=payload.speaker,
            instruct=payload.instruct
        )
        return FileResponse(
            output_audio_path,
            media_type="audio/wav",
            filename="custom_voice.wav"
        )
    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        gpu_lock.release()

# ============================================================
# 5. Entrypoint
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=SERVER_PORT,
        workers=1
    )
