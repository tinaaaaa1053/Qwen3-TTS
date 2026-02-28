#!/usr/bin/env python3
"""
Qwen3-TTS Voice Clone Service - FastAPI Wrapper for H100
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

import torch
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse

# 忽略无关警告
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
# 1. 核心依赖引入
# ============================================================
# 确保能找到本地的 qwen_tts 模块
sys.path.append(os.getcwd())

try:
    from qwen_tts import Qwen3TTSModel
except ImportError as e:
    print(f"Error importing qwen_tts: {e}")
    print("请确保您在包含 'qwen_tts' 文件夹的目录下运行，或已 pip install -e .")
    sys.exit(1)

# ============================================================
# 2. 配置与工具函数
# ============================================================
SERVER_PORT = 6000 # 避免与 InfiniteTalk 端口冲突

# 配置类
class TTSConfig:
    model_path = "Qwen/Qwen3-TTS-12Hz-1.7B-Base" # 或本地绝对路径
    device = "cuda:0" 
    dtype = torch.bfloat16        # H100 最佳精度
    attn_impl = "flash_attention_2" # H100 必备加速

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def save_upload_to_temp(upload_file: UploadFile) -> str:
    """保存上传文件到临时目录"""
    ext = os.path.splitext(upload_file.filename)[1]
    if not ext: ext = ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        shutil.copyfileobj(upload_file.file, tmp)
        return tmp.name

# ============================================================
# 3. 推理服务类 (Service Class)
# ============================================================
class QwenTTSService:
    def __init__(self):
        logger.info("Initializing Qwen3-TTS Service on H100...")
        
        try:
            self.model = Qwen3TTSModel.from_pretrained(
                TTSConfig.model_path,
                device_map=TTSConfig.device,
                dtype=TTSConfig.dtype,
                attn_implementation=TTSConfig.attn_impl,
            )
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise e

    def infer(self, ref_audio_path: str, ref_text: str, target_text: str, language: str = "English"):
        """
        执行 Voice Clone 推理
        """
        temp_dir = tempfile.mkdtemp()
        try:
            logger.info(f"Processing Request. Target Text Length: {len(target_text)}")
            
            # 1. 生成 (Generate)
            # x_vector_only_mode 逻辑：如果 ref_text 为空，则开启该模式
            use_x_vector = False
            if not ref_text or ref_text.strip() == "":
                use_x_vector = True
                logger.info("Ref text is empty, using x_vector_only_mode=True")

            if use_x_vector:
                # 如果需要构建 prompt 再传参，可以在这里做。
                # 简单起见，直接调用 generate_voice_clone 的 x_vector 逻辑
                # 注意：目前 model.generate_voice_clone 接口在源码中可能不直接暴露 x_vector_only_mode 参数
                # 如果源码支持 create_voice_clone_prompt，则手动构建：
                prompt = self.model.create_voice_clone_prompt(
                    ref_audio=ref_audio_path,
                    ref_text="",
                    x_vector_only_mode=True
                )
                wavs, sr = self.model.generate_voice_clone(
                    text=target_text,
                    language=language,
                    voice_clone_prompt=prompt
                )
            else:
                # 标准模式：提供参考音频和参考文本
                wavs, sr = self.model.generate_voice_clone(
                    text=target_text,
                    language=language,
                    ref_audio=ref_audio_path,
                    ref_text=ref_text,
                )

            # 2. 保存输出 (Save Output)
            output_filename = f"qwen_clone_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
            temp_output_path = os.path.join(temp_dir, output_filename)
            
            # 写入文件
            sf.write(temp_output_path, wavs[0], sr)
            
            # 3. 持久化移动 (Move to persistent temp for download)
            final_persist_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
            shutil.move(temp_output_path, final_persist_path)
            
            return final_persist_path

        except Exception as e:
            logger.error(f"Inference Error: {e}", exc_info=True)
            raise e
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

# ============================================================
# 4. FastAPI App Setup
# ============================================================
gpu_lock = threading.Lock()
service = None

@asynccontextmanager
async def lifespan(_: FastAPI):
    global service
    # 启动时加载模型
    service = QwenTTSService()
    yield
    # 关闭时清理逻辑 (如有)
    logger.info("Shutting down Qwen-TTS Service...")

app = FastAPI(title="Qwen3-TTS Voice Clone API", lifespan=lifespan)

@app.post("/api/voice_clone")
def voice_clone_endpoint(
    ref_audio: UploadFile = File(..., description="Reference audio file (wav/mp3) of the speaker"),
    ref_text: str = Form(None, description="Transcript of what is said in the ref_audio"),
    target_text: str = Form(..., description="New text you want the voice to speak"),
    language: str = Form("English", description="Target language (English/Chinese)")
):
    """
    Voice Clone 接口: 
    上传一段声音和它的文本，输入新文本，返回克隆后的音频。
    - ref_text 可选
    - 未提供或为空时自动启用 x-vector-only 模式
    """
    tmp_audio_path = None
    output_audio_path = None
    
    # 获取 GPU 锁，防止并发导致显存溢出
    if not gpu_lock.acquire(blocking=True):
        raise HTTPException(503, "Server is busy processing another request.")
        
    try:
        # 1. 保存上传的参考音频
        tmp_audio_path = save_upload_to_temp(ref_audio)
        
        # 2. 调用模型推理
        output_audio_path = service.infer(
            ref_audio_path=tmp_audio_path,
            ref_text=ref_text,
            target_text=target_text,
            language=language
        )
        
        # 3. 返回生成的音频文件
        return FileResponse(
            output_audio_path, 
            media_type="audio/wav", 
            filename="cloned_voice.wav"
        )
        
    except Exception as e:
        logger.exception("Endpoint Error")
        raise HTTPException(500, detail=str(e))
    finally:
        gpu_lock.release()
        # 清理上传的临时文件
        if tmp_audio_path and os.path.exists(tmp_audio_path):
            os.remove(tmp_audio_path)
        # 注意：output_audio_path 不在这里删除，因为 FileResponse 需要读取它。
        # 实际生产中建议使用后台任务定期清理 /tmp

if __name__ == "__main__":
    import uvicorn
    # 监听所有 IP，端口 6000
    uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT)