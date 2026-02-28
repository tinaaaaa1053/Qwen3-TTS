import torch
import soundfile as sf
# 直接导入，因为已经 pip install 过了
from qwen_tts import Qwen3TTSModel 

# 配置 H100 最佳参数
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:0",
    dtype=torch.bfloat16,         # H100 必须用 BF16
    attn_implementation="flash_attention_2", # 必须用 FA2
)

# 定义素材
# 替换为您实际的参考音频路径
ref_audio = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone.wav" 
ref_text  = "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you."

# 目标文本
target_text = "Checking system status on H100 GPU. Voice cloning sequence initiated. All systems nominal."

print("开始生成...")
wavs, sr = model.generate_voice_clone(
    text=target_text,
    language="English",
    ref_audio=ref_audio,
    ref_text=ref_text,
)

# 保存
output_path = "h100_clone_output.wav"
sf.write(output_path, wavs[0], sr)
print(f"完成！音频已保存至 {output_path}")