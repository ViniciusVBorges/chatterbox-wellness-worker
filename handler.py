import runpod
import torch
import torchaudio as ta
import re
import os
import io
import base64
import tempfile
import urllib.request
import numpy as np

# --- Configurações de Áudio ---
SAMPLE_RATE = 24000
SILENCE_DURATION = 0.15  # Segundos de silêncio entre cada frase (chunk)

# Global model instance
tts_model = None

def load_model():
    global tts_model
    if tts_model is not None: return tts_model
    from chatterbox.tts import ChatterboxTTS
    device = "cuda"
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    print(f"[Handler] Using device: {device}")
    return tts_model

def split_text_into_chunks(text, chunk_size=200):
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if (len(current_chunk) + len(sentence) + 1 > chunk_size) and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk = (current_chunk + " " + sentence).strip() if current_chunk else sentence
    if current_chunk: chunks.append(current_chunk.strip())
    return chunks

def download_reference_audio(url: str) -> str:
    """Download reference audio from URL to temp file."""
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)

    try:
        urllib.request.urlretrieve(url, temp_file.name)
        return temp_file.name
    except Exception as e:
        os.unlink(temp_file.name)
        raise Exception(f"Failed to download reference audio: {e}")


def base64_to_audio_file(b64_data: str) -> str:
    """Convert base64 audio to temp file."""
    # Remove data URL prefix if present
    if "," in b64_data:
        b64_data = b64_data.split(",")[1]

    audio_bytes = base64.b64decode(b64_data)
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_file.write(audio_bytes)
    temp_file.close()
    return temp_file.name

def handler(job: dict) -> dict:
    job_input = job.get("input", {})
    text = job_input.get("text", "")
    if not text: return {"error": "No text provided"}


    ref_url = job_input.get("reference_audio_url")
    ref_audio_base64 = job_input.get("reference_audio_base64")
    temp = job_input.get("temperature", 0.7)
    exag = job_input.get("exaggeration", 1.0)
    speed = job_input.get("speed", 1.0)

    try:
        
        text_chunks = split_text_into_chunks(text)
        
        # Criar o tensor de silêncio (Padding)
        # 
        silence_samples = int(SAMPLE_RATE * SILENCE_DURATION)
        silence_tensor = torch.zeros((1, silence_samples))

        audio_list = []
        ref_path = None
        if ref_url:
            print(f"[Handler] Downloading reference audio from URL...")
            ref_path = download_reference_audio(ref_audio_url)
        elif ref_audio_base64:
            print(f"[Handler] Decoding reference audio from base64...")
            ref_path = base64_to_audio_file(ref_audio_base64)

        for i, chunk in enumerate(text_chunks):
            print(f"[Handler] Gerando chunk {i+1}/{len(text_chunks)}")
            
            with torch.no_grad():
                wav = model.generate(
                    text=chunk,
                    audio_prompt_path=ref_path,
                    temperature=temp,
                    exaggeration=exag
                )
            
            if wav.dim() == 1: wav = wav.unsqueeze(0)
            audio_list.append(wav.cpu())

            # Adiciona o silêncio após o chunk, exceto no último
            if i < len(text_chunks) - 1:
                audio_list.append(silence_tensor)

            if torch.cuda.is_available(): torch.cuda.empty_cache()

        # Concatenação final
        final_audio = torch.cat(audio_list, dim=1)

        # Ajuste de velocidade se necessário
        if speed != 1.0:
            import torchaudio.transforms as T
            resampler = T.Resample(orig_freq=int(SAMPLE_RATE * speed), new_freq=SAMPLE_RATE)
            final_audio = resampler(final_audio)

        # Normalização para evitar clipping
        final_audio = final_audio / (torch.max(torch.abs(final_audio)) + 1e-9) * 0.9

        # Conversão Base64
        buffer = io.BytesIO()
        ta.save(buffer, final_audio, SAMPLE_RATE, format="WAV")
        audio_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        if ref_path: os.unlink(ref_path)

        return {
            "audio_base64": audio_b64,
            "duration": final_audio.shape[1] / SAMPLE_RATE,
            "chunks": len(text_chunks)
        }

    except Exception as e:
        return {"error": str(e)}

print("[Handler] Pre-loading model...")

try:

    model = load_model()

except Exception as e:

    print(f"[Handler] Warning: Could not pre-load model: {e}")
    

runpod.serverless.start({"handler": handler})
