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

# --- Configurações de Áudio e Caminho ---
SAMPLE_RATE = 24000
SILENCE_DURATION = 0.15 
OUTPUT_DIR = "/runpod-volume" 

# Garante que a pasta de destino exista
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Global model instance
tts_model = None

def load_model():
    global tts_model
    if tts_model is not None: return tts_model
    from chatterbox.tts import ChatterboxTTS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts_model = ChatterboxTTS.from_pretrained(device=device)
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

def base64_to_audio_file(b64_data: str) -> str:
    """Converte áudio em base64 para um arquivo temporário .wav"""
    if "," in b64_data:
        b64_data = b64_data.split(",")[1]
    audio_bytes = base64.b64decode(b64_data)
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_file.write(audio_bytes)
    temp_file.close()
    return temp_file.name

def handler(job: dict) -> dict:
    model = load_model()
    job_id = job.get("id") # ID único do RunPod para evitar sobrescrita
    job_input = job.get("input", {})
    text = job_input.get("text", "")
    
    if not text: return {"error": "No text provided"}

    # Parâmetros do modelo (Preservados)
    ref_url = job_input.get("reference_audio_url")
    ref_b64 = job_input.get("reference_audio_base64")
    temp = job_input.get("temperature", 0.7)
    exag = job_input.get("exaggeration", 1.0)
    speed = job_input.get("speed", 1.0)

    ref_path = None
    try:
        # Lógica de Chunking e Silêncio (Preservada)
        text_chunks = split_text_into_chunks(text)
        silence_samples = int(SAMPLE_RATE * SILENCE_DURATION)
        silence_tensor = torch.zeros((1, silence_samples))
        audio_list = []

        # Download/Decodificação de referência (Preservada)
        if ref_url:
            temp_f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            urllib.request.urlretrieve(ref_url, temp_f.name)
            ref_path = temp_f.name
        elif ref_b64:
            ref_path = base64_to_audio_file(ref_b64)

        # Geração dos chunks (Lógica Original Intocada)
        for i, chunk in enumerate(text_chunks):
            with torch.no_grad():
                wav = model.generate(
                    text=chunk,
                    audio_prompt_path=ref_path,
                    temperature=temp,
                    exaggeration=exag
                )
            
            if wav.dim() == 1: wav = wav.unsqueeze(0)
            audio_list.append(wav.cpu())

            if i < len(text_chunks) - 1:
                audio_list.append(silence_tensor)

            if torch.cuda.is_available(): torch.cuda.empty_cache()

        # Concatenação e Processamento final (Preservada)
        final_audio = torch.cat(audio_list, dim=1)

        # Ajuste de velocidade (Preservada)
        if speed != 1.0:
            import torchaudio.transforms as T
            resampler = T.Resample(orig_freq=int(SAMPLE_RATE * speed), new_freq=SAMPLE_RATE)
            final_audio = resampler(final_audio)

        # Normalização (Preservada)
        final_audio = final_audio / (torch.max(torch.abs(final_audio)) + 1e-9) * 0.9

        # --- NOVA LÓGICA DE OUTPUT (SEM BASE64) ---
        file_name = f"output_{job_id}.wav"
        output_path = os.path.join(OUTPUT_DIR, file_name)
        
        # Salva o arquivo fisicamente no volume
        ta.save(output_path, final_audio, SAMPLE_RATE)

        # Retorna os detalhes do arquivo e metadados
        return {
            "audio_url": output_path, # Caminho para acesso no volume
            "file_name": file_name,
            "duration": final_audio.shape[1] / SAMPLE_RATE,
            "chunks": len(text_chunks),
            "status": "completed"
        }

    except Exception as e:
        return {"error": str(e)}
    
    finally:
        # Cleanup do arquivo temporário de referência (Preservada)
        if ref_path and os.path.exists(ref_path):
            os.unlink(ref_path)

# Pre-loading
print("[Handler] Pre-loading model...")
try:
   load_model()
except Exception as e:
    print(f"[Handler] Warning: Could not pre-load model: {e}")

runpod.serverless.start({"handler": handler})
