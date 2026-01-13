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
import string
import difflib
import gc

# --- Configurações ---
SAMPLE_RATE = 24000
SILENCE_DURATION = 0.15
OUTPUT_DIR = "/runpod-volume"
TEMP_DIR = "/tmp/tts_temp"  # Local temp storage (deleted after execution)
WHISPER_THRESHOLD = 0.85  # Minimum similarity score to accept audio
DEFAULT_CHUNK_SIZE = 200  # Default chunk size in characters
MAX_RETRY_ATTEMPTS = 3    # Max attempts per chunk if validation fails

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# Global model instances
tts_model = None
whisper_model = None


def load_tts_model():
    """Load the ChatterboxTTS model."""
    global tts_model
    if tts_model is not None:
        return tts_model
    from chatterbox.tts import ChatterboxTTS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts_model = ChatterboxTTS.from_pretrained(device=device)
    print(f"[Handler] TTS model loaded on {device}")
    return tts_model


def load_whisper_model():
    """Load the Faster-Whisper model for validation."""
    global whisper_model
    if whisper_model is not None:
        return whisper_model
    
    from faster_whisper import WhisperModel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Try different compute types based on device
    if device == "cuda":
        compute_types = ["float16", "int8_float16", "int8"]
    else:
        compute_types = ["int8", "float32"]
    
    for ct in compute_types:
        try:
            print(f"[Handler] Loading Whisper model (device={device}, compute_type={ct})...")
            whisper_model = WhisperModel("base", device=device, compute_type=ct)
            print(f"[Handler] Whisper model loaded successfully")
            return whisper_model
        except Exception as e:
            print(f"[Handler] Failed to load Whisper with {ct}: {e}")
    
    raise RuntimeError("Failed to load Whisper model with any compute type")


def normalize_for_compare(text: str) -> str:
    """
    Normalize text for comparison by removing punctuation,
    converting to lowercase, and normalizing whitespace.
    """
    # Replace dashes with spaces
    text = re.sub(r'[–—-]', ' ', text)
    # Remove all punctuation
    text = re.sub(rf"[{re.escape(string.punctuation)}]", '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.lower().strip()


def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate similarity ratio between two texts."""
    t1 = normalize_for_compare(text1)
    t2 = normalize_for_compare(text2)
    return difflib.SequenceMatcher(None, t1, t2).ratio()


def transcribe_audio(audio_path: str) -> str:
    """Transcribe audio file using Whisper."""
    model = load_whisper_model()
    segments, _ = model.transcribe(audio_path)
    transcribed = "".join([seg.text for seg in segments]).strip()
    return transcribed


def validate_audio_with_whisper(audio_path: str, expected_text: str) -> tuple[bool, float, str]:
    """
    Validate that the generated audio matches the expected text.
    Returns: (is_valid, score, transcribed_text)
    """
    try:
        transcribed = transcribe_audio(audio_path)
        score = calculate_similarity(transcribed, expected_text)
        is_valid = score >= WHISPER_THRESHOLD
        
        print(f"[Whisper] Score: {score:.3f} | Valid: {is_valid}")
        print(f"[Whisper] Expected: '{expected_text[:50]}...'")
        print(f"[Whisper] Got: '{transcribed[:50]}...'")
        
        return is_valid, score, transcribed
    except Exception as e:
        print(f"[Whisper] Transcription error: {e}")
        return False, 0.0, ""


def split_text_into_chunks(text: str, chunk_size: int = 200) -> list[str]:
    """Split text into chunks at sentence boundaries."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if (len(current_chunk) + len(sentence) + 1 > chunk_size) and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk = (current_chunk + " " + sentence).strip() if current_chunk else sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


def base64_to_audio_file(b64_data: str) -> str:
    """Convert base64 audio to a temporary .wav file."""
    if "," in b64_data:
        b64_data = b64_data.split(",")[1]
    audio_bytes = base64.b64decode(b64_data)
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=TEMP_DIR)
    temp_file.write(audio_bytes)
    temp_file.close()
    return temp_file.name


def generate_chunk_with_validation(
    model,
    chunk_text: str,
    chunk_index: int,
    ref_path: str | None,
    temp: float,
    exag: float,
    bypass_whisper: bool = False
) -> tuple[torch.Tensor, dict]:
    """
    Generate audio for a chunk with Whisper validation.
    Retries up to MAX_RETRY_ATTEMPTS if validation fails.
    Returns: (audio_tensor, metadata)
    """
    best_audio = None
    best_score = 0.0
    best_transcription = ""
    
    for attempt in range(1, MAX_RETRY_ATTEMPTS + 1):
        print(f"[Chunk {chunk_index}] Attempt {attempt}/{MAX_RETRY_ATTEMPTS}")
        
        # Generate audio
        with torch.no_grad():
            wav = model.generate(
                text=chunk_text,
                audio_prompt_path=ref_path,
                temperature=temp,
                exaggeration=exag
            )
        
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        
        # If bypassing Whisper, return immediately
        if bypass_whisper:
            return wav.cpu(), {
                "validated": False,
                "score": None,
                "attempts": attempt,
                "transcription": None
            }
        
        # Save to temp file for Whisper validation
        temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=TEMP_DIR)
        ta.save(temp_audio.name, wav.cpu(), SAMPLE_RATE)
        temp_audio.close()
        
        try:
            # Validate with Whisper
            is_valid, score, transcription = validate_audio_with_whisper(
                temp_audio.name, 
                chunk_text
            )
            
            # Track best attempt
            if score > best_score:
                best_score = score
                best_audio = wav.cpu()
                best_transcription = transcription
            
            # If valid, return immediately
            if is_valid:
                print(f"[Chunk {chunk_index}] ✓ Passed validation on attempt {attempt}")
                return wav.cpu(), {
                    "validated": True,
                    "score": score,
                    "attempts": attempt,
                    "transcription": transcription
                }
            else:
                print(f"[Chunk {chunk_index}] ✗ Failed validation (score: {score:.3f}), retrying...")
        
        finally:
            # Cleanup temp file
            if os.path.exists(temp_audio.name):
                os.unlink(temp_audio.name)
        
        # Clear CUDA cache between attempts
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # All attempts failed, return best one
    print(f"[Chunk {chunk_index}] ⚠ All attempts failed, using best (score: {best_score:.3f})")
    return best_audio, {
        "validated": False,
        "score": best_score,
        "attempts": MAX_RETRY_ATTEMPTS,
        "transcription": best_transcription
    }


def handler(job: dict) -> dict:
    """Main RunPod handler function."""
    model = load_tts_model()
    job_id = job.get("id")
    job_input = job.get("input", {})
    text = job_input.get("text", "")
    
    if not text:
        return {"error": "No text provided"}
    
    # Parameters
    ref_url = job_input.get("reference_audio_url")
    ref_b64 = job_input.get("reference_audio_base64")
    temp = job_input.get("temperature", 0.7)
    exag = job_input.get("exaggeration", 1.0)
    speed = job_input.get("speed", 1.0)
    bypass_whisper = job_input.get("bypass_whisper", False)
    chunk_size = job_input.get("chunk_size", DEFAULT_CHUNK_SIZE)
    
    ref_path = None
    chunk_metadata = []
    
    try:
        # Split text into chunks
        text_chunks = split_text_into_chunks(text, chunk_size=chunk_size)
        print(f"[Handler] Processing {len(text_chunks)} chunks")
        
        # Prepare silence tensor
        silence_samples = int(SAMPLE_RATE * SILENCE_DURATION)
        silence_tensor = torch.zeros((1, silence_samples))
        audio_list = []
        
        # Download/decode reference audio
        if ref_url:
            temp_f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=TEMP_DIR)
            urllib.request.urlretrieve(ref_url, temp_f.name)
            ref_path = temp_f.name
        elif ref_b64:
            ref_path = base64_to_audio_file(ref_b64)
        
        # Generate and validate each chunk
        total_attempts = 0
        validated_chunks = 0
        
        for i, chunk in enumerate(text_chunks):
            print(f"\n[Handler] === Chunk {i+1}/{len(text_chunks)} ===")
            print(f"[Handler] Text: '{chunk[:80]}...'")
            
            audio, metadata = generate_chunk_with_validation(
                model=model,
                chunk_text=chunk,
                chunk_index=i + 1,
                ref_path=ref_path,
                temp=temp,
                exag=exag,
                bypass_whisper=bypass_whisper
            )
            
            audio_list.append(audio)
            chunk_metadata.append({
                "chunk_index": i + 1,
                "text": chunk[:100] + "..." if len(chunk) > 100 else chunk,
                **metadata
            })
            
            total_attempts += metadata["attempts"]
            if metadata.get("validated"):
                validated_chunks += 1
            
            # Add silence between chunks
            if i < len(text_chunks) - 1:
                audio_list.append(silence_tensor)
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Concatenate all audio
        final_audio = torch.cat(audio_list, dim=1)
        
        # Apply speed adjustment
        if speed != 1.0:
            import torchaudio.transforms as T
            resampler = T.Resample(
                orig_freq=int(SAMPLE_RATE * speed), 
                new_freq=SAMPLE_RATE
            )
            final_audio = resampler(final_audio)
        
        # Normalize audio
        final_audio = final_audio / (torch.max(torch.abs(final_audio)) + 1e-9) * 0.9
        
        # Save output file
        file_name = f"output_{job_id}.wav"
        output_path = os.path.join(OUTPUT_DIR, file_name)
        ta.save(output_path, final_audio, SAMPLE_RATE)
        
        # Calculate validation stats
        validation_rate = (validated_chunks / len(text_chunks) * 100) if text_chunks else 0
        
        return {
            "audio_url": output_path,
            "file_name": file_name,
            "duration": final_audio.shape[1] / SAMPLE_RATE,
            "chunks": len(text_chunks),
            "status": "completed",
            "validation": {
                "enabled": not bypass_whisper,
                "threshold": WHISPER_THRESHOLD,
                "validated_chunks": validated_chunks,
                "total_chunks": len(text_chunks),
                "validation_rate": f"{validation_rate:.1f}%",
                "total_attempts": total_attempts,
                "chunk_details": chunk_metadata
            }
        }
    
    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }


# Pre-loading models
print("[Handler] Pre-loading models...")
try:
    load_tts_model()
    load_whisper_model()
    print("[Handler] All models loaded successfully")
except Exception as e:
    print(f"[Handler] Warning: Could not pre-load models: {e}")

runpod.serverless.start({"handler": handler})
