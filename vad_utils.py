import torchaudio
import torch
import os
import subprocess
import shutil
from silero_vad import get_speech_timestamps, load_silero_vad

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def find_ffmpeg():
    """Find FFmpeg executable in various locations"""
    # First check local project folder
    local_ffmpeg = os.path.join(SCRIPT_DIR, "ffmpeg", "bin", "ffmpeg.exe")
    if os.path.exists(local_ffmpeg):
        return local_ffmpeg
    
    # Check if ffmpeg is in PATH
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    
    # Common Windows installation paths
    common_paths = [
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe",
        # Add path from your current setup
        r"C:\Users\aloki\Downloads\ffmpeg-2025-06-23-git-e6298e0759-essentials_build\ffmpeg-2025-06-23-git-e6298e0759-essentials_build\bin\ffmpeg.exe"
    ]
    
    for path in common_paths:
        if os.path.exists(path):
            return path
    
    # If not found, return None
    return None

FFMPEG_BIN = find_ffmpeg()

def convert_to_wav(input_file):
    """Convert audio file to WAV format using ffmpeg"""
    if FFMPEG_BIN is None:
        raise RuntimeError(
            "FFmpeg not found. Please:\n"
            "1. Download FFmpeg from https://ffmpeg.org/download.html\n"
            "2. Extract it to your project folder as 'ffmpeg/bin/ffmpeg.exe'\n"
            "3. Or install it system-wide and add to PATH"
        )
    
    wav_path = os.path.splitext(input_file)[0] + '.wav'
    command = [
        FFMPEG_BIN,
        "-y",  # overwrite if exists
        "-i", input_file,
        "-ar", "16000",  # target sample rate for Whisper/VAD
        "-ac", "1",      # mono channel
        "-acodec", "pcm_s16le",  # specify codec for better compatibility
        wav_path
    ]
    
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                              check=True, timeout=300)  # 5 minute timeout
    except subprocess.TimeoutExpired:
        raise RuntimeError("FFmpeg conversion timed out")
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode() if e.stderr else "Unknown FFmpeg error"
        raise RuntimeError(f"FFmpeg failed to convert the file: {error_msg}")
    
    if not os.path.exists(wav_path):
        raise RuntimeError("FFmpeg conversion failed - output file not created")
    
    return wav_path

def read_audio_mono(file_path, target_sr=16000):
    """Read audio file and convert to mono with target sample rate"""
    original_path = file_path
    
    # Convert to WAV if not already
    if not file_path.lower().endswith('.wav'):
        file_path = convert_to_wav(file_path)
    
    try:
        waveform, sr = torchaudio.load(file_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load audio file: {str(e)}")
    
    # Resample if necessary
    if sr != target_sr:
        try:
            waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        except Exception as e:
            raise RuntimeError(f"Failed to resample audio: {str(e)}")
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # Ensure 1D tensor
    if waveform.dim() > 1:
        waveform = waveform.squeeze()
    
    return waveform, target_sr

def split_audio_tensor(audio_tensor, sample_rate, chunk_duration_sec=60):
    """
    Splits a long audio tensor into smaller chunks (default 60s).
    Returns a list of chunks.
    """
    if len(audio_tensor) == 0:
        return []
    
    chunk_len = int(sample_rate * chunk_duration_sec)
    chunks = []
    for i in range(0, len(audio_tensor), chunk_len):
        chunk = audio_tensor[i:i+chunk_len]
        if len(chunk) > 0:  # Only add non-empty chunks
            chunks.append(chunk)
    return chunks

def merge_segments(timestamps, sample_rate, max_silence_sec=0.5):
    """
    Merge segments that are close to each other (< max_silence_sec gap)
    """
    if not timestamps:
        return []
    
    merged = []
    current = timestamps[0].copy() 
    
    for ts in timestamps[1:]:
        gap = (ts['start'] - current['end']) / sample_rate
        if gap <= max_silence_sec:
            current['end'] = ts['end']  # merge with current
        else:
            merged.append(current)
            current = ts.copy()    
    
    merged.append(current)  # append the last segment
    return merged

def get_speech_segments(audio_tensor, sample_rate, min_duration_sec=0.5, max_silence_sec=0.5, chunk_duration_sec=60):
    """
    Uses Silero VAD to extract speech-only segments from long audio by chunking first.

    Args:
        audio_tensor (torch.Tensor): Mono audio
        sample_rate (int): e.g. 16000
        min_duration_sec (float): Minimum length of accepted segment
        max_silence_sec (float): Maximum silence between mergeable segments
        chunk_duration_sec (float): Duration of audio to process at once

    Returns: List of 1D torch.Tensor segments
    """

    # Early exit on empty input
    if len(audio_tensor) == 0:
        return []
    
    try:
        model = load_silero_vad()
    except Exception as e:
        raise RuntimeError(f"Failed to load Silero VAD model: {str(e)}")
    
    chunks = split_audio_tensor(audio_tensor, sample_rate, chunk_duration_sec)
    
    if not chunks:
        return []
    
    all_segments = []
    sample_offset = 0  # keeps track of where each chunk starts

    for chunk_idx, chunk in enumerate(chunks):
        try:
            # Ensure chunk is float32 and properly shaped for VAD
            if chunk.dtype != torch.float32:
                chunk = chunk.float()
            
            # Get speech timestamps for this chunk
            timestamps = get_speech_timestamps( # Returns a list of {"start": ..., "end": ...} timestamps in samples, not seconds
                chunk, 
                model, 
                sampling_rate=sample_rate, 
                return_seconds=False,
                min_speech_duration_ms=int(min_duration_sec * 1000), # Filters out very short speech blips.
                min_silence_duration_ms=50  # 50ms minimum silence, Controls minimum gap between segments.
            )
            
            # merge close segments
            timestamps = merge_segments(timestamps, sample_rate, max_silence_sec=max_silence_sec)

            # extract segments from the original audio
            for ts in timestamps:
                start, end = ts['start'], ts['end']
                duration = (end - start) / sample_rate
                
                if duration >= min_duration_sec:
                    global_start = sample_offset + start
                    global_end = sample_offset + end # ensures timestamps are relative to the entire audio file, not chunk.
                    
                    # don't go beyond the audio length
                    global_end = min(global_end, len(audio_tensor))
                    
                    if global_start < len(audio_tensor) and global_end > global_start:
                        segment = audio_tensor[global_start:global_end]
                        if len(segment) > 0:
                            all_segments.append(segment)

        except Exception as e:
            print(f"Error processing chunk {chunk_idx}: {str(e)}")
            continue
        
        sample_offset += len(chunk)

    return all_segments