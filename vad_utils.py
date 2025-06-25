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

def get_speech_segments(audio_tensor, sample_rate, min_duration_sec=0.5, max_silence_sec=2.0):
    """
    Simplified and optimized speech segment extraction using Silero VAD.
    
    Args:
        audio_tensor (torch.Tensor): Mono audio tensor
        sample_rate (int): Sample rate (e.g. 16000)
        min_duration_sec (float): Minimum segment duration
        max_silence_sec (float): Maximum silence gap to merge segments
    
    Returns:
        List[torch.Tensor]: List of speech segments
    """
    
    # Early exit on empty input
    if len(audio_tensor) == 0:
        return []
    
    try:
        model = load_silero_vad()
    except Exception as e:
        raise RuntimeError(f"Failed to load Silero VAD model: {str(e)}")
    
    # Ensure tensor is float32
    if audio_tensor.dtype != torch.float32:
        audio_tensor = audio_tensor.float()
    
    print(f"Processing audio: {len(audio_tensor) / sample_rate:.1f} seconds")
    
    try:
        # Get speech timestamps directly on the full audio
        # Silero VAD can handle long audio efficiently
        timestamps = get_speech_timestamps(
            audio_tensor, 
            model, 
            sampling_rate=sample_rate, 
            return_seconds=False,
            min_speech_duration_ms=int(min_duration_sec * 1000),
            min_silence_duration_ms=100,  # 100ms minimum silence
            window_size_samples=1536,  # Optimized for 16kHz
            speech_pad_ms=100  # Add 100ms padding around speech
        )
        
    except Exception as e:
        print(f"VAD processing error: {str(e)}")
        return []
    
    if not timestamps:
        print("No speech detected")
        return []
    
    print(f"Found {len(timestamps)} initial speech segments")
    
    # Merge nearby segments
    merged_timestamps = merge_close_segments(timestamps, sample_rate, max_silence_sec)
    print(f"After merging: {len(merged_timestamps)} segments")
    
    # Extract segments from audio
    segments = []
    for i, ts in enumerate(merged_timestamps):
        start, end = ts['start'], ts['end']
        duration = (end - start) / sample_rate
        
        # Skip segments that are too short
        if duration < min_duration_sec:
            continue
            
        # Extract segment
        segment = audio_tensor[start:end]
        
        if len(segment) > 0:
            segments.append(segment)
            print(f"Segment {i+1}: {duration:.1f}s")
    
    print(f"Final segments: {len(segments)}")
    return segments

def merge_close_segments(timestamps, sample_rate, max_gap_sec=1.0):
    """
    Merge segments that are close together
    """
    if not timestamps:
        return []
    
    merged = []
    current = timestamps[0].copy()
    
    for ts in timestamps[1:]:
        gap_sec = (ts['start'] - current['end']) / sample_rate
        
        if gap_sec <= max_gap_sec:
            # Merge segments
            current['end'] = ts['end']
        else:
            # Keep current segment and start new one
            merged.append(current)
            current = ts.copy()
    
    # Don't forget the last segment
    merged.append(current)
    
    return merged