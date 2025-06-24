from flask import Flask, request, render_template
import os
from vad_utils import read_audio_mono, get_speech_segments
from faster_whisper import WhisperModel
import numpy as np

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Initialize model once
model = WhisperModel("small", device="cpu", compute_type="int8")

MAX_SEGMENT_DURATION_SEC = 30

def split_segment(segment, sample_rate, max_duration_sec=MAX_SEGMENT_DURATION_SEC):
    """
    Split audio segment into smaller chunks
    Args:
        segment: 1D pytorch tensor of audio samples (from vad)
        sample_rate: number of samples per second (16000)
        max_duration_sec: maximum duration of each chunk in seconds (30 seconds)
    
    """
    max_samples = int(max_duration_sec * sample_rate) # convert duration in seconds to samples
    for start in range(0, len(segment), max_samples): # split audio into non-overlapping segments of at most max_samples
        end = min(start + max_samples, len(segment))
        yield segment[start:end]

@app.route('/', methods=['GET', 'POST'])
def index():
    transcript = ""
    error_message = ""
    
    if request.method == 'POST':
        try:
            # Check if file was uploaded
            if 'audio' not in request.files:
                error_message = "No audio file uploaded"
                return render_template('index.html', transcript=transcript, error=error_message)
            
            file = request.files['audio']
            
            # Check if file was selected
            if file.filename == '':
                error_message = "No file selected"
                return render_template('index.html', transcript=transcript, error=error_message)
            
            if file:
                # Save uploaded file
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(file_path)

                try:
                    # Read audio file and extract speech segments
                    audio, sr = read_audio_mono(file_path)
                    segments = get_speech_segments(audio, sr)

                    # Check if any speech segments were found
                    if not segments:
                        error_message = "No speech detected in the audio file"
                        return render_template('index.html', transcript=transcript, error=error_message)

                    # Transcribe each speech segment
                    texts = []
                    for i, seg in enumerate(segments):
                        for j, chunk in enumerate(split_segment(seg, sr)):
                            # Skip chunks that are too long (2 minutes = 120 seconds)
                            chunk_duration = len(chunk) / sr
                            if chunk_duration > 120:
                                print(f"Skipping chunk {i+1}.{j+1} – too long ({chunk_duration:.2f} sec)")
                                continue

                            # Skip very short chunks (less than 0.1 seconds)
                            if chunk_duration < 0.1:
                                print(f"Skipping chunk {i+1}.{j+1} – too short ({chunk_duration:.2f} sec)")
                                continue

                            print(f"Processing chunk {i+1}.{j+1}: {chunk_duration:.2f} sec")
                            
                            # Convert to numpy array with proper dtype
                            seg_np = chunk.numpy().astype(np.float32)
                            
                            # Transcribe the chunk
                            segments_result, _ = model.transcribe(seg_np, language="en")
                            text = " ".join([segment.text.strip() for segment in segments_result])
                            
                            if text.strip():  # Only add non-empty transcriptions
                                texts.append(text.strip())

                    # Join all transcribed text
                    transcript = " ".join(texts)
                    
                    if not transcript.strip():
                        error_message = "No speech could be transcribed from the audio file"

                except Exception as e:
                    error_message = f"Error processing audio: {str(e)}"
                    print(f"Audio processing error: {e}")
                
                finally:
                    # Clean up uploaded file
                    try:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                        # Also remove converted WAV file if it exists
                        wav_path = os.path.splitext(file_path)[0] + '.wav'
                        if os.path.exists(wav_path):
                            os.remove(wav_path)
                    except Exception as e:
                        print(f"Error cleaning up files: {e}")
        
        except Exception as e:
            error_message = f"Unexpected error: {str(e)}"
            print(f"Unexpected error: {e}")
    
    return render_template('index.html', transcript=transcript, error=error_message)

if __name__ == "__main__":
    app.run(debug=True)