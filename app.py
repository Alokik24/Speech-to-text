from flask import Flask, request, render_template
import os
from vad_utils import read_audio_mono, get_speech_segments
from faster_whisper import WhisperModel
import numpy as np
import time

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size


model = None
model_loaded = False

@app.before_request
def load_model_once():
    global model, model_loaded
    if not model_loaded:
        print("Loading Whisper model...")
        model = WhisperModel("small", device="cpu", compute_type="int8")
        model_loaded = True

def split_long_segment(segment, sample_rate, max_duration_sec=150):
    """
    Split a segment if it's longer than max_duration_sec
    """
    segment_duration = len(segment) / sample_rate
    
    if segment_duration <= max_duration_sec:
        yield segment
        return
    
    # Split into chunks
    max_samples = int(max_duration_sec * sample_rate)
    overlap_samples = int(0.5 * sample_rate)  # 0.5 second overlap
    
    for start in range(0, len(segment), max_samples - overlap_samples):
        end = min(start + max_samples, len(segment))
        chunk = segment[start:end]
        
        # Skip very short chunks at the end
        if len(chunk) >= int(0.5 * sample_rate):  # At least 0.5 seconds
            yield chunk

@app.route('/', methods=['GET', 'POST'])
def index():
    transcript = ""
    error_message = ""
    
    if request.method == 'POST':
        start_time = time.time()
        
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
                print(f"Processing file: {file.filename}")

                try:
                    # Read audio file
                    print("Loading audio...")
                    audio, sr = read_audio_mono(file_path)
                    audio_duration = len(audio) / sr
                    print(f"Audio loaded: {audio_duration:.1f} seconds")
                    
                    # Extract speech segments using VAD
                    print("Extracting speech segments...")
                    segments = get_speech_segments(audio, sr, min_duration_sec=1.0)

                    # Check if any speech segments were found
                    if not segments:
                        error_message = "No speech detected in the audio file"
                        return render_template('index.html', transcript=transcript, error=error_message)

                    # Transcribe each speech segment
                    if app.debug:
                        print(f"Transcribing {len(segments)} segments...")
                    texts = []
                    total_segments = 0
                    
                    for i, segment in enumerate(segments):
                        segment_duration = len(segment) / sr
                        if app.debug:
                            print(f"Processing segment {i+1}/{len(segments)}: {segment_duration:.1f}s")
                        
                        # Split long segments
                        for j, chunk in enumerate(split_long_segment(segment, sr, max_duration_sec=120)):
                            chunk_duration = len(chunk) / sr
                            
                            # Skip very short chunks
                            if chunk_duration < 1:
                                continue
                                
                            total_segments += 1
                            if app.debug:
                                print(f"  Chunk {j+1}: {chunk_duration:.1f}s")
                            
                            # Convert to numpy array for Whisper
                            chunk_np = chunk.contiguous().numpy().astype(np.float32)
                            
                            try:
                                # Transcribe
                                segments_result, info = model.transcribe(
                                    chunk_np, 
                                    language="en",
                                    condition_on_previous_text=True,
                                    vad_filter=False,  # We already did VAD
                                    word_timestamps=False,  # Faster without word timestamps
                                    beam_size=1
                                )
                                
                                # Extract text
                                text_parts = []
                                for segment in segments_result:
                                    if segment.text.strip():
                                        text_parts.append(segment.text.strip())
                                
                                if text_parts:
                                    chunk_text = " ".join(text_parts)
                                    texts.append(chunk_text)
                                    print(f"    -> {len(chunk_text)} chars")
                                
                            except Exception as e:
                                print(f"    -> Error transcribing chunk: {e}")
                                continue

                    # Join all transcribed text
                    transcript = " ".join(texts)
                    
                    # Processing time
                    processing_time = time.time() - start_time
                    print(f"Processing completed in {processing_time:.1f}s")
                    print(f"Audio duration: {audio_duration:.1f}s, Processing time: {processing_time:.1f}s")
                    print(f"Speed ratio: {audio_duration/processing_time:.1f}x")
                    
                    if not transcript.strip():
                        error_message = "No speech could be transcribed from the audio file"
                    else:
                        print(f"Transcript length: {len(transcript)} characters")

                except Exception as e:
                    error_message = f"Error processing audio: {str(e)}"
                    print(f"Audio processing error: {e}")
                    import traceback
                    traceback.print_exc()
                
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
            import traceback
            traceback.print_exc()
    
    return render_template('index.html', transcript=transcript, error=error_message)

if __name__ == "__main__":
    app.run(debug=True)