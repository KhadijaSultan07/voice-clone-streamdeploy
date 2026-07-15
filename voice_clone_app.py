import streamlit as st
import os
import tempfile
import soundfile as sf
from voxcpm import VoxCPM
import signal
import time

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(page_title="Urdu Voice Cloning", page_icon="🎙️")
st.title("🎙️ Urdu Voice Cloning Studio")
st.markdown("30+ Languages | 48kHz Quality | Free")

# ============================================
# LOAD MODEL
# ============================================
@st.cache_resource
def load_model():
    return VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False)

try:
    model = load_model()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Model load error: {str(e)}")
    st.stop()

# ============================================
# GENERATE VOICE FUNCTION
# ============================================
def generate_voice(text, audio_file):
    try:
        if audio_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_ref:
                tmp_ref.write(audio_file.read())
                wav = model.generate(
                    text, 
                    reference_wav_path=tmp_ref.name,
                    cfg_value=1.5,
                    inference_timesteps=5
                )
        else:
            wav = model.generate(
                text,
                cfg_value=1.5,
                inference_timesteps=5
            )
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_out:
            sf.write(tmp_out.name, wav, 48000)
            return tmp_out.name
    except Exception as e:
        return f"Error: {str(e)}"

# ============================================
# TIMEOUT HANDLING
# ============================================
def generate_with_timeout(text, audio_file, timeout=120):
    def handler(signum, frame):
        raise TimeoutError("Generation timed out!")
    
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout)
    
    try:
        result = generate_voice(text, audio_file)
        signal.alarm(0)
        return result
    except TimeoutError:
        return "Error: Generation took too long. Please try shorter text."

# ============================================
# UI
# ============================================
with st.form("voice_form"):
    text = st.text_area(
        "📝 Text", 
        height=100, 
        placeholder="Urdu, English, Arabic... kisi bhi language mein text likhein..."
    )
    audio_file = st.file_uploader(
        "🎤 Reference Audio (Optional)", 
        type=["wav", "mp3"],
        help="Voice cloning ke liye reference audio upload karein"
    )
    submitted = st.form_submit_button("🎵 Generate Voice")

# ============================================
# GENERATE
# ============================================
if submitted and text:
    with st.spinner("Generating voice (1-2 minutes)..."):
        result = generate_with_timeout(text, audio_file, timeout=120)
        
        if result.startswith("Error"):
            st.error(result)
        else:
            st.success("✅ Voice generated successfully!")
            st.audio(result, format="audio/wav")
            
            with open(result, "rb") as f:
                st.download_button(
                    "⬇️ Download Audio",
                    f,
                    file_name="voice.wav",
                    mime="audio/wav"
                )
