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
def generate_voice(text, audio_file, speed, inference_steps, cfg_value, denoise):
    try:
        if audio_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_ref:
                tmp_ref.write(audio_file.read())
                wav = model.generate(
                    text, 
                    reference_wav_path=tmp_ref.name,
                    speed=speed,
                    inference_timesteps=inference_steps,
                    cfg_value=cfg_value,
                    denoise=denoise
                )
        else:
            wav = model.generate(
                text,
                speed=speed,
                inference_timesteps=inference_steps,
                cfg_value=cfg_value,
                denoise=denoise
            )
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_out:
            sf.write(tmp_out.name, wav, 48000)
            return tmp_out.name
    except Exception as e:
        return f"Error: {str(e)}"

# ============================================
# TIMEOUT HANDLING
# ============================================
def generate_with_timeout(text, audio_file, speed, inference_steps, cfg_value, denoise, timeout=120):
    def handler(signum, frame):
        raise TimeoutError("Generation timed out!")
    
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout)
    
    try:
        result = generate_voice(text, audio_file, speed, inference_steps, cfg_value, denoise)
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
    
    # ============================================
    # GENERATION SETTINGS (ADVANCED)
    # ============================================
    with st.expander("⚙️ Generation Settings (Advanced)"):
        col1, col2 = st.columns(2)
        
        with col1:
            speed = st.slider(
                "Speed", 
                min_value=0.5, 
                max_value=2.0, 
                value=1.0, 
                step=0.1,
                help="1.0 = normal. >1 faster, <1 slower"
            )
            inference_steps = st.slider(
                "Inference Steps", 
                min_value=4, 
                max_value=64, 
                value=10, 
                step=1,
                help="Lower = faster, higher = better quality"
            )
        
        with col2:
            cfg_value = st.slider(
                "Guidance Scale (CFG)", 
                min_value=0.0, 
                max_value=4.0, 
                value=1.8, 
                step=0.1,
                help="Voice similarity control"
            )
            denoise = st.checkbox(
                "Denoise", 
                value=True,
                help="Remove background noise"
            )
    
    submitted = st.form_submit_button("🎵 Generate Voice")

# ============================================
# GENERATE
# ============================================
if submitted and text:
    with st.spinner("Generating voice (1-2 minutes)..."):
        result = generate_with_timeout(text, audio_file, speed, inference_steps, cfg_value, denoise)
        
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
