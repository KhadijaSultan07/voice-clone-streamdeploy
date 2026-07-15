
import streamlit as st
import os
import tempfile
import soundfile as sf
from voxcpm import VoxCPM

# Page config
st.set_page_config(page_title="Urdu Voice Cloning", page_icon="🎙️")

st.title("🎙️ Urdu Voice Cloning Studio")
st.markdown("30+ Languages | 48kHz Quality | Free")

# Model load with caching
@st.cache_resource
def load_model():
    return VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False)

try:
    model = load_model()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Model load error: {str(e)}")
    st.stop()

def generate_voice(text, audio_file):
    try:
        if audio_file is not None:
            temp_ref = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            temp_ref.write(audio_file.read())
            wav = model.generate(text, reference_wav_path=temp_ref.name, cfg_value=1.8, inference_timesteps=10, denoise=True)
        else:
            wav = model.generate(text, cfg_value=1.8, inference_timesteps=10, denoise=True)
        temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        sf.write(temp_path.name, wav, 48000)
        return temp_path.name
    except Exception as e:
        return f"Error: {str(e)}"

# UI
with st.form("voice_form"):
    text = st.text_area("📝 Text", height=100, placeholder="Urdu, English, Arabic...")
    audio_file = st.file_uploader("🎤 Reference Audio (Optional)", type=["wav", "mp3"])
    submitted = st.form_submit_button("🎵 Generate")

if submitted and text:
    with st.spinner("Generating (1-2 minutes)..."):
        result = generate_voice(text, audio_file)
        if result.startswith("Error"):
            st.error(result)
        else:
            st.audio(result, format="audio/wav")
            with open(result, "rb") as f:
                st.download_button("⬇️ Download", f, file_name="voice.wav")
