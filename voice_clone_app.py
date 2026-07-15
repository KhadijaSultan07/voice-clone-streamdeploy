import streamlit as st
import os
import tempfile
import soundfile as sf
from voxcpm import VoxCPM

st.set_page_config(page_title="Urdu Voice Cloning", page_icon="🎙️")
st.title("🎙️ Urdu Voice Cloning Studio")
st.markdown("30+ Languages | 48kHz Quality | Free")

@st.cache_resource
def load_model():
    return VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False)

try:
    model = load_model()
    st.success("✅ Model loaded!")
except Exception as e:
    st.error(f"❌ Error: {str(e)}")
    st.stop()

def generate_voice(text, audio_file):
    try:
        if audio_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_ref:
                tmp_ref.write(audio_file.read())
                wav = model.generate(text, reference_wav_path=tmp_ref.name)
        else:
            wav = model.generate(text, cfg_value=1.5, inference_timesteps=5)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_out:
            sf.write(tmp_out.name, wav, 48000)
            return tmp_out.name
    except Exception as e:
        return f"Error: {str(e)}"

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
            st.audio(result)
            with open(result, "rb") as f:
                st.download_button("⬇️ Download", f, file_name="voice.wav")
