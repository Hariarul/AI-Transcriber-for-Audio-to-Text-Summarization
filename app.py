import streamlit as st
import yt_dlp
import ffmpeg
import io
import soundfile as sf
import numpy as np
import whisper
from transformers import BartForConditionalGeneration, BartTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Function to extract audio from video
def extract_audio_from_video(video_input):
    try:
        if video_input.startswith("http"):  # YouTube URL
            ydl_opts = {
                "format": "bestaudio/best",
                "quiet": True,
                "no_warnings": True,
                "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
                "outtmpl": "temp_audio.%(ext)s",
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info_dict = ydl.extract_info(video_input, download=True)
                audio_path = ydl.prepare_filename(info_dict).replace(info_dict["ext"], "wav")
        else:  # Local file
            audio_path = "temp_audio.wav"
            ffmpeg.input(video_input).output(audio_path, format="wav").run(overwrite_output=True)

        audio_data, sample_rate = sf.read(audio_path)
        return audio_data, sample_rate, audio_path
    except Exception as e:
        raise RuntimeError(f"❌ Error extracting audio: {e}")

# Function to transcribe audio
def transcribe_audio(audio_path, multilingual_model=False):
    try:
        model = whisper.load_model("small.en.pt" if not multilingual_model else "small.pt")
        audio = whisper.load_audio(audio_path)
        result = model.transcribe(audio, fp16=False)
        return result["text"]
    except Exception as e:
        raise RuntimeError(f"❌ Error during transcription: {e}")

# Function to summarize text
def summarize_text(text, model, tokenizer):
    try:
        inputs = tokenizer(text, max_length=1024, return_tensors="pt", truncation=True)
        summary_ids = model.generate(inputs["input_ids"], max_length=150, min_length=50, length_penalty=2.0, num_beams=4)
        return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    except Exception as e:
        return f"⚠️ Error summarizing text: {e}"

# Function to extract topic
def extract_topic(text):
    if len(text.strip()) == 0:
        return "⚠️ The text provided is empty or too short to extract a topic."
    
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5)
    try:
        X = vectorizer.fit_transform([text])
        feature_names = np.array(vectorizer.get_feature_names_out())
        return ", ".join(feature_names)
    except ValueError as e:
        return f"⚠️ Error during topic extraction: {e}"

# Streamlit UI
st.set_page_config(page_title="AI Video Transcriber", layout="wide")
st.title("🎙️ AI-Powered Video Transcription & Summarization")
st.markdown("Upload a video file or enter a YouTube URL to transcribe and summarize the content.")

# Sidebar for input options
st.sidebar.header("Input Options")
input_type = st.sidebar.radio("Select Input Type", ("YouTube URL", "Upload Video File"))

video_input = None
temp_video_path = None

if input_type == "YouTube URL":
    video_input = st.sidebar.text_input("Enter YouTube Video URL:")
elif input_type == "Upload Video File":
    uploaded_file = st.sidebar.file_uploader("Upload a video file", type=["mp4", "mkv", "avi", "mov"])
    if uploaded_file:
        temp_video_path = f"temp_uploaded_video.{uploaded_file.name.split('.')[-1]}"
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_file.read())
        video_input = temp_video_path

if video_input:
    st.write("Extracting audio from the video...")
    try:
        audio_data, sample_rate, audio_path = extract_audio_from_video(video_input)
        st.success("✅ Audio extracted successfully!")

        multilingual_model = st.sidebar.checkbox("Use Multilingual Model for Transcription")

        st.write("🔊 **Transcribing audio... Please wait.**")
        with st.spinner("Transcribing..."):
            transcription = transcribe_audio(audio_path, multilingual_model)
        
        st.success("✅ Transcription Complete!")
        st.text_area("📝 Transcription", transcription, height=200)

        # Load BART model and tokenizer
        model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

        # Summarize and extract topic
        st.write("📄 **Summarizing transcription...**")
        summary = summarize_text(transcription, model, tokenizer)

        st.write("📌 **Extracting key topics...**")
        topic = extract_topic(transcription)

        st.subheader("📌 Topic Extracted:")
        st.write(f"`{topic}`")

        st.subheader("📄 Summary:")
        st.write(summary)

    except Exception as e:
        st.error(f"❌ Error: {e}")
