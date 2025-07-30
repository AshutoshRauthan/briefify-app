import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import tempfile

@st.cache_resource(show_spinner=False)
def load_tinyllama():
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

@st.cache_resource(show_spinner=False)
def load_whisper():
    return pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-tiny.en",
        chunk_length_s=30,
    )

def format_prompt_with_context(context):
    return (
        "You are a helpful summarizer.\n"
        "Your task is to extract key points ONLY from the given context.\n"
        "Follow these strict rules:\n"
        "- Use ONLY the information from the text. Do NOT add outside knowledge.\n"
        "- Do NOT make assumptions or guesses.\n"
        "- Avoid contradictions and repetition.\n"
        "- Keep key points accurate, concise, and directly supported by the text.\n"
        "- Format the output as bullet points.\n\n"
        "- Do NOT make assumptions\n\n"
        f"Context:\n{context}\n\n"
        "Key Points:"
    )
    )

def generate_key_points(pipe, context, max_tokens=400):
    prompt = format_prompt_with_context(context)
    output = pipe(prompt, max_new_tokens=max_tokens, do_sample=True, temperature=0.7)
    generated_text = output[0]["generated_text"]
    key_points = generated_text[len(prompt):].strip()
    return key_points

def transcribe_audio(whisper_pipe, audio_file):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        tmp.write(audio_file.read())
        tmp.flush()
        transcript_txt = whisper_pipe(tmp.name, batch_size=8)["text"]
    return transcript_txt

st.set_page_config(page_title="Briefify 🎧: Turn Audio into Key Points", layout="centered")

st.markdown(
    """
    <h1 style="
        text-align: center; 
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-weight: 700; 
        font-size: 3rem; 
        color: #ffffff;
        margin-bottom: 0.2rem;
    ">
        Briefify <span style="font-size:1.8rem;">🎧</span>
    </h1>
    <h3 style="
        text-align: center;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #ffffff;
        font-weight: 400;
        font-size: 1.45rem;
        margin-top: 0;
        margin-bottom: 1.2rem;
    ">
        Turn Audio into Key Points
    </h3>
    """,
    unsafe_allow_html=True
)

st.sidebar.title("Instructions")
st.sidebar.info(
    "1. Upload an audio file (mp3, wav)\n"
    "2. Wait for auto-processing\n"
    "3. The transcript appears below and you can expand/collapse its view!\n"
    "4. Key points are summarized after transcript generation."
)

audio_file = st.file_uploader("Upload your audio file", type=["wav", "mp3"])

if audio_file:
    st.audio(audio_file)

    whisper_pipe = load_whisper()
    tinyllama_pipe = load_tinyllama()

    with st.spinner("Transcribing audio with Whisper..."):
        transcript = transcribe_audio(whisper_pipe, audio_file)

    with st.expander("See Transcript"):
        st.write(transcript)

    with st.spinner("Generating key points with TinyLLaMA..."):
        key_points_str = generate_key_points(tinyllama_pipe, transcript)

    st.markdown("### Key Points")
    for point in key_points_str.split('\n'):
        if point.strip():
            st.markdown(f"- {point.strip()}")

    st.success("Processing complete!")

else:
    st.info("Please upload an audio file to get started.")
