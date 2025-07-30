import streamlit as st
from gtts import gTTS
from googletrans import Translator
import os
import uuid

# Set page config
st.set_page_config(page_title="Indian Language Text-to-Speech", page_icon="🗣️")

# Title
st.title("🗣️ English to Indian Language Text-to-Speech Converter")

# Text input
english_text = st.text_area("Enter English Text:", placeholder="Type something like 'India is launching a satellite.'")

# Language options
language_names = {
    "Hindi": "hi",
    "Tamil": "ta",
    "Telugu": "te",
    "Malayalam": "ml",
    "Kannada": "kn",
    "Bengali": "bn",
    "Gujarati": "gu",
    "Marathi": "mr",
    "Punjabi": "pa",
    "Urdu": "ur"
}
language_choice = st.selectbox("Select Target Indian Language:", list(language_names.keys()))
language_code = language_names[language_choice]

# Action button
if st.button("Translate & Convert to Speech"):
    if not english_text.strip():
        st.warning("Please enter some English text.")
    else:
        # Translate text
        translator = Translator()
        try:
            translated = translator.translate(english_text, dest=language_code)
            st.success(f"Translated Text in {language_choice}:")
            st.markdown(f"**{translated.text}**")

            # Generate speech
            tts = gTTS(translated.text, lang=language_code)
            audio_file = f"output_{uuid.uuid4().hex}.mp3"
            tts.save(audio_file)

            # Audio playback
            with open(audio_file, "rb") as audio:
                st.audio(audio.read(), format="audio/mp3")

            os.remove(audio_file)

        except Exception as e:
            st.error(f"Translation or speech failed: {e}")
