import streamlit as st
from transformers import pipeline

# Load the summarization pipelines
t5_summarizer = pipeline("summarization", model="t5-small", tokenizer="t5-small")
bart_summarizer = pipeline("summarization", model="facebook/bart-large-cnn", tokenizer="facebook/bart-large-cnn")

# Display the summarization generated
def main():
    st.set_page_config(
        page_title="Bored of reading looooong texts!? Let's SUMMARIZE!",
        page_icon=":writing_hand:",
        layout="wide"
    )

    # Custom CSS for styling
    st.markdown(
        """
        <style>
        body {
            background-color: #f4f4f4;
        }
        .st-sidebar {
            background-color: #222;
            color: #fff;
        }
        .stRadio span {
            color: #333;
        }
        .stTextInput textarea {
            background-color: #fff;
            color: #333;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("Bored of reading looooong texts!? Let's SUMMARIZE!")

    model_choice = st.selectbox("Select a Model", ["T5", "BART"])

    user_input = st.text_area("Enter text to summarize:")
    if st.button("Generate Summary"):
        if model_choice == "T5":
            summarizer = t5_summarizer
        else:
            summarizer = bart_summarizer

        if user_input:
            summary = summarizer(user_input, max_length=150, min_length=30, do_sample=True)
            st.subheader("Generated Summary:")
            st.write(summary[0]['summary_text'])

if __name__ == "__main__":
    main()
