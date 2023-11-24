from transformers import GPT2Tokenizer, GPT2LMHeadModel
import streamlit as st
import PyPDF2
from io import BytesIO
import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, LlamaTokenizerFast
from peft import PeftModel 

# Load pre-trained model and tokenizer
MODEL_NAME = "EleutherAI/gpt-neo-2.7B"  # Example GPT model; replace with FinGPT if available
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)


def read_pdf(file):
    """Reads the content of a PDF and returns the text."""
    pdf = PyPDF2.PdfFileReader(file)
    text = ""
    for page_num in range(pdf.numPages):
        text += pdf.getPage(page_num).extractText()
    return text

def process_text_with_model(text):
    """Process text with GPT model from Hugging Face."""
    input_ids = tokenizer.encode(text, return_tensors="pt")
    
    # Generate response from the model
    with torch.no_grad():
        output = model.generate(input_ids, max_length=150, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    
    # Decode the output to text
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return response

# Streamlit interface
st.title("PDF Upload and Process")

uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file:
    st.write("Reading PDF...")
    text = read_pdf(uploaded_file)
    st.write("Processing with the model...")
    result = process_text_with_model(text)
    
    st.write("Result:")
    st.write(result)

if __name__ == "__main__":
    st.run()