import streamlit as st
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoProcessor, AutoModelForCausalLM
from PIL import Image
from groq import Groq

# Initializing the Groq API for refining captions
groq_api_key = st.secrets["GROQ_API_KEY"]

if groq_api_key:
    client = Groq(api_key=groq_api_key)
else:
    st.error("Groq API key not found. Please set the GROQ_API_KEY in the Streamlit Cloud Secrets.")

# Streamlit app for deploying the model
st.title("🛍️ Fashion Image Captioning")
st.write("**Upload an image or take a picture using the webcam to generate a caption.**")

st.markdown("""
<style>
    .css-18e3th9 {
        padding-top: 4rem;
        padding-bottom: 4rem;
    }
</style>
""", unsafe_allow_html=True)

# Option to select model type
model_type = st.selectbox("Choose Model Type:", ["BLIP Fine-Tuned", "GIT Fine-Tuned", "BLIP", "GIT"])

# Load models and processors based on user selection
if model_type == "BLIP Fine-Tuned":
    model = BlipForConditionalGeneration.from_pretrained("sagniksengupta/blip-finetuned-facad-v2")
    processor = BlipProcessor.from_pretrained("sagniksengupta/blip-finetuned-facad")
elif model_type == "GIT Fine-Tuned":
    model = AutoModelForCausalLM.from_pretrained("sagniksengupta/git-finetuned-facad-v2")
    processor = AutoProcessor.from_pretrained("sagniksengupta/git-finetuned-facad-v2")
elif model_type == "BLIP":
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
elif model_type == "GIT":
    model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-coco")
    processor = AutoProcessor.from_pretrained("microsoft/git-base-coco")
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Option to upload image or use the webcam
option = st.radio(
    "Choose input method:",
    ('Upload Image', 'Take a Picture via Webcam')
)

# Webcam functionality using built-in camera_input of Streamlit
image = None
if option == 'Take a Picture via Webcam':
    enable_camera = st.checkbox("Enable camera")
    picture = st.camera_input("Take a picture", disabled=not enable_camera)

    if picture:
        image = Image.open(picture)
        st.image(image, caption="Captured Image", use_column_width=True)

# Upload functionality
elif option == 'Upload Image':
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

# Generate original model caption and refine caption using Llama 3.1
if image is not None:
    with st.spinner("Generating caption..."):
        inputs = processor(image, return_tensors="pt").to(device)
        with torch.no_grad():
            generated_ids = model.generate(**inputs)
            caption = processor.decode(generated_ids[0], skip_special_tokens=True)

    st.write("Generated Original Model Caption: ", caption)

    # LLaMA 3.1 Refining
    if st.button("Refine Caption using Llama 3.1"):
        with st.spinner("Refining the caption with LLaMA..."):
            prompt = f"""Fix the grammar and make the following caption coherent:

            Caption: "{caption}"

            Return the only the updated caption."""

            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "user", "content": prompt}
                ],
                model="llama-3.1-70b-versatile"
            )

            refined_caption = chat_completion.choices[0].message.content.strip()

        st.success("Refined Caption: " + refined_caption)
