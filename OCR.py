import streamlit as st
from fpdf import FPDF
import base64
import requests
import cv2
import numpy as np
import pytesseract
from PIL import Image
from streamlit_extras.stateful_button import button
import unicodedata
import google.generativeai as genai
# Initialize Google Gemini Pro
genai.configure(api_key="AIzaSyATeuavfgyi58IOrWbYnikFchU4BCoZuhw")
# Create the model with the configuration
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

modell = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)
def inputmed(content):

    prompt = f"""
        This is the content {content}, Can you summarize it.
        """
        
    # Generate the response from the model
    response = modell.generate_content(prompt)
    return response.text
pytesseract.pytesseract.tesseract_cmd = r"C:\program files\Tesseract-OCR\tesseract.exe"

def download_image(url, filename):
    r = requests.get(url)
    with open(filename, 'wb') as out_file:
        out_file.write(r.content)

def process_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    cv2.imwrite("removed_noise.png", img)
    cv2.imwrite(img_path, img)
    result = pytesseract.image_to_string(Image.open(img_path))
    return result

def replace_special_characters(text):
    # Remove accents and special characters by normalizing the text
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    return text

st.title("Image Downloader and OCR App")
st.write("This app downloads an image from a URL, processes it, and performs OCR on it.")

# Input URL
url = st.text_input("Enter image URL:")
extracted_text = ""

if button("Download and Process Image", key="button1"):
    if url:
        filename = 'downloaded_image.jpg'
        download_image(url, filename)
        st.write("Image downloaded successfully!")

        st.image(filename, caption='Downloaded Image', use_column_width=True)

        st.write("Processing Image...")
        extracted_text = process_image(filename)
        st.write("Text Extracted from Image:")
        st.write(extracted_text)

        report_text = st.text_input("Report Text", value=extracted_text)
        
        if button("Save Edited Text", key="button2"):
            extracted_text = report_text
        
        extracted_textt = str(extracted_text)
        st.text(extracted_textt)

        # Replace special characters in the text
        cleaned_text = replace_special_characters(extracted_textt)
        summ = inputmed(content=cleaned_text)
        st.info(f"{summ}")

        def create_download_link(val, filename):
            b64 = base64.b64encode(val)  # Encode the value
            return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}.pdf">Download file</a>'
        
        if cleaned_text:
            if button("Export Report", key="button3"):
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font('Arial', 'B', 16)
                
                # Now you can safely use the cleaned text with FPDF
                pdf.multi_cell(0, 10, cleaned_text)

                # Generate the download link
                html = create_download_link(pdf.output(dest="S").encode("latin-1"), "test")

                st.markdown(html, unsafe_allow_html=True)
