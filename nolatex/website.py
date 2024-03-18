import streamlit as st
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
from PIL import Image
import sympy as sp
import base64
#from ml_logic import preprocessing as pp
from API import fast
import requests
'''
## Latex Code Prediction Website
'''
# def convert_to_latex(image):

#     # conversion logic here
#     #latex_characters = fast.predict(uploadedImage=image)
#     #return latex_characters
#     return "Latex code"
def about_us():
    st.title("About The Project")
    components.iframe("https://docs.google.com/presentation/d/e/2PACX-1vRqsZjvgTuxgNIlAySRLcN0V4XzzRh8Hy_J5_VWxuD-P2c7glDs8szlXy5vZCBVnFxzPA11ApCkLueF/embed?start=false&loop=false&delayms=3000", width=960, height=569)
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Convert to LaTeX", "About The Project"])
    if page == "Convert to LaTeX":
        st.title("Image to LaTeX Converter")
        uploaded_file = st.file_uploader("Upload Picture", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            encoded_image = base64.b64encode(uploaded_file.getvalue()).decode('utf-8')
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Picture", use_column_width=True)
            if st.button("Convert to LaTeX"):
                request = requests.post("http://127.0.0.1:8000/predict", json={"image_json":encoded_image})
                result = request.json()
                latex_code = result["latex"]
                #latex_code = convert_to_latex(image)
                st.subheader("Latex Code:")
                st.code(latex_code, language="latex")
                st.markdown(f"# ${latex_code}$")
    elif page == "About The Project":
        about_us()
if __name__ == "__main__":
    main()
