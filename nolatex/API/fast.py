import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from tf.keras.utils import load_img

#TODO potentially update imports
from NoLaTeX.nolatex.ml_logic.preprocessing import image_preprocessing
from NoLaTeX.ml_logic.registry import load_model


app = FastAPI()
app.state.model = load_model()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

#TODO Update function according to our project
@app.get("/predict")
def predict(
        uploadedImage: UploadedFile,  # UploadedFile Class of Streamlit
    ) -> str :
    """
    Make a single conversion from picture of hand-written math formular to
    LaTeX code
    """

    # Loading the uploaded image
    img = load_img(uploadedImage)

    # Preprocess Image
    # TODO update once function is finalized
    img_preprocessed = image_preprocessing(img, height, width)

    # Compute `fare_prediction`
    prediction = app.state.model.predict(img_preprocessed)

    return prediction


@app.get("/")
def root():
    return {
        'greeting': 'Hello'
    }
