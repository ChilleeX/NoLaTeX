import pandas as pd
import base64
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
#from tf.keras.utils import load_img
from nolatex.ml_logic.registry import make_prediction

#TODO potentially update imports
#from NoLaTeX.nolatex.ml_logic.preprocessing import image_preprocessing
#from NoLaTeX.ml_logic.registry import load_model


app = FastAPI()
#app.state.model = load_model()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

#TODO Update function according to our project
# @app.get("/predict")
# def predict(
#         uploadedImage:   # UploadedFile Class of Streamlit
#     ) -> str :
#     """
#     Make a single conversion from picture of hand-written math formular to
#     LaTeX code
#     """

#     # Loading the uploaded image
#     #img = load_img(uploadedImage)
#     result = make_prediction(uploadedImage)

#     # Compute `fare_prediction`
#     #prediction = app.state.model.predict(img_preprocessed)

#     return result


@app.get("/")
def root():
    return {
        'greeting': 'Hello'
    }

@app.post("/predict")
async def predict(image_json: Request):

    image = await image_json.json()

    var = base64.b64decode(image["image_json"])
    path = "/home/diegoberan/code/ChilleeX/NoLaTeX/initial_test_data/saved_images/temp_image.jpg"
    with open(path, "wb+") as f:
        f.write(var)
    result = make_prediction(path)

    return {"latex":result}
