# Importations
from fastapi import FastAPI
import joblib
from pydantic import BaseModel
import os, uvicorn
import pandas as pd
from fastapi.encoders import jsonable_encoder

# Additional information to include in app description
additional_info = """
- Plasma_glucose: Plasma Glucose
- BWR1: Blood Work Result-1 (mu U/ml)
- Blood_pressure: Blood Pressure (mm Hg)
- BWR2: Blood Work Result-2 (mm)
- BWR3: Blood Work Result-3 (mu U/ml)
- Bodymass_index: Body Mass Index (weight in kg/(height in m)^2)
- BWR4: Blood Work Result-4 (mu U/ml)
- Age: Patient's Age (years)
- Insurance: If a patient holds a valid insurance card

Output:
- Sepsis: Positive if a patient in ICU will develop sepsis, Negative if a patient in ICU will not develop sepsis.
"""

# APP
app = FastAPI(
    title='Sepsis Prediction App',
    description='This app was built to predict if patients are Sepsis Positive or Sepsis Negative. '
                'It uses a machine learning model to make predictions based on patient data. '
                'Below is the kind of patient data required for the prediction, and the meaning of the prediction output.'
                + additional_info
)


# #API INPUT
class Input(BaseModel):
    Plasma_glucose: int
    BWR1: int
    Blood_pressure: int
    BWR2: int
    BWR3: int
    Bodymass_index: float
    BWR4: float
    Age: int
    Insurance: int
    


#ENDPOINT
@app.get("/")
async def root():
    return{"message": "Online"}


@app.post("/predict")
def predict(input: Input):
    scaler = joblib.load("Assets/scaler.joblib")
    model = joblib.load("Assets/model.joblib")
    
    features = [input.Plasma_glucose, 
input.BWR1, 
input.Blood_pressure,
input.BWR2,
input.BWR3,
input.Bodymass_index,
input.BWR4,
input.Age,
input.Insurance]


    scaled_features = scaler.transform([features])[0]
    prediction = model.predict([scaled_features])[0]
    
    # Serialize the prediction result using jsonable_encoder
    serialized_prediction = jsonable_encoder({"prediction": int(prediction)})
    if serialized_prediction["prediction"] == 1:
        Diagnosis = {"Results" : "Positive"}
    else:
        Diagnosis = {"Results" : "Negative"}
    return Diagnosis


if __name__ == '__main__':
    uvicorn.run('api:app', reload =True)