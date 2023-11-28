# Importations
from fastapi import FastAPI
import joblib
from pydantic import BaseModel
import os, uvicorn
import pandas as pd
from fastapi.encoders import jsonable_encoder

# Additional information to include in app description
additional_info = """
- Plasma_Glucose: Plasma Glucose
- Elevated_Glucose: Blood Work Result-1 (mu U/ml)
- Diastolic_Blood_Pressure: Blood Pressure (mm Hg)
- Triceps_Skinfold_Thickness: Blood Work Result-2 (mm)
- Insulin_Levels: Blood Work Result-3 (mu U/ml)
- Body_Mass_Index_BMI: Body Mass Index (weight in kg/(height in m)^2)
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
    Plasma_Glucose: float
    Elevated_Glucose: float
    Diastolic_Blood_Pressure: float
    Triceps_Skinfold_Thickness: float
    Insulin_Levels: float
    Body_Mass_Index_BMI: float
    Diabetes_Pedigree_Function: float
    Age: int
   
    


#ENDPOINT
@app.get("/")
async def root():
    return{"message": "Online"}


@app.post("/predict")
def predict(input: Input):
    pipeline = joblib.load("./ML_models/sepsis_classification_pipeline.joblib")
    model = joblib.load("./ML_models/random_forest_model.joblib")
    
    features = [input.Plasma_Glucose, 
input.Elevated_Glucose, 
input.Diastolic_Blood_Pressure,
input.Triceps_Skinfold_Thickness,
input.Insulin_Levels,
input.Body_Mass_Index_BMI,
input.Diabetes_Pedigree_Function,
input.Age
]


    transformer = pipeline.predict([features])[0]
    prediction = model.predict([features])[0]
    
    # Serialize the prediction result using jsonable_encoder
    serialized_prediction = jsonable_encoder({"prediction": int(prediction)})
    if serialized_prediction["prediction"] == 1:
        Diagnosis = {"Results" : "Positive"}
    else:
        Diagnosis = {"Results" : "Negative"}
    return Diagnosis


if __name__ == '__main__':
    uvicorn.run('api:app', reload =True)