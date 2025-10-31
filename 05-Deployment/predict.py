
# to activate the environment, use: source .venv/bin/activate after deactivating the conda environment
import pickle
import sklearn
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel


print("scikit-learn version:", sklearn.__version__)



with open('pipeline_v1.bin','rb') as file:
    pipeline = pickle.load(file)

print("Pipeline loaded successfully!")

class Lead(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float



app = FastAPI(title= "lead_scoring_app")

lead = {
    "lead_source": "paid_ads",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0
}

def predict_lead_single(lead: dict):

    probability = pipeline.predict_proba([lead])[0][1]  # probability of class 1 (conversion)
    return float(probability)
    
@app.post("/predict")    
def predict_lead(lead: Lead):
    lead_dict = lead.model_dump()
    probability = predict_lead_single(lead_dict)

    return {"conversion_probability": probability, "decision": probability >= 0.5}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)