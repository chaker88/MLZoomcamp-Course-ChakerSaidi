
import pickle
import sklearn

print("scikit-learn version:", sklearn.__version__)
with open('pipeline_v1.bin','rb') as file:
    pipeline = pickle.load(file)

print("Pipeline loaded successfully!")

lead = {
    "lead_source": "paid_ads",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0
}
probability = pipeline.predict_proba([lead])[0][1]  # probability of class 1 (conversion)

print(f"Probability of conversion: {probability:.4f}")