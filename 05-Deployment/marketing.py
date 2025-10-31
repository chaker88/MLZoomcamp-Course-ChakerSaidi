import requests

url ="http://127.0.0.1:9696/predict"

lead = {
    "lead_source": "organic_search",
    "number_of_courses_viewed": 4,
    "annual_income": 80304.0
}

response = requests.post(url, json=lead)


prediction = response.json()
print("Conversion Probability:", prediction["conversion_probability"])

if prediction["decision"]:
    print("The lead is likely to convert.")
else:
    print("The lead is unlikely to convert.")