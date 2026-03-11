from fastapi import FastAPI, Request
import joblib

model = joblib.load('models/best_estimator_TfidfVectorizer_80000_LogisticRegression.joblib')
vectorizer = joblib.load('models/vectorizer_TfidfVectorizer_100000.pkl')
mlb = joblib.load('models/mlb_100000.pkl')

app = FastAPI()

def prediction(text):
    X = vectorizer.transform([text])
    y = model.predict(X)
    tags = mlb.inverse_transform(y)[0]
    return list(tags)

@app.get("/")
async def home():
    return "Hello, World!"

@app.post("/predict")
async def predict(request: Request):
    text = (await request.body()).decode()
    tags = prediction(text)
    return {"tags": tags}
