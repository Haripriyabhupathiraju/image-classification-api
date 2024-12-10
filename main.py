from fastapi import FastAPI, File, UploadFile
from model import ImageClassifier

app = FastAPI()
classifier = ImageClassifier()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    prediction = classifier.predict(contents)
    return {"prediction": prediction}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)