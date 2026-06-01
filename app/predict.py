import joblib

model = joblib.load("model.pkl")

def predict(features):
    return model.predict(features