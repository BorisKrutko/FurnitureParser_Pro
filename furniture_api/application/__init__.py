from flask import Flask
import spacy
from transformers import pipeline
from dotenv import load_dotenv
load_dotenv()

# Global variables to hold the loaded models
nlp = None
bert_model = None

def create_app():
    app = Flask(__name__)
    # Load the models when the app is created
    global ner_model, bert_model
    nlp = spacy.load('trained_model')  
    bert_model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    from .routes import init_routes
    init_routes(app, nlp, bert_model)  # Pass the models to the routes

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=5000)