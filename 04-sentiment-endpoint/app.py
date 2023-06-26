
from fastapi import FastAPI
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import torch
from accelerate import Accelerator
from pydantic import BaseModel
from typing import List

accelerator = Accelerator()

# Create a FastAPI instance
app = FastAPI(
    title="Sentiment Analysis API",
    description="This API outputs the sentiment score of the continuations using a pre-trained sentiment analysis model from the Transformers library.",
    version="0.1",
    contact={
        "name": "Serdar Cekinmez",
        "url": "https://github.com/serdarcekinmez",
        "email": "serdarcekinmez@gmail.com",
    }
)

# Initialize sentiment analysis model and tokenizer
sentiment_checkpoint="cardiffnlp/twitter-roberta-base-sentiment-latest"
sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_checkpoint)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_checkpoint)

class PredictionFeatures(BaseModel):
    continuations: List[str]

# Sentiment analysis
@app.post("/analyze_sentiment", tags=["Guiding"], summary="Guiding Generative Model",
description="This endpoint analyses the generated continuations based on their sentiments (Positive, Negative, Neutral).")
async def analyze_sentiment(PredictionFeatures: PredictionFeatures):
    # Guiding the model with his ranking
    continuations = PredictionFeatures.continuations
    print(continuations)
    sentiment_inputs = sentiment_tokenizer([output for output in continuations],
                                      return_tensors='pt', padding=True, truncation=True)
    sentiment_outputs = sentiment_model(**sentiment_inputs)
    # the slicing at the end [:,x]: x=0 for negative, x=1 for neutral, x=2 for positive
    sentiment_scores = torch.nn.functional.softmax(sentiment_outputs.logits, dim=-1)[:, 0].tolist()

    return {"sentiment_scores": sentiment_scores}
