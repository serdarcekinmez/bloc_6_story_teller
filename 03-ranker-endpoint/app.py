from fastapi import FastAPI
from transformers import AutoModelForMultipleChoice, AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
import torch
from accelerate import Accelerator
from pydantic import BaseModel
from typing import List

accelerator = Accelerator()

# Create a FastAPI instance
app = FastAPI(
    title="Text Ranking API",
    description="This API ranks a list of generated text outputs according to their continuation probability from a given story input. It uses a pre-trained language model from the Transformers library.",
    version="0.1",
    contact={
        "name": "Serdar Cekinmez",
        "url": "https://github.com/serdarcekinmez",
        "email": "serdarcekinmez@gmail.com",
    }
)

# Initialize ranker model and tokenizer
rank_checkpoint="hypefi/my_awesome_swag_model"
rank_tokenizer = AutoTokenizer.from_pretrained(rank_checkpoint)
ranker = AutoModelForMultipleChoice.from_pretrained(rank_checkpoint)

class PredictionFeatures(BaseModel):
    story_input: str
    continuations: List[str]

@app.post("/rank_outputs", tags=["Ranking"], summary="Rank Text Outputs", description="This endpoint ranks a list of generated text outputs based on their continuation probability from a given story input.")
async def rank_outputs(PredictionFeatures: PredictionFeatures):
    """
    This function ranks a list of generated text outputs based on their continuation probability from a given story input.
    :param context: The story input to base the ranking on.
    :param continuation: The list of generated text outputs to be ranked.
    :return: The index of the output with the highest score.
    """
    # Prepare the ranking inputs
    context = PredictionFeatures.story_input
    continuations = PredictionFeatures.continuations
    rank_inputs = rank_tokenizer([[context, continuation] for continuation in continuations],
                                    return_tensors="pt",
                                    padding=True)
    rank_labels = torch.tensor(0).unsqueeze(0)

    # Pass the inputs to the ranker model
    rank_outputs = ranker(**{k: v.unsqueeze(0) for k, v in rank_inputs.items()}, labels=rank_labels)

    # Compute the softmax of the logits to get the probabilities
    rank_predictions = torch.nn.functional.softmax(rank_outputs.logits, dim=-1)[0].tolist()

    # Return the index of the output with the highest score
    return {"rank_predictions": rank_predictions}
