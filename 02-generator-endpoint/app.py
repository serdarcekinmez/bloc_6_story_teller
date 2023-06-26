

from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
import torch
from accelerate import Accelerator
from pydantic import BaseModel

accelerator = Accelerator()

# Create a FastAPI instance
app = FastAPI(
    title="Text Generator Endpoint",
    description="This API generates text using a pre-trained then fine tuned language model from the Huggingface Transformers library. It takes a story context as input and generates multiple text outputs.",
    version="0.1",
    contact={
        "name": "Serdar Cekinmez",
        "url": "https://github.com/serdarcekinmez",
        "email": "serdarcekinmez@gmail.com",
    }
)

# Initialize generator model and tokenizer
generator = AutoModelForCausalLM.from_pretrained('CSerdar014191/opt-350m-HorrorCreepyPasta')
gen_tokenizer = AutoTokenizer.from_pretrained('CSerdar014191/CreepyPasta-opt-350')

# Define a stopping criteria class
class KeywordsStoppingCriteria(StoppingCriteria):
    """
    A custom stopping criteria that stops generation when a certain keyword has been generated a specific number of times.
    """
    def __init__(self, keywords_ids: list, occurrences: int):
        super().__init__()
        self.keywords = keywords_ids
        self.occurrences = occurrences
        self.count = 0
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids[0][-1] in self.keywords:
            self.count += 1
            if self.count == self.occurrences:
                return True
        return False

class PredictionFeatures(BaseModel):
    story_input: str


@app.post("/generate", tags=["Generation"], summary="Generate Text",
          description="This endpoint generates multiple outputs given a story input.")
async def generate_output(PredictionFeatures: PredictionFeatures):
    """
    This function generates text based on the given story input.
    :param story_input: The initial story input to base the text generation on.
    :return: A list of generated text outputs with and without the initial input context.
    """
    # Define stop words
    stop_words = ['.']
    stop_ids = [gen_tokenizer.encode(w)[1] for w in stop_words]
    gen_outputs = []
    gen_outputs_no_input = []
    story_input = PredictionFeatures.story_input
    gen_input = gen_tokenizer(story_input, return_tensors="pt")
    for _ in range(5):
        stop_criteria = KeywordsStoppingCriteria(stop_ids, occurrences=2)
        gen_output = generator.generate(gen_input.input_ids, do_sample=True,
                                        top_k=10,
                                        top_p=0.95,
                                        max_new_tokens=100,
                                        penalty_alpha=0.6,
                                        stopping_criteria=StoppingCriteriaList([stop_criteria])
                                        )
        gen_outputs.append(gen_output)
        gen_outputs_no_input.append(gen_output[0][len(gen_input.input_ids[0]):])

    # Decode outputs
    gen_outputs_decoded = [gen_tokenizer.decode(gen_output[0], skip_special_tokens=True) for gen_output in gen_outputs]
    gen_outputs_no_input_decoded = [gen_tokenizer.decode(gen_output_no_input, skip_special_tokens=True) for gen_output_no_input in gen_outputs_no_input]

    return {"gen_outputs_decoded": gen_outputs_decoded, "gen_outputs_no_input_decoded": gen_outputs_no_input_decoded}

