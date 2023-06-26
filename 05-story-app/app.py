import streamlit as st
from PIL import Image
import json
import requests
import numpy as np

# Define the base URL of the endpoints
generator_endpoint = "http://0.0.0.0:8001"
ranker_endpoint = "http://0.0.0.0:8002"
sentiment_endpoint = "http://0.0.0.0:8003"
#generator_endpoint = "http://story-gen-api:80"
#ranker_endpoint = "http://ranking-api:80"
#sentiment_endpoint = "http://sentiment-api:80"

def generate(story_input):
    response = requests.post(f"{generator_endpoint}/generate",
                              json={"story_input": story_input})
    return response.json()

def rank(story_input, continuations):
    response = requests.post(f"{ranker_endpoint}/rank_outputs",
                              json={"story_input": story_input, "continuations": continuations})
    return response.json()

def guider(continuations):
    response = requests.post(f"{sentiment_endpoint}/analyze_sentiment",
                              json={"continuations": continuations})
    return response.json()


def generate_best_continuation(story_input, sentiment_weight=0.2):
    generator_output = generate(story_input)
    gen_outputs_decoded = generator_output["gen_outputs_decoded"]
    gen_outputs_no_input_decoded = generator_output["gen_outputs_no_input_decoded"]
    ranker_output = rank(story_input, gen_outputs_no_input_decoded)
    ranks = ranker_output["rank_predictions"]
    guider_output = guider(gen_outputs_no_input_decoded)
    sentiment_scores = guider_output["sentiment_scores"]
    # Calculate the prior probability from the ranker score
    prior = ranks
    # Calculate the likelihood from the sentiment score
    likelihood = [sentiment_score**sentiment_weight for sentiment_score in sentiment_scores]
    # Calculate the posterior probability using Bayes' theorem
    posterior = [p * l for p, l in zip(prior, likelihood)]
    # Then in your story generation loop:
    continuation = gen_outputs_no_input_decoded[posterior.index(max(posterior))]
    return continuation


### Config
st.set_page_config(
    page_title="The Interactive Storyteller",
    page_icon=":skull:",
    layout="wide"
)



### Header 
header_left, title, header_right = st.columns([1,5,1])

with header_left:
    st.write("")

with title:
    st.title("The Interactive Storyteller")

with header_right:
    st.write("")


### Body
body_left, body, body_right = st.columns([1,5,1])

with body_left:
    st.write("")

with body:

    # Set session states for the widgets
    if "disable_input" not in st.session_state:
        st.session_state.disable_input = False
    
    if "disable_button" not in st.session_state:
        st.session_state.disable_button = False

    if "log" not in st.session_state:
        st.session_state.log = []

    if "generate_best_continuation" not in st.session_state:
        st.session_state.generate_best_continuation = False

    if "start_story" not in st.session_state:
        st.session_state.start_story = False
    
    if "reload" not in st.session_state:
        st.session_state.reload = True
    
    # Set functions for the widgets
    def btn1_clicked() :
        st.session_state.reload = False
        st.session_state.start_story = True
        st.session_state.disable_input = True
        st.session_state.disable_button = True

    def reload_clicked() :
        #st.cache_data.clear()
        st.session_state.reload = True
        st.session_state.disable_input = False
        st.session_state.disable_button = False
        st.session_state.start_story = False
        st.session_state.generate_best_continuation = False
        st.session_state.checkbox = False
        st.session_state.log.clear()

    # Front-end start
    image = Image.open("./images/skulls2.jpg")
    st.image(image)

    # Inputs
    st.header("Enter your prompt")
    user_input = st.text_input(label="You can write the beginning of the story, a title etc...",
                               key="input1",
                               max_chars=1500,
                               disabled=st.session_state.disable_input)

    st.subheader("OR")

    st.header("Choose a given prompt")
    option = st.selectbox("Pick a sentence from the list below ",
                ["Suddenly, she started screaming like she saw a ghost", 
                "I was walking home alone when I saw an old man staring at me",
                "It was a Friday night, the sound of the storm outside was strong",
                "I was hiding myself under a bed, just next to a dead body"            
                ],
                key="input2",
                disabled=st.session_state.disable_input
            )

    # Button style
    m = st.markdown("""
        <style>
            div.stButton > button:first-child {
                background-color: rgb(113, 44, 44);
                font-size:15px;
                height:4em;
                width:10em

            }
        </style>""", unsafe_allow_html=True)

    st.button("START STORY", on_click=btn1_clicked, disabled=st.session_state.disable_button)

    text_box = st.empty()

    # Using cache to keep in memory the result of the first generation and not reload it after clicking on the second button
    
    def update_text_box():
        st.session_state.text_box = " ".join(st.session_state.log)


    # Predicted story
    if st.session_state.start_story :
        with st.spinner(text="Generation in progress...") :
            if st.session_state.input1 != "" :
                st.session_state.log.append(st.session_state.input1)
                story_input = " ".join(st.session_state.log)
                prediction = generate_best_continuation(story_input)
                st.session_state.log.append(prediction)
                
            else :
                st.session_state.log.append(st.session_state.input2)
                story_input = " ".join(st.session_state.log)
                prediction = generate_best_continuation(story_input)
                st.session_state.log.append(prediction) 
            
        update_text_box()   

        st.session_state.start_story = False
        st.session_state.generate_best_continuation = True

    if st.session_state.generate_best_continuation:
        st.markdown("If you want to continue the story from the previous output, you can write the rest of the story and press enter")
        new_user_input = st.text_input(label="You can continue the story yourself, or let the model do it...",
                                    max_chars=1500,
                                    key="input3")

        if new_user_input:
            with st.spinner(text="Generation in progress...") :
                st.session_state.log.append(st.session_state.input3)
                story_input = " ".join(st.session_state.log)
                prediction = generate_best_continuation(story_input)
                st.session_state.log.append(prediction)
            
            update_text_box()
    
    text_box.text_area(label="Generated story",
                        key="text_box",
                        height=200, 
                        label_visibility="hidden",
                        disabled=True)
    
    st.checkbox("Reload generation", 
                key="checkbox",
                on_change=reload_clicked, 
                disabled=st.session_state.reload)

with body_right:
    st.write("")

### Footer 
footer_left, footer, footer_right = st.columns([1,5,1])

with footer_left:
    st.write("")

with footer:
    st.write("")

with footer_right:
    st.write("")