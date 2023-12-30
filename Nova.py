import openai
import os
import pyttsx3
import speech_recognition as sr
recognizer = sr.Recognizer()

def recognize():
    with sr.Microphone() as mic:

        recognizer.adjust_for_ambient_noise(mic, duration=0.2)
        audio = recognizer.listen(mic)

        text = recognizer.recognize_google(audio)
        return text.lower()

def play_text_as_sound(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

API_KEY = ""

os.environ["OPENAI_API_KEY"] = API_KEY
openai.api_key = API_KEY
from llama_index import SimpleDirectoryReader, GPTListIndex, readers, GPTVectorStoreIndex, LLMPredictor, PromptHelper, ServiceContext, StorageContext, load_index_from_storage
from langchain import OpenAI


def construct_index(directory_path):
    max_input_size = 4096
    num_outputs = 2000
    max_chunk_overlap = 0.2
    chunk_size_limit = 600
    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.5, model_name="text-davinci-003", max_tokens=num_outputs))

    documents = SimpleDirectoryReader(directory_path).load_data()

    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)

    index.storage_context.persist('./storage')

    return index

def ask_ai():

    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    index = load_index_from_storage(storage_context=storage_context)
    print("Nova>> Hi I am Nova, your personal assistant")
    play_text_as_sound("Hi I am Nova, your personal assistant")
    while True:
        print('Nova>> Please enter/ask something')
        query = input("YOU>> ")

        if query.lower() == 'reconstruct_module':
            construct_index('data')
        else:
            query_engine = index.as_query_engine()
            response = query_engine.query(query)
            if response == '':
                continue
            print(f"Nova>> {response.response}")
            play_text_as_sound(response.response)
construct_index("data")
ask_ai()