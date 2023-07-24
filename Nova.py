from llama_index import SimpleDirectoryReader, GPTListIndex, readers, GPTVectorStoreIndex, LLMPredictor, PromptHelper, ServiceContext
from langchain import OpenAI
import sys
import os
from functions import *

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

    index.save_to_disk('index.json')

    return index

def ask_ai():
    index = GPTVectorStoreIndex.load_from_disk('index.json')
    print("Nova>> Hi I am Nova, your personal assistant")
    play_text_as_sound("Hi I am Nova, your personal assistant")
    print("Nova>> Please enter/ask something: ")
    while True:
        query = input("You>> ")
        if query.lower() == 'reconstruct_module':
            construct_index('data')
        else:
            response = index.query(query)
            if response == '':
                continue
            print(f"Nova>> {response.response}")
            play_text_as_sound(response.response)
os.environ["OPENAI_API_KEY"] = "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"
construct_index("data")
ask_ai()