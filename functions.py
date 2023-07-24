import json
import pyttsx3
import speech_recognition as sr
import openai

openai.api_key = "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"

 
from functions import *
recognizer = sr.Recognizer()

def write_json(input: str, response: str):
    with open('context.json', 'r+') as file:
        data = json.load(file)
    data.append({
        'role' : "assistant",
        'content': response
    })

    with open('context.json', 'w') as file:
        json.dump(data, file)


def add_user_prompt(input: str):
    with open('context.json', 'r+') as file:
        data = json.load(file)

    data.append({
        'role' : 'user',
        'content' : input
    })
    with open('context.json', 'w') as file:
        json.dump(data, file)


def play_text_as_sound():
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[3].id)
    engine.say("hello I am the voice from your PC")
    engine.runAndWait()
    

def recognize():
    with sr.Microphone as mic:

        recognizer.adjust_for_ambient_noise(mic, duration=0.2)
        audio = recognizer.listen(mic)

        text = recognizer.recognize_google(audio)
        return text.lower()

def play_text_as_sound(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def ask_chatgpt(messages: list):
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=messages,
        temperature=0.4,
        max_tokens=2048
    )
    return response

def read_json():
    with open('context.json', "r") as file:
        data = json.load(file)
        conversation = []
        for i in data:
            conversation.append(i)
    return conversation