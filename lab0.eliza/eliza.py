import re
import random
import eliza_language as lang
import pandas as pd

human_name='Stranger'
system_name='Eliza'

conversation=[]

def reflect(fragment):
    tokens = fragment.lower().split()
    for i, token in enumerate(tokens):
        if token in lang.REFLECTIONS:
            tokens[i] = lang.REFLECTIONS[token]
    return ' '.join(tokens)


def analyze(statement):
    for pattern, responses in lang.PSYCHOBABBLE:
        match = re.match(pattern, statement.rstrip(".!"))
        if match:
            response = random.choice(responses)
            return response.format(*[reflect(g) for g in match.groups()])


def talk_to_me():
    print("My name is {}. What is your name?".format(system_name))
    human_name=input("> ")
    turn_id = 0
    prompt = "Hello {}. How are you feeling today?".format(human_name)
    turn = {'utterance':prompt, 'speaker': system_name, 'turn_id':turn_id}
    conversation.append(turn)
    print(prompt)

    while True:
        statement = input("> ")

        if statement.lower()=='stop' or statement.lower()== "quit" or statement.lower()== "bye":
            break
        turn_id += 1   
        turn = {'utterance':statement, 'speaker': human_name, 'turn_id':turn_id}
        conversation.append(turn)

        prompt = analyze(statement)
        turn = {'utterance':prompt, 'speaker': system_name, 'turn_id':turn_id}
        conversation.append(turn)            
        print(prompt)


if __name__ == "__main__":
    talk_to_me()
