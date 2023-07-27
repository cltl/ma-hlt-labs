import re
import random
import eliza_language as lang
import json

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
    global human_name
    human_name=input("> ")
    turn_id = 0
    eliza_start = "Hello {}. How are you feeling today?".format(human_name)
    turn = {'utterance':eliza_start, 'speaker': system_name, 'turn_id':turn_id}
    conversation.append(turn)
    print(eliza_start)

    while True:
        human_input = input("> ")

        turn_id += 1   
        turn = {'utterance':human_input, 'speaker': human_name, 'turn_id':turn_id}
        conversation.append(turn)
        if human_input.lower()=='stop' or human_input.lower()== "quit" or human_input.lower()== "bye":
            break
        else:
            eliza_response = analyze(human_input)
            turn = {'utterance':eliza_response, 'speaker': system_name, 'turn_id':turn_id}
            conversation.append(turn)            
            print(eliza_response)
    return

def save_to_json(filename = "chat_with_eliza.json"):
    with open(human_name+"_"+filename,'w') as file:
        json.dump(conversation, file, indent = 4)

def print_chat():
    for turn in conversation:
        print(turn)

def annotate_chat(labels=[]):
    gold_labels = []
    for turn in conversation:
        gold = ""
        speaker = turn['speaker']
        utterance = turn['utterance']
        print(speaker, ":", utterance)
        if speaker=='Eliza':
            gold='neutral'
        else: 
            ### we keep getting the user input till one of them matches a label
            while not gold in labels:
                gold = input("label> ")
        turn['Gold']=gold

if __name__ == "__main__":
    talk_to_me()
