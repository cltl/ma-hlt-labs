from openai import OpenAI
import json

class LlamaClient:
    
    def __init__(self, url="http://localhost:9001/v1", character="You are friendly.", api_key="not-needed"):
        self._client = OpenAI(base_url="http://localhost:9001/v1", api_key=api_key)
        self._system_name='Llama'
        self._human_name='Human'
        self._file_name = "chat_with_llama.json"
        self._history = []
        self._conversation=[]
        self._turn_id = 0
        self._instruct = []
        self.create_chat_instruct(character)
        
    def create_chat_instruct(self, character):
        self._instruct = [{"role": "system", "content": "You are an intelligent assistant and your name is {}.".format(self._system_name)}]
        self._instruct.append({"role": "system", "content": "Give short answers, no more than two sentences."})
        self._instruct.append({"role": "system", "content": character})
        self._instruct.append({"role": "system", "content": "Introduce yourself with your name {} and start the conversation by asking for the name of the user. Ask the name.".format(self._system_name)})
        print("My instructions are:", self._instruct)

    def talk_to_me(self):
        stopwords = ["quit", "exit", "bye", "stop"]
        self._history = []
        self._turn_id = 1
    
        ### We call the openai client with a low temperature for the first turn
        completion = self._client.chat.completions.create(
            model="local-model", # this field is currently unused
            messages=self._instruct,
            temperature=0.0,
            stream=True,
        )
        ### For the first turn in the chat, we instruct the Llama server and prompt it to ask for your name.
        ### We send the instruct as a prompt to get system response as the first turn.
        ### We instructed the system to ask your name.
        new_message = {"role": "system", "content": ""}
        response = ""
        for chunk in completion:
            if chunk.choices[0].delta.content:
                response += chunk.choices[0].delta.content
        print(self._system_name+":"+str(self._turn_id)+"> "+response)
        ### We extend the history with new message
        new_message["content"] += response
        self._history.append(new_message)
        ### We also save the turn in conversation
        turn = {'utterance':new_message['content'], 'speaker': self._system_name, 'turn_id':self._turn_id}
        self._conversation.append(turn)
        print()
    
        ### We assume that the server asked for your name and that you gave your name as well
        self._turn_id += 1
        userinput=input("Human"+":"+str(self._turn_id)+"> ")
        ### We add your input to the history and the conversation as well
        self._history.append({"role": "user", "content": userinput})
        turn = {'utterance':userinput, 'speaker': self._human_name, 'turn_id':self._turn_id}
        self._conversation.append(turn)
    
        ### We try to get you name from the input
        if userinput.lower().startswith("my name is "):
            self._human_name = userinput[11:]
        elif userinput.lower().startswith("i am called "):
            self._human_name = userinput[11:]
        elif userinput.lower().startswith("they call me "):
            self._human_name = userinput[13:]
        else:
            self._human_name = userinput
        
        while True:
            ### We now iteratively call the server through the client with a high temperature
            ### We add the history as the prompt 
            self._turn_id += 1
            completion = self._client.chat.completions.create(
                model="local-model", # this field is currently unused
                messages=self._history,
                temperature=0.8,
                stream=True,
            )
        
            new_message = {"role": "system", "content": ""}
            response = ""
            for chunk in completion:
                if chunk.choices[0].delta.content:
                    response += chunk.choices[0].delta.content
            print(self._system_name+":"+str(self._turn_id)+"> "+response)
            new_message["content"] = response
            self._history.append(new_message)
            turn = {'utterance':new_message['content'], 'speaker': self._system_name, 'turn_id':self._turn_id}
            self._conversation.append(turn)
            print()
    
            ### We now ask for the next user input and add it to the history
            self._turn_id += 1
            userinput = input(self._human_name+":"+str(self._turn_id)+"> ")
            self._history.append({"role": "user", "content": userinput})
            turn = {'utterance':userinput, 'speaker': self._human_name, 'turn_id':self._turn_id}
            self._conversation.append(turn)
            for word in stopwords:
                if word in userinput.lower():
                    print("BYE BYE!")
                    filename = "human_"+self._human_name+"_"+self._file_name
                    self.save_to_json(filename)
                    print("I saved the conversation in:", filename)
                    return
            
    
    def save_to_json(self, filename = "human_chat_with_llama.json"):
        with open(filename,'w') as file:
            json.dump(self._conversation, file, indent = 4)

    def load_from_json(self, filename = "chat_with_llama.json"):
        self._file_name = filename
        f = open(filename)
        self._conversation = json.load(f)

    def print_chat(self):
        for turn in self._conversation:
            print(turn)
  
    def annotate_chat(self, labels=[]):
        gold_labels = []
        annotator = ""
        while len(annotator.strip())==0:
            print("Please provide the name of the annotator")
            annotator=input("> ")
        print("The annotator is", annotator)
        print("There will be", len(self._conversation)/2, "turns to annotate with one of the following labels:", gold_labels)
        print("When you are done, the annotations will be saved in a separate JSON file prefixed with the name of the annotator")
        for turn in self._conversation:
            gold = ""
            speaker = turn['speaker']
            utterance = turn['utterance']
            print(speaker, ":", utterance)
            if speaker==self._system_name:
                turn['Gold']="neutral"
                turn['Annotator']="auto"
            else: 
                ### we keep getting the user input till one of them matches a label
                while not gold in labels:
                    gold = input("label> ")
                turn['Gold']=gold
                turn['Annotator']=annotator
        filename = "annotator_"+annotator+"_"+self._file_name
        print("Thank you for annotating. The annotations are saved in", filename)
        self.save_to_json(filename)

if __name__ == "__main__":
    url="http://localhost:9001/v1"
    character="Your answers should be agressive and grumpy."
    character="Your answers should be uncertain and emotional."
    llama = LlamaClient(url = url, character=character)
    llama.talk_to_me()

                         
