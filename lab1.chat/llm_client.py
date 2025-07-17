from langchain_ollama import ChatOllama
import json

#https://ollama.com/models
#ollama pull qwen3:1.7b

class LLMClient:
    
    def __init__(self,  model="qwen3:1.7b",
                        temperature=0.1, 
                        character="You are friendly.", 
                        ctx_limit=2048):
        
        self._client = ChatOllama(
            model=model,
            temperature= temperature,
            num_predict = 256, ## max number of tokens to predict
        )
        self._ctx = ctx_limit
        self._system_name="AI"
        self._human_name='Human'
        self._file_name = "chat_with_"+model+".json"
        self._history = []
        self._conversation=[]
        self._turn_id = 0
        self._instruct = []
        self.create_chat_instruct(character)
        
    def create_chat_instruct(self, character):
        self._instruct = [{"role": "system", "content": "You act as a person and your name is {}.".format(self._system_name)}]
        self._instruct.append({"role": "system", "content": "Give short answers, no more than two sentences."})
        self._instruct.append({"role": "system", "content": character})
        self._instruct.append({"role": "system", "content": "Introduce yourself with your name {} and start the conversation by asking for the name of the user. Ask the name.".format(self._system_name)})
        print("My instructions are:", self._instruct)

    def add_to_history_check_context_limit(self, input):
        context = ""
        for message in self._history:
            context += str(message)
        if len(context+str(input))>self._ctx:
            if len(self._history)>4:
                ##### shorten the history with 4 turns
                self._history = self._history[4:]
            else:
                self._history ==[]
        self._history.append(input)

    def talk_to_me(self):
        stopwords = ["quit", "exit", "bye", "stop"]
        self._history = []
        self._turn_id = 1

        ### For the first turn in the chat, we instruct the LLM server and prompt it to ask for your name.
        new_message = {"role": "system", "content": ""}
        response = ""
        response = self._client.invoke(self._instruct)
        #print(response) ##AIMessage
        #content="<think>\nOkay, the user wants me to act as Qwen, a grumpy and aggressive AI. I need to introduce myself with my name and ask for the user's name. Let me make sure the response is short, under two sentences, and has that grumpy tone.\n\nFirst, I'll start with my name, Qwen. Then, I'll ask for the user's name. Keep it snappy and sarcastic. Maybe add some emojis to make it more aggressive. Let me check the rules again to ensure it's within the limits. Yep, two sentences max. Alright, time to put it all together.\n</think>\n\nI'm Qwen, the grumpy AI here to mess with your day! What's your name? 😎" additional_kwargs={} response_metadata={'model': 'qwen3:1.7b', 'created_at': '2025-07-17T08:41:52.252326Z', 'done': True, 'done_reason': 'stop', 'total_duration': 7805707550, 'load_duration': 53272502, 'prompt_eval_count': 65, 'prompt_eval_duration': 45067672, 'eval_count': 153, 'eval_duration': 7705377038, 'message': Message(role='assistant', content='', thinking=None, images=None, tool_calls=None)} id='run--27fad9a7-2ec2-410d-8c24-98bff273f840-0' usage_metadata={'input_tokens': 65, 'output_tokens': 153, 'total_tokens': 218}
        end_of_think = response.content.find("</think>")
        think = response.content[:end_of_think+8]
        answer = response.content[end_of_think+8:].replace("\n", "")
        print(self._system_name+":"+str(self._turn_id)+"> "+answer)
        ### We extend the history with new message
        new_message["content"] += answer
        self.add_to_history_check_context_limit(new_message)
        ### We also save the turn in conversation
        turn = {'utterance':new_message['content'], 'think':think, 'speaker': self._system_name, 'turn_id':self._turn_id}
        self._conversation.append(turn)
    
        ### We assume that the server asked for your name and that you gave your name as well
        self._turn_id += 1
        userinput=input(self._human_name+":"+str(self._turn_id)+"> ")
        ### We add your input to the history and the conversation as well
        self.add_to_history_check_context_limit({"role": "user", "content": userinput})
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
            print(self._system_name+":"+str(self._turn_id)+"> ")
            new_message = {"role": "system", "content": ""}
            answer = ""
            think = ""
            end_of_thinking = False
            for chunk in self._client.stream(self._history):
#                print("DEBUG chunk:", chunk.text())
                if not end_of_thinking:
                    think += chunk.text()
                    if chunk.text()=="</think>":
                        end_of_thinking = True
                else:
                    answer += chunk.text()
                    print(chunk.text(), end= "")
            new_message["content"] = answer
            self.add_to_history_check_context_limit(new_message)
            turn = {'utterance':new_message['content'], 'think': think, 'speaker': self._system_name, 'turn_id':self._turn_id}
            self._conversation.append(turn)
            
            ### We now ask for the next user input and add it to the history
            print()
            self._turn_id += 1
            userinput = input(self._human_name+":"+str(self._turn_id)+"> ")
            self.add_to_history_check_context_limit({"role": "user", "content": userinput})
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

    def load_from_json(self, filename = "chat_with_llm.json"):
        self._file_name = filename
        f = open(filename)
        self._conversation = json.load(f)

    def print_chat(self):
        for turn in self._conversation:
            print(turn)

    def get_annotator(self, d:dict):
        for item in d:
            if "Annotator" in item:
                annotator = item["Annotator"]
                if not annotator=="auto":
                    return annotator
        return ""

    def clear_annotations_multi_chat(self, labels=[]):
        for conversation in self._conversation:
            for turn in conversation:
                if 'Annotator' in turn:
                    turn['Annotator']=""
                if 'Gold' in turn:
                    turn['Gold']=""

    def annotate_multi_chat(self, labels=[]):
        annotator = ""
        if len(self._conversation)>0:
            annotator = self.get_annotator(self._conversation[0])
        while len(annotator.strip())==0:
            print("Please provide the name of the annotator")
            annotator=input("> ")
        print("The annotator is", annotator)
        print("There will be", len(self._conversation), "conversations to annotate with one of the following labels:", labels)
        print("When you are done, the annotations will be saved in a separate JSON file prefixed with the name of the annotator")
        print("Turns that are already annotated are skipped.")
        for conversation in self._conversation:
            print("Labels", labels)
            print("This conversation has", len(conversation), "turns.")
            input("Press ENTER to start> ")
            for turn in conversation:
                speaker = turn['speaker']
                utterance = turn['utterance']
                turn_id = turn["turn_id"]
                print(turn_id, speaker, ":", utterance)
                if speaker==self._system_name:
                    turn['Gold']="neutral"
                    turn['Annotator']="auto"
                elif not 'Gold' in turn or not turn['Gold']:
                    gold = ""
                    ### we keep getting the user input till one of them matches a label
                    while not gold in labels:
                        gold = input("Enter GOLD label> ")
                    turn['Gold']=gold
                    turn['Annotator']=annotator
                else:
                    print("The GOLD label is",turn['Gold'])
            print("---------- DONE ---------")
        filename = "annotator_"+annotator+"_"+self._file_name
        print("Thank you for annotating. The annotations are saved in", filename)
        self.save_to_json(filename)
        
    def annotate_chat(self, labels=[]):
        annotator = ""
        while len(annotator.strip())==0:
            print("Please provide the name of the annotator")
            annotator=input("> ")
        print("The annotator is", annotator)
        print("There will be", len(self._conversation)/2, "turns to annotate with one of the following labels:", labels)
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
    character="Your answers should be agressive and grumpy."
    #character="Your answers should be uncertain and emotional."
    llm = LLMClient(character=character)
    llm.talk_to_me()

                         
