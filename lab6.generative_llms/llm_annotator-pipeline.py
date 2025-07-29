import json
from datetime import datetime
from transformers import pipeline

#https://ollama.com/models
#ollama pull qwen3:1.7b

ekman_labels = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]

pipe = pipeline("text-generation", model="Qwen/Qwen3-1.7B-Base")

class LLMAnnotator:
    
    def __init__(self, model="Qwen/Qwen3-1.7B-Base", labels=ekman_labels, examples=[], max_context=5):
        self._client = pipeline("text-generation", model=model)
        self._max_context = max_context
        self._history = []
        self._instruct = []
        self.create_label_instruct(labels, examples)

    def create_label_instruct(self, labels =[], examples=[]):
        self._instruct = [{"role": "system", "content": "You are an intelligent assistant."}]
        self._instruct.append({"role": "system", "content": "You will receive utterances from a conversation as Input in JSON format."})
        self._instruct.append({"role": "system", "content": "You need to determine the emotion of the last utterance."})
        self._instruct.append({"role": "system", "content": "Only use one of the following labels:{}.".format(str(labels))})
        self._instruct.append({"role": "system", "content": "Only determine the emotion of LAST utterance, use the preceding utterances as context."})
        self._instruct.append({"role": "system", "content": "Output the most appropriate label in JSON format."})
        self._instruct.append({"role": "system", "content": "Do not output anything else."})
        if examples:
            self._instruct.append({"role": "system", "content": "Here are a few examples:"})
            for example in examples:
                self._instruct.append({"role": "user", "content": example["Input"]})
                self._instruct.append({"role": "system", "content": example["Output"]})
        print("My instructions are:", self._instruct)

    def annotate_conversation(self, input=[]):
        report = 5
        annotations = []
        counter = 0
        start = datetime.now()
        previous = start
        print("Annotating a conversation with {} utterances".format(len(input)))
        for text in input:
            annotation = self.annotate(text)
            annotations.append(annotation)
            counter+=1
            if counter % report == 0:
                now  = datetime.now()
                print('Processed', report, 'in', (now - previous).seconds, 'seconds')
                print("Processed", counter, "turns in total out of", len(input))
                previous = now
        return annotations
  
    def annotate(self, utterance):
        annotation ={}
        self._history.append({"role": "user", "content": "Input: {}".format(utterance)})

        ## if the history exceeds the maximum context length, we trim it by one
        if len(self._history)>self._max_context:
            self._history = self._history[1:]

        prompt = self._instruct+self._history
        response = pipe(prompt)
        #response = self._client.invoke(prompt)
        
        ### We need to remove the <think></think> part from the output
        end_of_think = response.content.find("</think>")
        answer = response
        if end_of_think>0:
            think = response.content[:end_of_think+8]
            answer = response.content[end_of_think+8:].replace("\n", "")
        annotation={"Input": utterance, "Output": answer}
        return annotation

if __name__ == "__main__":
    ekman_labels = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]
    labels = ["positive", "negative", "neutral"]
    input = ["I am very happy", "I am surprised", "What can I do about it?", "Now I am having real fun.", "It make me sad and depressed", "Sorry to hear that"]
    examples = [{"Input": "I love dogs", "Output": "joy"}, {"Input": "I hate cats", "Output": "disgust"}]
    llm_annotator  = LLMAnnotator(labels=ekman_labels, examples=examples, max_context=3)
    annotations = llm_annotator.annotate_conversation(input)
    for annotation in annotations:
        print(annotation)
                         
