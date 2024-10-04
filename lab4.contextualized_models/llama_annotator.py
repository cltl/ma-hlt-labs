from openai import OpenAI
import json

ekman_labels = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]

class LlamaAnnotator:
    
    def __init__(self, url="http://localhost:9001/v1", labels=ekman_labels, examples=[]):
        self._client = OpenAI(base_url="http://localhost:9001/v1", api_key="not-needed")
        self._history = []
        self._instruct = []
        self.create_label_instruct(labels, examples)

    def create_label_instruct(self, labels =[], examples=[]):
        self._instruct = [{"role": "system", "content": "You are an intelligent assistant."}]
        self._instruct.append({"role": "system", "content": "You will receive utterances from a conversation as Input in JSON format."})
        self._instruct.append({"role": "system", "content": "You need to decide whether one of the following labels apply:{}".format(str(labels))})
        self._instruct.append({"role": "system", "content": "Output the most appropriate label in JSON format."})
        self._instruct.append({"role": "system", "content": "Do not output anything else."})
        if examples:
            self._instruct.append({"role": "system", "content": "Here are a few examples:"})
            for example in examples:
                self._instruct.append({"role": "user", "content": example["Input"]})
                self._instruct.append({"role": "system", "content": example["Output"]})
        print("My instructions are:", self._instruct)

    def annotate(self, input=[]):
        annotations = []
        counter = 0
        for text in input:
            counter+=1
            if counter % 10 == 0:
                print("Processed", counter, "turns out of", len(input))
            self._history = self._instruct
            self._history.append({"role": "user", "content": "Input: {}".format(text)})
            ### We call the openai client with a low temperature for the first turn
            completion = self._client.chat.completions.create(
                model="local-model", # this field is currently unused
                messages=self._history,
                temperature=0.0,
                stream=True,
            )
            response = ""
            for chunk in completion:
                if chunk.choices[0].delta.content:
                    response += chunk.choices[0].delta.content
            annotations.append({"Input": text, "Output": response})
        return annotations

if __name__ == "__main__":
    url="http://localhost:9001/v1"
    ekman_labels = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]
    labels = ["positive", "negative", "neutral"]

    input = ["I am very happy", "I am surprised", "What can I do about it"]
    examples = [{"Input": "I love dogs", "Output": "joy"}, {"Input": "I hate cats", "Output": "disgust"}]
    llama  = LlamaAnnotator(url = url, labels=ekman_labels, examples=examples)
    annotations = llama.annotate(input)
    print(annotations)
                         
