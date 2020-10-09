import json
import requests


def read_token():
    tokens_path = './data/tokens.json'
    with open(tokens_path) as f:
        tokens = json.load(f)

    return tokens['CLTL_token']


def read_qa(qa_path):
    with open(qa_path) as f:
        qa_data = json.load(f)

    return qa_data


class BotHandler:
    def __init__(self, token):
        self.token = token
        self.api_url = "https://api.telegram.org/bot{}/".format(token)

    def get_all_messages(self, offset=None, timeout=200):
        """ Function to get all messages sent to the bot """
        method = 'getUpdates'
        params = {'timeout': timeout, 'offset': offset}
        resp = requests.get(self.api_url + method, params)
        
        if resp.status_code == 401:  
            print("Error: Unauthorized access. Please check the Bot's token")
        
        if resp.status_code == 200:
            if resp.json() and 'result' in resp.json():
                return resp.json()['result']
            else:
                print("Error: No messages found")

    def filter_messages_by(self, update, chat_id):
        """ Function to filter messages by user id"""
        return 'message' in update.keys() and update['message']['chat']['id'] == chat_id

    def get_last_message_by(self, chat_id):
        """ Function to get the last message sent to the bot by a specific user"""
        messages = self.get_all_messages()
        if messages:
            messages_by_user = list(filter(lambda m: self.filter_messages_by(m, chat_id), messages))
            
            if messages_by_user:
                last_message = None
                if messages_by_user and 'message' in messages_by_user[-1].keys():
                    try:
                        last_message = messages_by_user[-1]['message']['text']
                    except:
                        last_message = ''

                    return last_message
            else:
                print("Error: No messages by this user found")

    def send_message_to(self, chat_id, text):
        """ Function to send a message from the bot to a specific user"""
        params = {'chat_id': chat_id, 'text': text}
        method = 'sendMessage'
        resp = requests.post(self.api_url + method, params)
