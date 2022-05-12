import os, sys
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoModelWithLMHead, AutoTokenizer
from nltk import word_tokenize
import tqdm
from pprint import pprint
import copy

MAX_LENGTH = int(128)

class Textdataset(Dataset):
    def __init__(self, texts, tokenizer, max_seq_len=512):
        self.tokenizer = tokenizer
        self.x = texts
        self.max_seq_len = max_seq_len
        self.special_token = '<|endoftext|>'

    def __getitem__(self, index):
        input = self.x[index]
        max_seq_len = self.max_seq_len
        input = input[:max_seq_len - 2]
        input = [self.special_token] + input + [self.special_token]
        input = input + [0] * (max_seq_len - len(input))
        input_dict = self.tokenizer.encode_plus(input, add_special_tokens=True)
        inputids = torch.tensor(input_dict['input_ids']).long()
        attention_mask = torch.tensor(input_dict['attention_mask']).long()
        token_type_ids = torch.tensor(input_dict['token_type_ids']).long()

        return inputids, attention_mask, token_type_ids

    def __len__(self):
        return len(self.x)


class Chatbot():
    def __init__(self, dataset_path = None, temperature = 1.1, length = 50, k = 3, p = 0.6, repetition_penalty = 10.0, size='small', topics  = ['movie-animation','movie-horror', 'movie-action', 'movie-scifi']):
        if size=='small':
            self.model = AutoModelWithLMHead.from_pretrained("microsoft/DialoGPT-small")
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
        elif size=='medium':
            self.model = AutoModelWithLMHead.from_pretrained("microsoft/DialoGPT-medium")
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        self.temperature = temperature
        self.length = length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_gpu = torch.cuda.device_count()
        self.MAX_LENGTH = int(1000)

        prompt = " "
        self.stop_token = '.'
        self.repetition_penalty = repetition_penalty
        self.k = k
        self.p = p
        self.topics = topics
        paddin_text = ""
        seed = 42
        self.train_max_seq_len = 512
        num_return_senquences = 1
        self.dataset_path = dataset_path
        pprint("Language Dialogue generator loaded successfully....")

    def train_model(self, dataloader):
        model = self.model
        device  = self.device
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-7, eps=1e-08)
        model.train()
        total_len = len(dataloader)
        data_iterator = tqdm.tqdm_notebook(dataloader, total = total_len)
        for data in data_iterator:
            inputs = {'input_ids': data[0].to(device), 'attention_mask': data[1].to(device),
                      'token_type_ids': data[2].to(device)}
            outputs = model(**inputs, labels=inputs['input_ids'])
            loss = outputs[0]
            loss.backward(retain_graph = True)
            optimizer.step()
            optimizer.zero_grad()
            data_iterator.set_description("Loss {}".format(loss.item()))

    def train(self, filename = None):
        if filename is None:
            print("Please provide a filename...")
        else:
            if self.dataset_path is None:
                filepath = os.path.join(os.getcwd(), 'datasets', filename)
            else:
                filepath = os.path.join(self.dataset_path, filename)

            if not os.path.exists(filepath):
                print("FILENAME DOES NOT EXIST")
                return

            else:
                with open(filepath, 'rb') as trainfile:
                    textdata = trainfile.read()
                    textdata = str(textdata).replace('\n', '')
                    tokens = word_tokenize(textdata)
                    max_seq_len = self.train_max_seq_len
                    training_data_len = int(len(tokens)/max_seq_len)
                    tokens_lst = [tokens[max_seq_len*i:max_seq_len*(i+1)] for i in range(training_data_len)]
                    training_dataset = Textdataset(tokens_lst, tokenizer=self.tokenizer)
                    train_data_loader = torch.utils.data.DataLoader(dataset=training_dataset, batch_size=1)
                    self.train_model(train_data_loader)

    def train_on_topics(self, topic = 'movie-animation'):
        topic = topic.lower()
        if topic not in self.topics:
            print("The topic provided is not available")
            print("Please select from the following topics")
            pprint("Topics {}".format(self.topics))
        else:
            if topic == 'movie-animation':
                self.train(filename='animlines.txt')
            elif topic == 'movie-scifi':
                self.train(filename='scifilines.txt')
            elif topic == 'movie-horror':
                self.train(filename='horrorlines.txt')
            elif topic == 'movie-action':
                self.train(filename='actionlines.txt')
            else:
                print("The topic is not supported...")

    def ask_question(self):
        prompt_text = input("Start Chatting...... \n")
        self.prompt_text = prompt_text
        return prompt_text

    def chat_with_bot(self):
        prompt_text = input("You >>> ")
        self.prompt_text = prompt_text
        return prompt_text

    def bot_reply(self, text):
        bot_reply = "BOT >>> " + text
        print(bot_reply)

    def adjust_length_to_model(self, length):
        max_sequence_length = self.model.config.max_position_embeddings

        if length < 0 and max_sequence_length > 0:
            length = max_sequence_length
        elif 0 < max_sequence_length < length:
            length = max_sequence_length  # No generation bigger than model size
        elif length < 0:
            length = MAX_LENGTH  # avoid infinite loop
        return length

    def generate_sentences(self):
        length = self.length
        device = self.device
        length = self.adjust_length_to_model(length)
        prompt_text = self.ask_question()
        model = self.model
        model = model.to(device)
        tokenizer = self.tokenizer
        stop_token = self.stop_token
        #encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")
        #encoded_prompt = encoded_prompt.to(device)
        step = 0
        while True:
            encoded_prompt = tokenizer.encode(prompt_text + tokenizer.eos_token, return_tensors='pt')
            encoded_prompt = encoded_prompt.to(device)
            bot_input_ids = torch.cat([chat_history_ids, encoded_prompt], dim=-1) if step > 0 else encoded_prompt
            chat_history_ids = model.generate(
                input_ids=bot_input_ids,
                max_length=128,
                pad_token_id=tokenizer.eos_token_id)

        # Decode text
            reply = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
            reply=reply.replace('intercourse','fight')
            self.bot_reply(reply)
            prompt_text = self.chat_with_bot()

            new_text = copy.deepcopy(prompt_text)
            new_text = new_text.lower()
            if new_text.find('stop') != -1:
                print("Bye..Have a nice day!!")
                break
            step += 1
