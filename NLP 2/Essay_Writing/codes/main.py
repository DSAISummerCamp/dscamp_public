import os, sys 
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from nltk import word_tokenize
import tqdm
from pprint import pprint

MAX_LENGTH = int(10000)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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


class GenerateSentence():
    def __init__(self, dataset_path = None, temperature = 1.0, length = 50, k = 3, p = 0.6, repetition_penalty = 3.0, topics = ['history', 'machine_learning', 'ai']):
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.temperature = temperature
        self.length = length
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.n_gpu = torch.cuda.device_count()
        self.MAX_LENGTH = int(10000)
        self.topics = topics
        prompt = " "
        self.stop_token = '.'
        self.repetition_penalty = repetition_penalty
        self.k = k
        self.p = p
        self.train_max_seq_len = 512
        paddin_text = ""
        seed = 42
        num_return_senquences = 1
        self.dataset_path = dataset_path
        pprint("Language Generator loaded successfully....")


    def train_model(self, dataloader):
        model = self.model
        device  = self.device
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-6, eps=1e-08)
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
                with open(filepath, 'r') as trainfile:
                    textdata = trainfile.read().replace('\n', '')
                    tokens = word_tokenize(textdata)
                    max_seq_len = self.train_max_seq_len
                    training_data_len = int(len(tokens)/max_seq_len)
                    tokens_lst = [tokens[max_seq_len*i:max_seq_len*(i+1)] for i in range(training_data_len)]
                    training_dataset = Textdataset(tokens_lst, tokenizer=self.tokenizer)
                    train_data_loader = torch.utils.data.DataLoader(dataset=training_dataset, batch_size=2)
                    self.train_model(train_data_loader)

    def train_on_topics(self, topic = 'history'):
        topic = topic.lower()
        if topic not in self.topics:
            print("The topic provided is not available")
            print("Please select from the following topics")
            pprint("Topics {}".format(self.topics))
        else:
            if topic == 'history':
                self.train(filename='history.txt')
            elif topic == 'machine_learning':
                self.train(filename='ml.txt')
            elif topic == 'ai':
                self.train(filename='ai.txt')

    def start_writing(self):
        prompt_text = input("Write the first sentence >>> ")
        self.prompt_text = prompt_text
  
    def insert_text(self):
        prompt_text = input("Insert your own sentence >>> ")
        return prompt_text

    def choose_option(self):
        option = input("Choose your option >>> ")
        return option

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
        choices = ['A', 'B', 'C', 'D', 'E']
        length = self.length
        device = self.device
        length = self.adjust_length_to_model(length)
        prompt_text = self.prompt_text
        model = self.model
        model = model.to(device)
        tokenizer = self.tokenizer
        stop_token = self.stop_token
        encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")
        encoded_prompt = encoded_prompt.to(device)

        while True:
            encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")
            encoded_prompt = encoded_prompt.to(device)

            output_sequences = model.generate(
                  input_ids=encoded_prompt,
                  max_length=length + len(encoded_prompt[0]),
                  temperature=self.temperature,
                  top_k=self.k,
                  top_p=self.p,
                  repetition_penalty=self.repetition_penalty,
                  do_sample=True,
                  num_return_sequences=3)
            generated_sequences = []

            generated_seq_dir = {}

            print("::::YOURS OPTIONS ARE :::")
            for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
                generated_sequence = generated_sequence.tolist()

                  # Decode text
                text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

                inv_text = text[::-1]
                stop_token_index = len(text) - inv_text.find(stop_token)

                  # Remove all text after the stop token
                text = text[:stop_token_index if stop_token_index < len(text) else None]

                  #text = text[: text.find(stop_token)+1 if stop_token else None]

                  # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
                total_sequence = (prompt_text + text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)) :])
                generated_sequences.append(total_sequence)
                generated_seq_dir[choices[generated_sequence_idx]] = total_sequence

                text_option = text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)) :]
                  #print(generated_sequence_idx)
                print(choices[generated_sequence_idx] + ". --> {}".format(text_option))
                print()

            print(choices[generated_sequence_idx + 1] + ". --> {}".format("Write your own sentences"))
            print(choices[generated_sequence_idx + 2] + ". --> {}".format("STOP ESSAY WRITING"))

            print(generated_seq_dir)
            userchoice = self.choose_option()
            userchoice = userchoice.lower()
            while True:
                if userchoice not in ['a', 'b', 'c', 'd', 'e']:
                    print("Please enter a valid option ")
                    userchoice = self.choose_option()
                    userchoice = userchoice.lower()
                else:
                    break
            if userchoice == 'a':
                total_sentence = generated_seq_dir['A'].replace('\n', ' ')
            elif userchoice == 'b':
                total_sentence = generated_seq_dir['B'].replace('\n', ' ')
            elif userchoice == 'c':
                total_sentence = generated_seq_dir['C'].replace('\n', ' ')
            elif userchoice == 'd':
                total_sentence = total_sequence + " " + self.insert_text()
            else:
                print("\n\n================  COMPLETE ESSAY  ======================")
                print(total_sentence)
                break
            print(" \n \n \n ****** ESSAY TILL THIS POINT *******\n{} \n\n\n".format(total_sentence))
            prompt_text = (total_sentence)