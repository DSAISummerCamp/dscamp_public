import os, sys 
import numpy as np
import torch
import copy
from transformers import GPT2Tokenizer, GPT2LMHeadModel


MAX_LENGTH = int(10000)
class Chatbot():
    def __init__(self, temperature = 1.0, length = 50, k = 3, p = 0.6, repetition_penalty = 3.0):
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.temperature = temperature
        self.length = length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_gpu = torch.cuda.device_count()
        self.MAX_LENGTH = int(10000)

        prompt = " "
        self.stop_token = '.'
        self.repetition_penalty = repetition_penalty
        self.k = k
        self.p = p
        paddin_text = ""
        seed = 42
        num_return_senquences = 1

    def ask_question(self):
        prompt_text = input("Start Chatting...... \n")
        self.prompt_text = prompt_text
        return prompt_text

    def chat_with_bot(self):
        prompt_text = input()
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
                num_return_sequences=1)


              #print("::::YOURS OPTIONS ARE :::")
              #for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
                  #print("=== GENERATED SEQUENCE {} ===".format(generated_sequence_idx + 1))
            generated_sequence = output_sequences[0].tolist()

                  # Decode text
            text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

            inv_text = text[::-1]
            stop_token_index = len(text) - inv_text.find(stop_token)

                  # Remove all text after the stop token
            text = text[:stop_token_index if stop_token_index < len(text) else None]
            reply = text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)) :]
            
            self.bot_reply(reply)
                  #print(generated_sequence_idx)
            prompt_text = self.chat_with_bot()

            new_text = copy.deepcopy(prompt_text)
            new_text = new_text.lower()
            if new_text.find('stop') != -1:
                print("Stopping Chatbot....")
                break

