import os, sys 
import numpy as np
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel


MAX_LENGTH = int(10000)
class GenerateSentence():
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
          #print("=== GENERATED SEQUENCE {} ===".format(generated_sequence_idx + 1))
          generated_sequence = generated_sequence.tolist()

          # Decode text
          text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

          inv_text = text[::-1]
          stop_token_index = len(text) - inv_text.find(stop_token)

          # Remove all text after the stop token
          text = text[:stop_token_index if stop_token_index < len(text) else None]
          
          #text = text[: text.find(stop_token)+1 if stop_token else None]

          # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
          total_sequence = (
              prompt_text + text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)) :]
          )
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
        total_sentence = generated_seq_dir['A']
      elif userchoice == 'b':
        total_sentence = generated_seq_dir['B']
      elif userchoice == 'c':
        total_sentence = generated_seq_dir['C']
      elif userchoice == 'd':
        total_sentence = total_sequence + " " + self.insert_text()
      else:
        print("\n\n================  COMPLETE ESSAY  ======================")
        print(total_sentence)
        break
      print(" \n \n \n ****** INCOMPLETE ESSAY *******\n{} \n\n\n".format(total_sentence))
      prompt_text = (total_sentence)