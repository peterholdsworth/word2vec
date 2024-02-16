import string
import sentencepiece as spm
from datasets import load_dataset, load_from_disk

class Tokenizer:
  def __init__(self, corpus=None, freq_threshold=1):
    self.sp = spm.SentencePieceProcessor(model_file="tinystories.model")
    dataset = load_from_disk('tinystories.data')
    self.corpus = [story['text'] for story in dataset['train']] 

  def encode(self, sentence):
    return self.sp.encode_as_ids(sentence)
  
  def decode(self, indices):
    return self.sp.decode(indices)

if __name__ == '__main__':
  dataset = load_dataset('roneneldan/TinyStories')
  dataset.save_to_disk('tinystories.data')
  stories = [story['text'] for story in dataset['train']] 
  with open("tinystories.txt", "w", encoding="utf-8") as f:
    for story in stories:
        f.write(story + "\n")
  spm.SentencePieceTrainer.train(
    input='tinystories.txt', 
    model_prefix='tinystories', 
    vocab_size=32000, 
    model_type='unigram'
  )
  sp = spm.SentencePieceProcessor(model_file="tinystories.model")
  sp.encode_as_ids('Mary had a little lamb')
  