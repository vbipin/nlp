
#Data class usnign torchtext
#from: https://github.com/pytorch/text/blob/master/test/translation.py
import re
import itertools

import spacy
#need to run
#python -m spacy download en
#python -m spacy download de

from torchtext import datasets
from torchtext import data
from torchtext.data import Field

spacy_de = spacy.load('de')
spacy_en = spacy.load('en')

url = re.compile('(<url>.*</url>)')

def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(url.sub('@URL@', text))]


def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(url.sub('@URL@', text))]

class Data :
    def __init__(self, batch_size=1) :
        DE = data.Field(tokenize=tokenize_de, init_token="<sos>", eos_token="<eos>" )
        EN = data.Field(tokenize=tokenize_en, init_token="<sos>", eos_token="<eos>" )
        self.train, self.val,self.test = datasets.TranslationDataset.splits(
                path='data/multi30k/', train='train',
                validation='val', exts=('.de', '.en'),
                fields=(DE, EN))
        DE.build_vocab(self.train.src,  min_freq=3) #specials=['<sos>','<eos>','<unk>','<pad>'],
        EN.build_vocab(self.train.trg,  max_size=50000) #specials=['<sos>','<eos>','<unk>','<pad>'],
        
        self.src_lang = DE
        self.trg_lang = EN
        
        #self.train_iter, self.val_iter = data.BucketIterator.splits((self.train,self.val), batch_size=batch_size)


    def train_batch(self, batch_size=1, n_data=None, device='cpu') :
        self.train_iter, self.val_iter = data.BucketIterator.splits((self.train,self.val), batch_size=batch_size)
        if n_data is None:
            n_data = len(self.train_iter)
            
        for batch in itertools.islice(self.train_iter, 0, n_data ) :
            yield batch.src.to(device), batch.trg.to(device)
        
    def val_batch(self, batch_size=1, n_data=None, device='cpu') :
        self.train_iter, self.val_iter = data.BucketIterator.splits((self.train,self.val), batch_size=batch_size)
        if n_data is None:
            n_data = len(self.val_iter)
            
        for batch in itertools.islice(self.val_iter, 0, n_data ) :
            yield batch.src.to(device), batch.trg.to(device)    
        
multi30k_data = Data()
