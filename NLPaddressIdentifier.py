#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 10:49:29 2018

@author: gabrielefrattaroli
"""

import pickle
import nltk
import string

from nltk import pos_tag
from nltk import word_tokenize
from nltk.chunk import ChunkParserI
from nltk.chunk import conlltags2tree, tree2conlltags
from nltk.tag import ClassifierBasedTagger
from nltk.tag.util import untag

# Import database 

import os
cwd = os.getcwd()
path = cwd+'/Datasets/IrishAddressDB.pkl' 

pickle_in = open(path,"rb")
IrishDB = pickle.load(pickle_in)

# IOB tag - GPE = Geo-Political Entity
GPE_TAG = "GPE"

# Original Model Pris
class Pris(ChunkParserI):
    def __init__(self, train_sents, **kwargs):  ### This function takes the sentences to be trained on and analyzes the features for each token
        self.tagger = ClassifierBasedTagger(
            train=train_sents,
            feature_detector=self.features,
            **kwargs)

    def parse(self, tagged_sent):
        chunks = self.tagger.tag(tagged_sent)
        iob_triplets = [(w, t, c) for ((w, t), c) in chunks] 
        ### w = word // t= token // c = IOB tag
        ### This transforms the data into triplets as it's the preferred format for evaluating the data
        return conlltags2tree(iob_triplets) # returns the triplets into a nltk.tree format
    
    def features(self, tk, ix, hist):
        
        # Tx = tokens - every token has a word and a PoS tag (w,t)
        # Ix = Index - this is the token's index
        # hist = history - previously classified IOB tags

        # To start the sequence some placeholder (Start1,End1) have been created
        tk = [('[START1]', '[START1]')] + list(tk) + [('[END1]', '[END1]')]
        hist = ['[START1]'] + list(hist)

        # This model uses a Bigram so the index will be shifted with 1
        ix += 1

        word, pos = tk[ix]
        prevword, prevpos = tk[ix - 1]
        nextword, nextpos = tk[ix + 1]
        previob = hist[ix - 1]

        iscapitalized = word[0] in string.ascii_uppercase
        previscapitalized = prevword[0] in string.ascii_uppercase
        nextiscapitalized = prevword[0] in string.ascii_uppercase

        f = {
            'word': word,
            'pos': pos,

            'next-word': nextword,
            'next-pos': nextpos,

            'prev-word': prevword,
            'prev-pos': prevpos,

            'prev-iob': previob,

            'is-capitalized': iscapitalized,
            'prev-is-capitalized': previscapitalized,
            'next-is-capitalized': nextiscapitalized,
        }

        return f
    
### Splitting the database into a Traning/validation set and a Test set
        
import random
random.shuffle(IrishDB)        
size = int(len(IrishDB) * 0.1)
train_sents, test_sents = IrishDB[size:], IrishDB[:size]    


#### Evaluating the score with CrossValidation
#### 10-fold Cvalidation on the training set and the validation set


from sklearn.model_selection import KFold # import KFold        
kf = KFold(n_splits=10)
kf.get_n_splits(train_sents)

totaccuracy=[]
totrecall=[]
totprecision=[]
totf1=[]
from prettytable import PrettyTable
for train_index, validation_index in kf.split(train_sents):
    Kfoldtrain = [IrishDB[i] for i in train_index]
    Kfoldvalidation = [IrishDB[i] for i in validation_index]
    chunker = Pris(Kfoldtrain)
    score = chunker.evaluate([
        conlltags2tree([(w, t, iob) for (w, t), iob in iobs])
        for iobs in Kfoldvalidation
        ]) 
    t = PrettyTable(['Result', 'Value'])
    t.add_row(['Accuracy', score.accuracy()])
    t.add_row(['Recall', score.recall()])
    t.add_row(['Precision', score.precision()])
    t.add_row(['F1', score.f_measure()])
    totaccuracy.append(score.accuracy())
    totrecall.append(score.recall())
    totprecision.append(score.precision())
    totf1.append(score.f_measure())
    
print (sum(totaccuracy) / float(len(totaccuracy)))
print (sum(totrecall) / float(len(totrecall)))
print (sum(totprecision) / float(len(totprecision)))
print (sum(totf1) / float(len(totf1)))

# Improved Model
from nltk.stem.snowball import SnowballStemmer

class Roy(ChunkParserI):
    def __init__(self, train_sents, **kwargs):
        self.tagger = ClassifierBasedTagger(
            train=train_sents,
            feature_detector=self.features,
            **kwargs)

    def parse(self, tagged_sent):
        chunks = self.tagger.tag(tagged_sent)
        iob_triplets = [(w, t, c) for ((w, t), c) in chunks]
        return conlltags2tree(iob_triplets)
    
    def features(self, tk, ix, hist):

        # apply the stemmer
        stemmer = SnowballStemmer('english')

        # Since this model uses a Trigram an additional placeholder has been created [Start2,end2]
        tk = [('[START2]', '[START2]'), ('[START1]', '[START1]')] + list(tk) + [('[END1]', '[END1]'), ('[END2]', '[END2]')]
        hist = ['[START2]', '[START1]'] + list(hist)

        # shift the index with 2 since it's going to use a Trigram
        ix += 2

        word, pos = tk[ix]
        prevword, prevpos = tk[ix - 1]
        prevprevword, prevprevpos = tk[ix - 2]
        nextword, nextpos = tk[ix + 1]
        nextnextword, nextnextpos = tk[ix + 2]
        previob = hist[ix - 1]
        contains_dash = '-' in word
        contains_dot = '.' in word
        isallascii = all([True for c in word if c in string.ascii_lowercase])

        isallcaps = word.isupper()
        nextisallcaps = prevword.isupper()
        previsallcaps = prevword.isupper()
        
        iscapitalized = word[0] in string.ascii_uppercase
        previscapitalized = prevword[0] in string.ascii_uppercase
        nextiscapitalized = prevword[0] in string.ascii_uppercase

        f = {
            'word': word,
            'lemma': stemmer.stem(word),
            'pos': pos,
            'all-ascii': isallascii,

            'next-word': nextword,
            'next-lemma': stemmer.stem(nextword),
            'next-pos': nextpos,

            'next-next-word': nextnextword,
            'nextnextpos': nextnextpos,

            'prev-word': prevword,
            'prev-lemma': stemmer.stem(prevword),
            'prev-pos': prevpos,

            'prev-prev-word': prevprevword,
            'prev-prev-pos': prevprevpos,

            'prev-iob': previob,

            'contains-dash': contains_dash,
            'contains-dot': contains_dot,
            
            'is-allcaps': isallcaps,
            'prev-is-allcaps': previsallcaps,
            'next-is-allcaps': nextisallcaps,

            'is-capitalized': iscapitalized,
            'prev-is-capitalized': previscapitalized,
            'next-is-capitalized': nextiscapitalized,
        }

        return f
    
###### redo 10-fold
        
chunker = Roy(train_sents)    

from sklearn.model_selection import KFold # import KFold        
kf = KFold(n_splits=10)
kf.get_n_splits(train_sents)

totaccuracy=[]
totrecall=[]
totprecision=[]
totf1=[]
from prettytable import PrettyTable
for train_index, validation_index in kf.split(train_sents):
    Kfoldtrain = [IrishDB[i] for i in train_index]
    Kfoldvalidation = [IrishDB[i] for i in validation_index]
    chunker = Roy(Kfoldtrain)
    score = chunker.evaluate([
        conlltags2tree([(w, t, iob) for (w, t), iob in iobs])
        for iobs in Kfoldvalidation
        ]) 
    t = PrettyTable(['Result', 'Value'])
    t.add_row(['Accuracy', score.accuracy()])
    t.add_row(['Recall', score.recall()])
    t.add_row(['Precision', score.precision()])
    t.add_row(['F1', score.f_measure()])
    totaccuracy.append(score.accuracy())
    totrecall.append(score.recall())
    totprecision.append(score.precision())
    totf1.append(score.f_measure())
    
print (sum(totaccuracy) / float(len(totaccuracy)))
print (sum(totrecall) / float(len(totrecall)))
print (sum(totprecision) / float(len(totprecision)))
print (sum(totf1) / float(len(totf1)))

#### Better Results so will go with Chunker 2

chunker = Roy(train_sents)    

#### Evaluating the score

score = chunker.evaluate([
        conlltags2tree([(w, t, iob) for (w, t), iob in iobs])
        for iobs in test_sents
        ])    
    
from prettytable import PrettyTable
t = PrettyTable(['Result', 'Value'])
t.add_row(['Accuracy', score.accuracy()])
t.add_row(['Recall', score.recall()])
t.add_row(['Precision', score.precision()])
t.add_row(['F1', score.f_measure()])
print (t)   

##### Extracting and replacing an address from a sentence

def tree_filter(tree):
    return GPE_TAG == tree.label()

print('Please enter a sentence')
sentence = input()

tagged_tree = chunker.parse(pos_tag(word_tokenize(sentence)))
addresses = list()
for subtree in tagged_tree.subtrees(filter=tree_filter):
    addresses.append(untag(subtree.leaves()))
print (addresses)
newsentence = sentence
for address in addresses:
    for x in address:
        newsentence=newsentence.replace(x,'###')               
print (newsentence)







