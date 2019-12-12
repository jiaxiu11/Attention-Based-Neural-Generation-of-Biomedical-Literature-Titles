#!/usr/bin/env python
# coding: utf-8

# Generate intial word embedding for headlines and description

# The embedding is limited to a fixed vocabulary size (`vocab_size`) but
# a vocabulary of all the words that appeared in the data is built.

# In[1]:


FN = 'vocabulary-embedding'


# In[2]:


seed=42


# In[3]:


vocab_size = 40000


# In[4]:


embedding_dim = 200


# In[5]:


lower = False # dont lower case the text # if we run again use this


# # read tokenized headlines and descriptions

# In[6]:


import pickle
FN0 = 'tokens' # this is the name of the data file which I assume you already have # use data.pkl instead
with open('./data.pkl', 'rb') as fp:
    heads, desc, keywords = pickle.load(fp) # keywords are not used in this project


# In[7]:


if lower:
    heads = [h.lower() for h in heads]


# In[8]:


if lower:
    desc = [h.lower() for h in desc]


# In[9]:


i=0
heads[i]


# In[10]:


desc[i]



# In[12]:


len(heads),len(set(heads))


# In[13]:


len(desc),len(set(desc))


# # build vocabulary

# In[14]:


from collections import Counter
from itertools import chain
def get_vocab(lst):
    vocabcount = Counter(w for txt in lst for w in txt.split())
    vocab = [x[0] for x in sorted(vocabcount.items(), key=lambda x: -x[1])]
    return vocab, vocabcount


# In[15]:


vocab, vocabcount = get_vocab(heads+desc)


# most popular tokens

# In[16]:


print (vocab[:50])
print ('...',len(vocab))


# # Index words

# In[18]:


empty = 0 # RNN mask of no data
eos = 1  # end of sentence
start_idx = eos+1 # first real word


# In[19]:


def get_idx(vocab, vocabcount):
    word2idx = dict((word, idx+start_idx) for idx,word in enumerate(vocab))
    word2idx['<empty>'] = empty
    word2idx['<eos>'] = eos
    
    idx2word = dict((idx,word) for word,idx in word2idx.items())

    return word2idx, idx2word


# In[20]:


word2idx, idx2word = get_idx(vocab, vocabcount)


# # Word Embedding

# ## read GloVe

import numpy as np
glove_n_symbols = 2351706


# In[22]:


# as the word vector file is large(about 5GB), it takes 5-10 minutes to load the vectors into RAM on my laptop.
# added print statements to track progress. Do not exeucate later cells until this cell finishes.
glove_index_dict = {}
glove_embedding_weights = np.empty((glove_n_symbols, embedding_dim))
globale_scale=.1
print("start reading word vector file")
with open("./PubMed-w2v.txt", 'r') as fp:
    i = 0
    is_first_line = True;
    for l in fp:
        if is_first_line:
            is_first_line = False
            continue
        l = l.strip().split()
        w = l[0]
        glove_index_dict[w] = i
        glove_embedding_weights[i,:] = [float(x) for x in l[1:]]
        i += 1
print("finished loading word vectors into dataframe and dict")
glove_embedding_weights *= globale_scale
print("finished scaling dataframe")


# In[30]:


glove_embedding_weights.std()


# In[23]:


for w,i in glove_index_dict.copy().items():
    w = w.lower()
    if w not in glove_index_dict:
        glove_index_dict[w] = i
print("done")


# ## embedding matrix

# use GloVe to initialize embedding matrix

# In[24]:


import numpy as np

# generate random embedding with same scale as glove
np.random.seed(seed)
shape = (vocab_size, embedding_dim)
scale = glove_embedding_weights.std()*np.sqrt(12)/2 # uniform and not normal
embedding = np.random.uniform(low=-scale, high=scale, size=shape)
print ('random-embedding/glove scale', scale, 'std', embedding.std())

# copy from glove weights of words that appear in our short vocabulary (idx2word)
c = 0
for i in range(vocab_size):
    w = idx2word[i]
    g = glove_index_dict.get(w, glove_index_dict.get(w.lower()))
    if g is None and w.startswith('#'): # glove has no hastags (I think...)
        w = w[1:]
        g = glove_index_dict.get(w, glove_index_dict.get(w.lower()))
    if g is not None:
        embedding[i,:] = glove_embedding_weights[g,:]
        c+=1
print ('number of tokens, in small vocab, found in glove and copied to embedding', c,c/float(vocab_size))


# lots of word in the full vocabulary (word2idx) are outside `vocab_size`.
# Build an alterantive which will map them to their closest match in glove but only if the match
# is good enough (cos distance above `glove_thr`)

# In[25]:


glove_thr = 0.5


# In[26]:


word2glove = {}
for w in word2idx:
    if w in glove_index_dict:
        g = w
    elif w.lower() in glove_index_dict:
        g = w.lower()
    elif w.startswith('#') and w[1:] in glove_index_dict:
        g = w[1:]
    elif w.startswith('#') and w[1:].lower() in glove_index_dict:
        g = w[1:].lower()
    else:
        continue
    word2glove[w] = g


# for every word outside the embedding matrix find the closest word inside the mebedding matrix.
# Use cos distance of GloVe vectors.
# 
# Allow for the last `nb_unknown_words` words inside the embedding matrix to be considered to be outside.
# Dont accept distances below `glove_thr`

# In[27]:


normed_embedding = embedding/np.array([np.sqrt(np.dot(gweight,gweight)) for gweight in embedding])[:,None]

nb_unknown_words = 100

glove_match = []
for w,idx in word2idx.items():
    if idx >= vocab_size-nb_unknown_words and w.isalpha() and w in word2glove:
        gidx = glove_index_dict[word2glove[w]]
        gweight = glove_embedding_weights[gidx,:].copy()
        # find row in embedding that has the highest cos score with gweight
        gweight /= np.sqrt(np.dot(gweight,gweight))
        score = np.dot(normed_embedding[:vocab_size-nb_unknown_words], gweight)
        while True:
            embedding_idx = score.argmax()
            s = score[embedding_idx]
            if s < glove_thr:
                break
            if idx2word[embedding_idx] in word2glove :
                glove_match.append((w, embedding_idx, s)) 
                break
            score[embedding_idx] = -1
glove_match.sort(key = lambda x: -x[2])
print ('# of glove substitutes found', len(glove_match))


# manually check that the worst substitutions we are going to do are good enough

# In[28]:


for orig, sub, score in glove_match[-10:]:
    print (score, orig,'=>', idx2word[sub])


# build a lookup table of index of outside words to index of inside words

# In[29]:


glove_idx2idx = dict((word2idx[w],embedding_idx) for  w, embedding_idx, _ in glove_match)


# # Data

# In[30]:


Y = [[word2idx[token] for token in headline.split()] for headline in heads]
len(Y)




# In[31]:


X = [[word2idx[token] for token in d.split()] for d in desc]
len(X)



# In[32]:


import pickle
with open('%s.pkl'%FN,'wb') as fp:
    pickle.dump((embedding, idx2word, word2idx, glove_idx2idx),fp,2)


# In[33]:


import pickle
with open('%s.data.pkl'%FN,'wb') as fp:
    pickle.dump((X,Y),fp,2)


# In[ ]:




