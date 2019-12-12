#!/usr/bin/env python
# coding: utf-8

# # Test

# Imports

# In[1]:


import os
os.environ['THEANO_FLAGS'] = 'device=cpu,floatX=float32'

#import theano

import keras
keras.__version__

from sklearn.model_selection import train_test_split

import cPickle as pickle

import numpy as np
from keras.preprocessing import sequence
from keras.utils import np_utils
import random, sys

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, RepeatVector
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.regularizers import l2
from keras.layers.core import Lambda
import keras.backend as K

import h5py

import sys
import Levenshtein

from nltk.translate.bleu_score import sentence_bleu


# #### Setting
# Set `TESTING` to `True` if you want to use test rather than validation set
# 
# Set `MODE`:
# -  `0` for using the first 50 words of each article
# -  `1` for using the first 25 words and last 50 words of each article
# -  `2` for using the first 50 words and last 25 words of each article
# -  `3` for using the first 50 words and last 50 words of each article

# In[3]:


TESTING = False
#TESTING = True
TRAINTEST = False
#TRAINTEST = True

MODE = 0
#MODE = 1
# MODE = 2
#MODE = 3


# ### Data
# Load validation/test set

# In[5]:


seed = 40

#with open('data.pkl', 'rb') as fp:
#    Y, X, keywords = pickle.load(fp)
with open('%s.data.pkl'%'vocabulary-embedding', 'rb') as fp:
    X, Y = pickle.load(fp)
    
# TODO: change back
# nb_val_samples = 3000
nb_val_samples = 3000


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=nb_val_samples, random_state=seed)
len(X_train), len(Y_train), len(X_test), len(Y_test)

X_val = X_test[:nb_val_samples/2]
Y_val = Y_test[:nb_val_samples/2]
X_test = X_test[nb_val_samples/2:]
Y_test = Y_test[nb_val_samples/2:] 

if TESTING:
    X_using = X_test
    Y_using = Y_test
else:
    X_using = X_val
    Y_using = Y_val
if TRAINTEST:
    X_using = []
    Y_using = []
    for i in range(nb_val_samples / 2):
        article = np.random.randint(0, len(X_train))
        X_using.append(X_train[article])
        Y_using.append(Y_train[article])

len(X_using), len(Y_using)


# ### Read word embedding

# In[6]:


#import cPickle as pickle

with open('%s.pkl'%'vocabulary-embedding', 'rb') as fp:
    embedding, idx2word, word2idx, glove_idx2idx = pickle.load(fp)
vocab_size, embedding_size = embedding.shape


# In[7]:


nb_unknown_words = 10


# In[8]:


print 'dimension of embedding space for words',embedding_size
print 'vocabulary size', vocab_size, 'the last %d words can be used as place holders for unknown/oov words'%nb_unknown_words
print 'total number of different words',len(idx2word), len(word2idx)
print 'number of words outside vocabulary which we can substitue using glove similarity', len(glove_idx2idx)
print 'number of words that will be regarded as unknonw(unk)/out-of-vocabulary(oov)',len(idx2word)-vocab_size-len(glove_idx2idx)


# In[9]:


for i in range(nb_unknown_words):
    idx2word[vocab_size-1-i] = '<%d>'%i


# In[10]:


for i in range(vocab_size-nb_unknown_words, len(idx2word)):
    idx2word[i] = idx2word[i]+'^'


# In[11]:


empty = 0
eos = 1
idx2word[empty] = '_'
idx2word[eos] = '~'


# In[12]:


def prt(label, x):
    print label+':',
    for w in x:
        print idx2word[w],
    print


# ### Model
# 
# Parameters

# In[13]:


if MODE == 0 or MODE == 1:
    maxlend=50 # 0 - if we dont want to use description at all
elif MODE == 2:
    maxlend=75
elif MODE == 3:
    maxlend = 100
else:
    raise "invalid MODE"
maxlenh=25
maxlen = maxlend + maxlenh
rnn_size = 512
rnn_layers = 3  # match FN1
batch_norm=False


# In[14]:


activation_rnn_size = 40 if maxlend else 0


# In[15]:


# training parameters
seed=40
p_W, p_U, p_dense, weight_decay = 0, 0, 0, 0
optimizer = 'adam'
batch_size=32


# In[17]:

nb_train_samples = 60000
nb_val_samples = 2000


# In[18]:


# seed weight initialization
random.seed(seed)
np.random.seed(seed)


# In[19]:


regularizer = l2(weight_decay) if weight_decay else None


# RNN model
# 
# start with a stacked LSTM, which is identical to the bottom of the model used in training

# In[20]:


rnn_model = Sequential()
rnn_model.add(Embedding(vocab_size, embedding_size,
                        input_length=maxlen,
#                         batch_input_shape=(batch_size,maxlen),
                        embeddings_regularizer=regularizer, weights=[embedding], mask_zero=True,
                        name='embedding_1'))
for i in range(rnn_layers):
    lstm = LSTM(rnn_size, return_sequences=True, # batch_norm=batch_norm,
                kernel_regularizer=regularizer, recurrent_regularizer=regularizer,
                bias_regularizer=regularizer, dropout=p_W, recurrent_dropout=p_U,
                name='lstm_%d'%(i+1)
                  )
    rnn_model.add(lstm)
    rnn_model.add(Dropout(p_dense, name='dropout_%d'%(i+1)))


# Load 
# 
# use the bottom weights from the trained model, and save the top weights for later

# In[22]:


# rnn_model.load_weights('data/%s.hdf5'%'train',by_name=True)
rnn_model.load_weights('res/%s.hdf5'%'train3',by_name=True)


# In[24]:


# with h5py.File('data/%s.hdf5'%'train', mode='r') as f:
with h5py.File('res/%s.hdf5'%'train3', mode='r') as f:
    if 'layer_names' not in f.attrs and 'model_weights' in f:
        f = f['model_weights']
    weights = [np.copy(v) for v in f['time_distributed_1']['time_distributed_1'].values()]
weights = np.array([weights[1], weights[0]])


# In[25]:


map(lambda x: x.shape, weights)


# Headline model
# 
# A special layer that reduces the input just to its headline part (second half).
# For each word in this part it concatenate the output of the previous layer (RNN)
# with a weighted average of the outputs of the description part.
# In this only the last `rnn_size - activation_rnn_size` are used from each output.
# The first `activation_rnn_size` output is used to computer the weights for the averaging.

# In[26]:


def simple_context(X, mask, n=activation_rnn_size, maxlend=maxlend, maxlenh=maxlenh):
    desc, head = X[:,:maxlend,:], X[:,maxlend:,:]
    head_activations, head_words = head[:,:,:n], head[:,:,n:]
    desc_activations, desc_words = desc[:,:,:n], desc[:,:,n:]
    
    # RTFM http://deeplearning.net/software/theano/library/tensor/basic.html#theano.tensor.batched_tensordot
    # activation for every head word and every desc word
    #activation_energies = K.batch_dot(head_activations, desc_activations, axes=([2],[2]))
    activation_energies = K.batch_dot(head_activations, desc_activations, axes=(2,2))
    # make sure we dont use description words that are masked out
    activation_energies = activation_energies + -1e20*K.expand_dims(1.-K.cast(mask[:, :maxlend],'float32'),1)
    
    # for every head word compute weights for every desc word
    activation_energies = K.reshape(activation_energies,(-1,maxlend))
    activation_weights = K.softmax(activation_energies)
    activation_weights = K.reshape(activation_weights,(-1,maxlenh,maxlend))

    # for every head word compute weighted average of desc words
    #desc_avg_word = K.batch_dot(activation_weights, desc_words, axes=([2],[1]))
    desc_avg_word = K.batch_dot(activation_weights, desc_words, axes=(2,1))
    return K.concatenate((desc_avg_word, head_words))


# In[27]:


model = Sequential()
model.add(rnn_model)

if activation_rnn_size:
    model.add(Lambda(simple_context,
                     mask = lambda inputs, mask: mask[:,maxlend:],
                     output_shape = lambda input_shape: (input_shape[0], maxlenh, 2*(rnn_size - activation_rnn_size)),
                     name='simplecontext_1'))


# In[28]:


# we are not going to fit so we dont care about loss and optimizer
model.compile(loss='categorical_crossentropy', optimizer='adam')


# In[29]:


n = 2*(rnn_size - activation_rnn_size)
n


# perform the top dense of the trained model in numpy so we can play around with exactly how it works

# In[30]:


# out very own softmax
def output2probs(output):
    output = np.dot(output, weights[0]) + weights[1]
    output -= output.max()
    output = np.exp(output)
    output /= output.sum()
    return output


# Test

# In[31]:


def lpadd(x, maxlend=maxlend, eos=eos):
    """left (pre) pad a description to maxlend and then add eos.
    The eos is the input to predicting the first word in the headline
    """
    assert maxlend >= 0
    if maxlend == 0:
        return [eos]
    n = len(x)
    if n > maxlend:
        x = x[-maxlend:]
        n = maxlend
    return [empty]*(maxlend-n) + x + [eos]


# In[32]:


samples = [lpadd([3]*26)]
# pad from right (post) so the first maxlend will be description followed by headline
data = sequence.pad_sequences(samples, maxlen=maxlen, value=empty, padding='post', truncating='post')


# In[33]:


np.all(data[:,maxlend] == eos)


# In[34]:


np.all(data[:,maxlend] == eos)


# In[35]:


probs = model.predict(data, verbose=0, batch_size=1)
probs.shape


# Sample generation

# In[36]:


# variation to https://github.com/ryankiros/skip-thoughts/blob/master/decoding/search.py
def beamsearch(predict, start=[empty]*maxlend + [eos], avoid=None, avoid_score=1,
               k=1, maxsample=maxlen, use_unk=True, oov=vocab_size-1, empty=empty, eos=eos, temperature=1.0):
    """return k samples (beams) and their NLL scores, each sample is a sequence of labels,
    all samples starts with an `empty` label and end with `eos` or truncated to length of `maxsample`.
    You need to supply `predict` which returns the label probability of each sample.
    `use_unk` allow usage of `oov` (out-of-vocabulary) label in samples
    """
    def sample(energy, n, temperature=temperature):
        """sample at most n different elements according to their energy"""
        n = min(n,len(energy))
        prb = np.exp(-np.array(energy) / temperature )
        res = []
        for i in xrange(n):
            z = np.sum(prb)
            r = np.argmax(np.random.multinomial(1, prb/z, 1))
            res.append(r)
            prb[r] = 0. # make sure we select each element only once
        return res

    dead_samples = []
    dead_scores = []
    live_samples = [list(start)]
    live_scores = [0]

    while live_samples:
        # for every possible live sample calc prob for every possible label 
        probs = predict(live_samples, empty=empty)
        assert vocab_size == probs.shape[1]

        # total score for every sample is sum of -log of word prb
        cand_scores = np.array(live_scores)[:,None] - np.log(probs)
        cand_scores[:,empty] = 1e20
        if not use_unk and oov is not None:
            cand_scores[:,oov] = 1e20
        if avoid:
            for a in avoid:
                for i, s in enumerate(live_samples):
                    n = len(s) - len(start)
                    if n < len(a):
                        # at this point live_sample is before the new word,
                        # which should be avoided, is added
                        cand_scores[i,a[n]] += avoid_score
        live_scores = list(cand_scores.flatten())
        

        # find the best (lowest) scores we have from all possible dead samples and
        # all live samples and all possible new words added
        scores = dead_scores + live_scores
        ranks = sample(scores, k)
        n = len(dead_scores)
        dead_scores = [dead_scores[r] for r in ranks if r < n]
        dead_samples = [dead_samples[r] for r in ranks if r < n]
        
        live_scores = [live_scores[r-n] for r in ranks if r >= n]
        live_samples = [live_samples[(r-n)//vocab_size]+[(r-n)%vocab_size] for r in ranks if r >= n]

        # live samples that should be dead are...
        # even if len(live_samples) == maxsample we dont want it dead because we want one
        # last prediction out of it to reach a headline of maxlenh
        def is_zombie(s):
            return s[-1] == eos or len(s) > maxsample
        
        # add zombies to the dead
        dead_scores += [c for s, c in zip(live_samples, live_scores) if is_zombie(s)]
        dead_samples += [s for s in live_samples if is_zombie(s)]
        
        # remove zombies from the living 
        live_scores = [c for s, c in zip(live_samples, live_scores) if not is_zombie(s)]
        live_samples = [s for s in live_samples if not is_zombie(s)]

    return dead_samples, dead_scores


# In[37]:


def keras_rnn_predict(samples, empty=empty, model=model, maxlen=maxlen):
    """for every sample, calculate probability for every possible label
    you need to supply your RNN model and maxlen - the length of sequences it can handle
    """
    sample_lengths = map(len, samples)
    assert all(l > maxlend for l in sample_lengths)
    assert all(l[maxlend] == eos for l in samples)
    # pad from right (post) so the first maxlend will be description followed by headline
    data = sequence.pad_sequences(samples, maxlen=maxlen, value=empty, padding='post', truncating='post')
    probs = model.predict(data, verbose=0, batch_size=batch_size)
    return np.array([output2probs(prob[sample_length-maxlend-1]) for prob, sample_length in zip(probs, sample_lengths)])


# In[38]:


def vocab_fold(xs):
    """convert list of word indexes that may contain words outside vocab_size to words inside.
    If a word is outside, try first to use glove_idx2idx to find a similar word inside.
    If none exist then replace all accurancies of the same unknown word with <0>, <1>, ...
    """
    xs = [x if x < vocab_size-nb_unknown_words else glove_idx2idx.get(x,x) for x in xs]
    # the more popular word is <0> and so on
    outside = sorted([x for x in xs if x >= vocab_size-nb_unknown_words])
    # if there are more than nb_unknown_words oov words then put them all in nb_unknown_words-1
    outside = dict((x,vocab_size-1-min(i, nb_unknown_words-1)) for i, x in enumerate(outside))
    xs = [outside.get(x,x) for x in xs]
    return xs


# In[39]:


def vocab_unfold(desc,xs):
    # assume desc is the unfolded version of the start of xs
    unfold = {}
    for i, unfold_idx in enumerate(desc):
        fold_idx = xs[i]
        if fold_idx >= vocab_size-nb_unknown_words:
            unfold[fold_idx] = unfold_idx
    return [unfold.get(x,x) for x in xs]


# In[57]:


def gensamples(X=None, X_test=None, Y_test=None, avoid=None, avoid_score=1, skips=2, k=10, batch_size=batch_size, short=True, temperature=1., use_unk=True):
    i = 0
    if X is None or isinstance(X,int):
        if X is None:
            i = random.randint(0,len(X_test)-1)
        else:
            i = X
        print 'HEAD %d:'%i,' '.join(idx2word[w] for w in Y_test[i])
        print 'DESC:',' '.join(idx2word[w] for w in X_test[i])
        sys.stdout.flush()
        x = X_test[i]
    else:
        x = [word2idx[w.rstrip('^')] for w in X.split()]
        
    if avoid:
        # avoid is a list of avoids. Each avoid is a string or list of word indeicies
        if isinstance(avoid,str) or isinstance(avoid[0], int):
            avoid = [avoid]
        avoid = [a.split() if isinstance(a,str) else a for a in avoid]
        avoid = [vocab_fold([w if isinstance(w,int) else word2idx[w] for w in a])
                 for a in avoid]

    #print 'HEADS:'
    samples = []
    if maxlend == 0:
        skips = [0]
    else:
        skips = range(min(maxlend,len(x)), max(maxlend,len(x)), abs(maxlend - len(x)) // skips + 1)
    for s in skips:
        start = lpadd(x[:s])
        fold_start = vocab_fold(start)
        sample, score = beamsearch(predict=keras_rnn_predict, start=fold_start, avoid=avoid, avoid_score=avoid_score,
                                   k=k, temperature=temperature, use_unk=use_unk)
        assert all(s[maxlend] == eos for s in sample)
        samples += [(s,start,scr) for s,scr in zip(sample,score)]

    samples.sort(key=lambda x: x[-1])
    codes = []
    
    predictions = []
    for sample, start, score in samples:
        code = ''
        words = []
        sample = vocab_unfold(start, sample)[len(start):]
        for w in sample:
            if w == eos:
                break
            words.append(idx2word[w])
            code += chr(w//(256*256)) + chr((w//256)%256) + chr(w%256)
        if short:
            distance = min([100] + [-Levenshtein.jaro(code,c) for c in codes])
            if distance > -0.6:
                predictions.append((score, words))
                #print score, ' '.join(words)
        #         print '%s (%.2f) %f'%(' '.join(words), score, distance)
        else:
            predictions.append((score, words))
                #print score, ' '.join(words)
        codes.append(code)
        
    
    if len(predictions) == 0:
        return predictions, 0, []

    for score, words in predictions:
        print score, ' '.join(words)
    score, words = predictions[0]
    return predictions, score, words


# In[41]:


seed = 8
random.seed(seed)
np.random.seed(seed)


# Weights

# In[42]:


def wsimple_context(X, mask, n=activation_rnn_size, maxlend=maxlend, maxlenh=maxlenh):
    desc, head = X[:,:maxlend], X[:,maxlend:]
    head_activations, head_words = head[:,:,:n], head[:,:,n:]
    desc_activations, desc_words = desc[:,:,:n], desc[:,:,n:]
    
    # RTFM http://deeplearning.net/software/theano/library/tensor/basic.html#theano.tensor.batched_tensordot
    # activation for every head word and every desc word
    #activation_energies = K.batch_dot(head_activations, desc_activations, axes=([2],[2]))
    activation_energies = K.batch_dot(head_activations, desc_activations, axes=(2,2))
    # make sure we dont use description words that are masked out
    #assert mask.ndim == 2
    assert mask.get_shape().ndims == 2
    fixed = K.zeros_like(activation_energies) + (-1e20)
    #activation_energies = K.switch(mask[:, None, :maxlend], activation_energies, -1e20)
    activation_energies = K.switch(mask[:, None, :maxlend], activation_energies, fixed)
    
    # for every head word compute weights for every desc word
    activation_energies = K.reshape(activation_energies,(-1,maxlend))
    activation_weights = K.softmax(activation_energies)
    activation_weights = K.reshape(activation_weights,(-1,maxlenh,maxlend))

    return activation_weights


class WSimpleContext(Lambda):
    def __init__(self):
        super(WSimpleContext, self).__init__(wsimple_context)
        self.supports_masking = True

    def compute_mask(self, input, input_mask=None):
        return input_mask[:, maxlend:]
    
    def get_output_shape_for(self, input_shape):
        nb_samples = input_shape[0]
        n = 2*(rnn_size - activation_rnn_size)
        return (nb_samples, maxlenh, n)


# In[43]:


wmodel = Sequential()
wmodel.add(rnn_model)


# In[44]:


wmodel.add(Lambda(wsimple_context,
                 mask = lambda inputs, mask: mask[:,maxlend:],
                 output_shape = lambda input_shape: (input_shape[0], maxlenh, maxlend)))


# In[45]:


wmodel.compile(loss='categorical_crossentropy', optimizer=optimizer)


# In[46]:


# summary here?? ***
wmodel.summary()


# ### Perform testing

# In[47]:


seed = 8
random.seed(seed)
np.random.seed(seed)


# Cycle through examples, may have to remove words not in vocabulary?

# In[58]:


bleu_scores = []
levenshtein_scores = []

max_bleu = 0.
min_lev = float('inf')

for i in range(len(X_using)):
    
    # check for shorter than samples
    if MODE == 0: 
        X_example = ' '.join([idx2word[w] for w in X_using[i]][:50])
    elif MODE == 1:
        X_example = ' '.join([idx2word[w] for w in X_using[i]][:25]) + ' ' + ' '.join([idx2word[w] for w in X_using[i]][-25:])
    elif MODE == 2:
        X_example = ' '.join([idx2word[w] for w in X_using[i]][:50]) + ' ' + ' '.join([idx2word[w] for w in X_using[i]][-25:])
    elif MODE == 3:
        X_example = ' '.join([idx2word[w] for w in X_using[i]][:50]) + ' ' + ' '.join([idx2word[w] for w in X_using[i]][-50:])
    else:
        pass
    
    Y_example = ' '.join([idx2word[w] for w in Y_using[i]])
    
    # call gensample, modified to return predicted headline
#     samples, lev_score, words = gensamples(X_example + " ~", skips=2, batch_size=batch_size, k=10, temperature=1.)
    samples, lev_score, words = gensamples(X_example, skips=2, batch_size=batch_size, k=10, temperature=1.)
    if len(samples) == 0: lev_score = len(Y_example)

    
    # compute bleu score, append
    bleu_score = sentence_bleu([[y.lower() for y in Y_example.split()]], [w.lower() for w in words])
    bleu_scores.append(bleu_score)
    # append levenshtein
    levenshtein_scores.append(lev_score)
    
    # add heat maps, etc. if you want
    
    print i, bleu_score, lev_score, "averages: ", np.mean(bleu_scores), np.mean(levenshtein_scores)
    if bleu_score > max_bleu:
        print "MAX BLEU"
        max_bleu = bleu_score
    if lev_score < min_lev:
        print "MIN LEV"
        min_lev = lev_score
    print "Abs: ", X_example
    print "Title: ", Y_example
    print "Prediction: ", ' '.join(words),"\n"
    
avg_bleu = np.mean(bleu_scores)
avg_lev = np.mean(levenshtein_scores)
print "average BLEU:", avg_bleu
print "average Levenshtein:", avg_lev


# In[ ]:




