#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import random
import numpy as np
import pandas as pd
import nltk


# In[2]:


# Data Set


# In[3]:


with open("en_US.twitter.txt", "r") as f:
    data = f.read()
print("Number of letters:", len(data))

print("First 300 letters of the data")
print("-------")
display(data[0:300])
print("-------")

print("Last 300 letters of the data")
print("-------")
display(data[-300:])
print("-------")


# In[4]:


def tokenize_the_data(data):
    
    sentences = data.split('\n')
    sentences = [s.strip() for s in sentences]
    sentences = [s for s in sentences if len(s) > 0]
        
    tokenized_sentences = []
    
    for sentence in sentences:
        sentence = sentence.lower()
        tokenized = nltk.word_tokenize(sentence)
        tokenized_sentences.append(tokenized)
    
    return tokenized_sentences


# In[5]:


# Spliting the data into Training & Validation & Testing Set


# In[6]:


tokenized_data = tokenize_the_data(data)

random.seed(87)
random.shuffle(tokenized_data)

train_size = int(len(tokenized_data) * 0.8)
train_data = tokenized_data[0:train_size]
test_data = tokenized_data[train_size:]

print("{} data are split into {} train and {} test set".format(len(tokenized_data), len(train_data), len(test_data)))

print("First training sample:")
print(train_data[0])
      
print("First test sample")
print(test_data[0])


# In[7]:


# Training 


# In[8]:


# Word Frequency


# In[9]:


def vocab(data, count_threshold):
        
    word_counts = {}    
                                                                # Word Count
    for sentence in tokenized_sentences: 
        for token in sentence:
            if token not in word_counts:
                word_counts[token] = 1
            else:
                word_counts[token] += 1
       
    vocab = []                                                 # Vocab - word with high frequency
    
    for word, cnt in word_counts.items(): 
        if cnt >= count_threshold:
            vocab.append(word)
    
    return vocab


# In[10]:


def out_of_vocab(data, vocab):
    
    replaced_tokenized_sentences = []                          # replace Out Of Vocabulary words by "unk"
    vocab = set(vocab)

    for sentence in tokenized_sentences:
        replaced_sentence = []
        for token in sentence:
            if token in vocabulary:
                replaced_sentence.append(token)
            else:
                replaced_sentence.append("<unk>")
                
        replaced_tokenized_sentences.append(replaced_sentence)
        

    return word_counts, vocab, replaced_tokenized_sentence


# In[11]:


# UNQ_C7 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
### GRADED_FUNCTION: preprocess_data ###
def preprocess_data(train_data, test_data, count_threshold):
    
    vocabulary = vocab(train_data, count_threshold)
    
    # For the train data, replace less common words with "<unk>"
    train_data_replaced = out_of_vocab(train_data, vocabulary)
    
    # For the test data, replace less common words with "<unk>"
    test_data_replaced = out_of_vocab(test_data, vocabulary)
    
    return train_data_replaced, test_data_replaced, vocabulary


# In[12]:


# Language Model


# In[13]:


# N-Grams


# In[14]:


def count_n_grams(data, n, start_token='<s>', end_token = '<e>'):
    
    n_grams = {}
    
    for sentence in data:
        sentence = [start_token] * (n-1) + sentence + [end_token]       # adding (n-1) Start and 1 End token
        sentence = tuple(sentence)
        
        m = len(sentence) if n==1 else len(sentence) - 1
        
        for i in range(m):
            n_gram = sentence[i:i+n]
            if n_gram in n_grams:           
                n_grams[n_gram] += 1
            else:
                n_grams[n_gram] = 1
    
    return n_grams


# In[15]:


sentences = [['i', 'like', 'a', 'cat'], ['this', 'dog', 'is', 'like', 'a', 'cat']]
print("Uni-gram:")
print(count_n_grams(sentences, 1))
print("Bi-gram:")
print(count_n_grams(sentences, 2))


# In[16]:


# Sequency Probability Predition
def estimate_probability(word, previous_tokens, n_minus_1_gram_counts, n_gram_counts, vocabulary_size, k=1.0):
    
    n_gram = tuple(previous_tokens) + (word,)
    n_gram_counts = n_gram_counts[n_gram] if n_gram in n_gram_counts  else 0
    
    numerator = n_gram_counts + k
    #print(numerator)
    
    previous_n_gram = tuple(previous_tokens) 
    previous_n_gram_count = n_minus_1_gram_counts[previous_n_gram] if previous_n_gram in n_minus_1_gram_counts  else 0
    
    denominator = previous_n_gram_count + k * vocabulary_size
    #print(denominator)
    
    probability = numerator/denominator
    
    return probability


# In[17]:


# Estimate of probability of word

sentences = [['i', 'like', 'a', 'cat'], ['this', 'dog', 'is', 'like', 'a', 'cat']]

unique_words = list(set(sentences[0] + sentences[1]))

unigram_counts = count_n_grams(sentences, 1)
bigram_counts = count_n_grams(sentences, 2)
tmp_prob = estimate_probability("cat", ["this","is","a"], unigram_counts, bigram_counts, len(unique_words), k=1)

print(f"The estimated probability of word 'cat' given the previous tokens 'this is a' is: {tmp_prob:.4f}")
print(f" P(cat/this is a) = C(this is a cat)/ C(this is a) is: {tmp_prob:.4f}")


# In[18]:


def estimate_probabilities(previous_token, n_minus_1_gram_counts, n_gram_counts, vocabulary, k=1.0):
    
    previous_n_gram = tuple(previous_token)
    
    vocabulary = vocabulary + ["<e>", "<unk>"]
    vocabulary_size = len(vocabulary)
    
    probabilities = {}
    for word in vocabulary:
        probability = estimate_probability(word, previous_token, n_minus_1_gram_counts, n_gram_counts, vocabulary_size, k=k)
        probabilities[word] = probability

    return probabilities


# In[19]:


sentences = [['i', 'like', 'a', 'cat'], ['this', 'dog', 'is', 'like', 'a', 'cat']]

unique_words = list(set(sentences[0] + sentences[1]))

unigram_counts = count_n_grams(sentences, 1)
bigram_counts = count_n_grams(sentences, 2)
estimate_probabilities("a", unigram_counts, bigram_counts, unique_words, k=1)


# In[20]:


# Probability Matrix


# In[21]:


def make_count_matrix(n_plus1_gram_counts, vocabulary):
    
    vocabulary = vocabulary + ["<e>", "<unk>"]
    
    n_grams = []
    for n_plus1_gram in n_plus1_gram_counts.keys():
        n_gram = n_plus1_gram[0:-1]
        n_grams.append(n_gram)
    n_grams = list(set(n_grams))
    
    row_index = {n_gram:i for i, n_gram in enumerate(n_grams)}

    col_index = {word:j for j, word in enumerate(vocabulary)}
    
    nrow = len(n_grams)
    ncol = len(vocabulary)
    count_matrix = np.zeros((nrow, ncol))
    for n_plus1_gram, count in n_plus1_gram_counts.items():
        n_gram = n_plus1_gram[0:-1]
        word = n_plus1_gram[-1]
        if word not in vocabulary:
            continue
        i = row_index[n_gram]
        j = col_index[word]
        count_matrix[i, j] = count
    
    count_matrix = pd.DataFrame(count_matrix, index=n_grams, columns=vocabulary)
    return count_matrix


# In[22]:


sentences = [['i', 'like', 'a', 'cat'],
                 ['this', 'dog', 'is', 'like', 'a', 'cat']]
unique_words = list(set(sentences[0] + sentences[1]))

bigram_counts = count_n_grams(sentences, 2)
trigram_counts = count_n_grams(sentences, 3)

print('bigram counts')
display(make_count_matrix(bigram_counts, unique_words))

print('trigram counts')
display(make_count_matrix(trigram_counts, unique_words))


# In[23]:


def make_probability_matrix(n_plus1_gram_counts, vocabulary, k):
    count_matrix = make_count_matrix(n_plus1_gram_counts, unique_words)
    count_matrix += k
    prob_matrix = count_matrix.div(count_matrix.sum(axis=1), axis=0)
    return prob_matrix


# In[24]:


sentences = [['i', 'like', 'a', 'cat'], ['this', 'dog', 'is', 'like', 'a', 'cat']]
unique_words = list(set(sentences[0] + sentences[1]))

bigram_counts = count_n_grams(sentences, 2)
trigram_counts = count_n_grams(sentences, 3)

print("bigram probabilities")
display(make_probability_matrix(bigram_counts, unique_words, k=0.1))

print("trigram probabilities")
display(make_probability_matrix(trigram_counts, unique_words, k=0.1))


# In[ ]:


# Suggest the high probable word


# In[32]:


def suggest_a_word(previous_tokens, n_gram_counts, n_plus1_gram_counts, vocabulary, k=1.0):
    
    n = len(list(n_gram_counts.keys())[0]) 
    
    previous_n_gram = previous_tokens[-n:]
    probabilities = estimate_probabilities(previous_n_gram, n_gram_counts, n_plus1_gram_counts,vocabulary, k=k)
    suggestion = None
    max_prob = 0
    
    
    for word, prob in probabilities.items():
        if max_prob < prob:
            suggestion = word
            max_prob = prob

    return suggestion, max_prob


# In[33]:


# test your code
sentences = [['i', 'like', 'a', 'cat'],['this', 'dog', 'is', 'like', 'a', 'cat']]

unique_words = list(set(sentences[0] + sentences[1]))

unigram_counts = count_n_grams(sentences, 1)
bigram_counts = count_n_grams(sentences, 2)

previous_tokens = ["i", "like"]
tmp_suggest1 = suggest_a_word(previous_tokens, unigram_counts, bigram_counts, unique_words, k=1.0)
print(f"The previous words are 'i like',\n\tand the suggested word is `{tmp_suggest1[0]}` with a probability of {tmp_suggest1[1]:.4f}")


# In[25]:


# Perplexity Measurement


# In[26]:


def calculate_perplexity(sentence, n_minus_1_gram_counts, n_gram_counts, vocabulary_size, k=1.0):
    
    n = len(list(n_minus_1_gram_counts.keys())[0]) 
    sentence = ["<s>"] * n + sentence + ["<e>"]
    
    sentence = tuple(sentence)
    N = len(sentence)
    
    product_pi = 1.0
    
    for t in range(n, N):
        n_gram = sentence[t-n:t]
        word = sentence[t]
        probability = estimate_probability(word, n_gram, n_minus_1_gram_counts, n_gram_counts, vocabulary_size, k=k)
        product_pi *= 1/probability

    perplexity = product_pi**(1/N)
    
    return perplexity


# In[27]:


sentences = [['i', 'like', 'a', 'cat'], ['this', 'dog', 'is', 'like', 'a', 'cat']]
unique_words = list(set(sentences[0] + sentences[1]))

unigram_counts = count_n_grams(sentences, 1)
bigram_counts = count_n_grams(sentences, 2)


perplexity_train1 = calculate_perplexity(sentences[0],unigram_counts, bigram_counts,len(unique_words), k=0.1)
print(f"Perplexity for first train sample: {perplexity_train1:.4f}")

test_sentence = ['i', 'like', 'a', 'dog']
perplexity_test = calculate_perplexity(test_sentence,unigram_counts, bigram_counts,len(unique_words), k=0.1)
print(f"Perplexity for test sample: {perplexity_test:.4f}")


# In[28]:


Using multiple N-Grams for suggestion 


# In[36]:


def get_suggestions(previous_tokens, n_gram_counts_list, vocabulary, k=1.0, start_with=None):
    model_counts = len(n_gram_counts_list)
    suggestions = []
    
    for i in range(model_counts-1):
        n_minus_1_gram_counts = n_gram_counts_list[i]
        n_gram_counts = n_gram_counts_list[i+1]
        
        suggestion = suggest_a_word(previous_tokens, n_minus_1_gram_counts, n_gram_counts, vocabulary,k=k)
        suggestions.append(suggestion)
    return suggestions


# In[37]:


# test your code
sentences = [['i', 'like', 'a', 'cat'],
             ['this', 'dog', 'is', 'like', 'a', 'cat']]
unique_words = list(set(sentences[0] + sentences[1]))

unigram_counts = count_n_grams(sentences, 1)
bigram_counts = count_n_grams(sentences, 2)
trigram_counts = count_n_grams(sentences, 3)
quadgram_counts = count_n_grams(sentences, 4)
qintgram_counts = count_n_grams(sentences, 5)

n_gram_counts_list = [unigram_counts, bigram_counts, trigram_counts, quadgram_counts, qintgram_counts]
previous_tokens = ["i", "like"]
tmp_suggest3 = get_suggestions(previous_tokens, n_gram_counts_list, unique_words, k=1.0)

print(f"The previous words are 'i like', the suggestions are:")
display(tmp_suggest3)


# In[ ]:


# We developed Language Model to predict next probable word giveb previous words using N-Grams.

