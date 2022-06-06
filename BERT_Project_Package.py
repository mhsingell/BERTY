#!/usr/bin/env python
# coding: utf-8

# In[1]:


#General
import numpy as np
import pandas as pd

#For all your pickle needs
import pickle

# For loading datasets
#from datasets import list_datasets, load_dataset
#import datasets

#For splitting tuples
import ast

#For Train and Test Split
import random

#For Running BERT
from transformers import BertTokenizer, BertForPreTraining, BertModel, TFBertForMaskedLM, BertConfig, BertForMaskedLM
import torch

#For pre-training not by hand for BERT
from transformers import TrainingArguments
from transformers import Trainer

#Initialize optimizer for pre-training
from transformers import AdamW

#For progress bar during training
from tqdm import tqdm

#For predicting masked words
import tensorflow as tf

#For splitting tuples
import ast

#For score prediction
import torch.nn.functional as F

# In[10]:


def trainvaltest(x):
    random.seed(94305)
    #First need to make a train and test split
    train_index = random.sample(list(range(len(x["text"]))), round(0.95*len(x["text"])))
    val_index = random.sample(list(set(range(len(x["text"])))-set(train_index)), round(0.04*len(x["text"])))
    test_index = list(set(range(len(x["text"])))-set(train_index)-set(val_index))

    train = {"id": [], "url": [], "title": [], "text": [], "pagevs": []}

    for indx in range(len(train_index)):
        myidx = train_index[indx]
        for key in x.keys():
            my_key = x[key]
            train[key].append(x[key][myidx])
            
    validation = {"id": [], "url": [], "title": [], "text": [], "pagevs": []}

    for indx in range(len(val_index)):
        myidx = train_index[indx]
        for key in x.keys():
            my_key = x[key]
            validation[key].append(x[key][myidx])

    test = {"id": [], "url": [], "title": [], "text": [], "pagevs": []}

    for indx in range(len(test_index)):
        myidx = test_index[indx]
        for key in x.keys():
            my_key = x[key]
            test[key].append(x[key][myidx])
    
    return train, validation, test


# In[3]:


tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
#model = BertForMaskedLM.from_pretrained('bert-base-cased')


# In[4]:


#Define function to predict most likely word from masked sentence
def get_top_k_predictions(input_string, modeler, k=1, tokenizer=tokenizer) -> str:

    tokenized_inputs = tokenizer(input_string, return_tensors = "pt", max_length = 512, truncation = True, padding = 'max_length')
    #Now we create our masked data
    tokenized_inputs['labels'] = tokenized_inputs.input_ids.detach().clone()

    #Masking Words
    #Then we mask a word, but not the classifier token (101), the end token (102) or the padding tokens (the 0s).
    mask_arr_p1 = (tokenized_inputs.input_ids != 101) * (tokenized_inputs.input_ids != 0) * (tokenized_inputs.input_ids != 102)
        #Set the id of our tokens we've randomly decided to mask to our mask token, 103
    rand = torch.rand(tokenized_inputs.input_ids.shape)
    my_potents = []
    my_list = list(mask_arr_p1[0])
    for itemz in range(len(my_list)):
        my_item = my_list[itemz]
        if my_item == True:
            my_potents.append(itemz)
    my_choice = random.choice(my_potents)
    mask_arr = (rand == rand[0][my_choice]) * (tokenized_inputs.input_ids != 101) * (tokenized_inputs.input_ids != 0) * (tokenized_inputs.input_ids != 102)
    for i in range(tokenized_inputs.input_ids.shape[0]):
        selection = torch.flatten(mask_arr[i].nonzero()).tolist()
        tokenized_inputs.input_ids[i, selection] = 103
    
    
    outputs = modeler(tokenized_inputs["input_ids"])
    #turn output predicted logits into tf
    if str(type(modeler)) == "<class 'transformers.models.bert.modeling_bert.BertForMaskedLM'>":
        my_logits = tf.convert_to_tensor(outputs.logits.detach().numpy())
        for_score = outputs.logits
    else:
        my_logits = tf.convert_to_tensor(outputs.prediction_logits.detach().numpy())
        for_score = outputs.prediction_logits

    #Find the top predicted token
    top_k_indices = tf.math.top_k(my_logits, k).indices[0].numpy()
    decoded_output = tokenizer.batch_decode(top_k_indices)
    mask_token = tokenizer.encode(tokenizer.mask_token)[1:-1]
    mask_index = np.where(tokenized_inputs["input_ids"][0].numpy()==mask_token)[0][0]

    #Output both predicted and actual token
    score_pred = F.softmax(for_score, dim=-1)[0][0][mask_index].detach().numpy()
    decoded_output_words = decoded_output[mask_index]
    actual_word = str(tokenizer.batch_decode(tokenized_inputs['labels'].tolist()[0])[mask_index]).replace(" ", "")

    return decoded_output_words, actual_word, score_pred


# In[ ]:


def find_accuracy(my_data, model_used):
    #First, use get top k predictions to find actual and predicted masked token
    my_guess = []
    my_actual = []
    full_stack = my_data["text"]
    mr_right = []
    my_prob_score = []
    for it in range(len(full_stack)):
        my_text = full_stack[it]
        my_guess_int, my_actual_int, my_score = get_top_k_predictions(my_text, k = 1, tokenizer=tokenizer, modeler = model_used)
        my_guess.append(my_guess_int)
        my_actual.append(my_actual_int)
        my_prob_score.append(my_score)
        if my_actual_int == my_guess_int:
            mr_right.append(1)
        else:
            mr_right.append(0)

    my_predictions_mlm = pd.DataFrame({"prediction" : my_guess, "actual" : my_actual, "correct" : mr_right, "prob_score": my_prob_score})
    
    #Store straight accuracy as number of correct guesses over total guesses
    my_accuracy = sum(my_predictions_mlm["correct"])/len(my_predictions_mlm["correct"])
    
    #Calculate weighted accuracy through below process
    #Find accuracy weighted by pageview
    my_pvs = my_data["pagevs"]
    total_pageviews_test = sum(my_pvs)
    #For weighting there were 88,230,936,650 total wikipedia views in our range of 3/1/2021 to 3/1/2022. 
    #Resource here: https://pageviews.wmcloud.org/siteviews/?platform=all-access&source=pageviews&agent=user&start=2021-03-01&end=2022-03-01&sites=en.wikipedia.org
    #Our pages, which were created between 10/11/2018 and 3/1/2019 represent a small fraction of those

    #First we calculate the weight for each individual prediction based on Pageview of article / Pageview of All Wikipedia
    weighted_pvs = [x / 88230936650 for x in my_pvs]
    my_predictions_mlm["PV_weight"] = weighted_pvs

    #Next multiply the PV weight by whether the prediction was correct (1) or not (0)
    my_predictions_mlm["PV_correct"] = my_predictions_mlm["PV_weight"] * my_predictions_mlm["correct"]

    #Now we sum this column and scale it up by all wikipedia / the number of total pageviews in our test set 
    my_weighted_accuracy = sum(my_predictions_mlm["PV_correct"]) * (88230936650 / total_pageviews_test)
    
    return my_accuracy, my_weighted_accuracy, my_predictions_mlm


# In[ ]:


def find_accuracy_notweighted(my_data, model_used):
    #First, use get top k predictions to find actual and predicted masked token
    my_tuples = []
    full_stack = my_data["text"]
    for it in range(len(full_stack)):
        my_text = full_stack[it]
        my_tuples.append(get_top_k_predictions(my_text, k = 1, tokenizer=tokenizer, modeler = model_used))
    my_preds = []
    my_actual = []
    mr_right = []
    #Next, figure out if predicted token = actual token
    for pairs in range(len(my_tuples)):
        my_preds.append(my_tuples[pairs][0])
        my_actual.append(my_tuples[pairs][1])
        if my_preds[pairs] == my_actual[pairs]:
            mr_right.append(1)
        else:
            mr_right.append(0)
    
    my_predictions_mlm = pd.DataFrame({"prediction" : my_preds, "actual" : my_actual, "correct" : mr_right})
    
    #Store straight accuracy as number of correct guesses over total guesses
    my_accuracy = sum(my_predictions_mlm["correct"])/len(my_predictions_mlm["correct"])
    
    return my_accuracy


# In[ ]:


#First, set up data to run the nsp and mlm tasks of pre-training
def nsp_mlm(data):
    #tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    bag = [sentence for para in data['text'] for sentence in para.split('.') if sentence != '']
    bag_size = len(bag)
    #Initialize list for pairing of sentences (a and b) and label
    sentence_a = []
    sentence_b = []
    label = [] # 0 means is the next sentence, 1 means not the next sentence
    #make the pairings such that half the sentences are following each other and half are not

    #Loop through each paragraph in the text
    for paragraph in data['text']: 
        #split each paragraph into sentences
        sentences = [
            sentence for sentence in paragraph.split('.') if sentence != ''
        ]
        #check if the paragraph has more than one sentence
        num_sentences = len(sentences)
        if num_sentences > 1:
            #if it does, select pairing of sentences where sentence a is directly before b 50% of the time
            start = random.randint(0, num_sentences -2)
            sentence_a.append(sentences[start])
            if random.random() > 0.5:
                #50% of the time, be random
                sentence_b.append(bag[random.randint(0,bag_size-1)])
                label.append(1)
            else:
                #50% of the time, put following sentence
                sentence_b.append(sentences[start+1])
                label.append(0)
    inputs = tokenizer(sentence_a, sentence_b, return_tensors = 'pt', max_length = 512, truncation = True, padding = 'max_length')
    inputs['next_sentence_label'] = torch.LongTensor([label]).T
    inputs['labels'] = inputs.input_ids.detach().clone()

    #Masking Words
    #Then we mask some words (15%) and but not the classifier token (101), the end token (102) or the padding tokens (the 0s).
    rand = torch.rand(inputs.input_ids.shape)
    #rand.shape #number of sequences, number of token, this is a random value between 0 and 1
    #Now mask anything under 0.15, that will be about 15%
    #But add logic to not mask the classifier or padding tokens
    mask_arr = (rand <0.15) * (inputs.input_ids != 101) * (inputs.input_ids != 0) * (inputs.input_ids != 102)
    #Set the id of our tokens we've randomly decided to mask to our mask token, 103
    for i in range(inputs.input_ids.shape[0]):
        selection = torch.flatten(mask_arr[i].nonzero()).tolist()
        inputs.input_ids[i, selection] = 103
    return inputs


# In[ ]:


#convert these values into a pytorch data object
#this formats our data into a dataset object
class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings.input_ids)

def attach_pvs(pvs, my_dict):
    #Organize Pvs
    my_titles = []
    my_pageview = []
    my_view_month = []
    title_yrview = {}
    for items in range(len(pvs)): 
        my_titl = ast.literal_eval(pvs[items])[0]
        my_titles.append(my_titl)
        #my_pageview.append(ast.literal_eval(lines[items])[1])
        my_view_month_list = ast.literal_eval(pvs[items])[2:][0][0:12]
        my_view_month.append(sum(my_view_month_list))
        title_yrview[my_titl] = sum(my_view_month_list)
    #Attach pageviews to data
    my_dict["pagevs"] = [0] * len(my_dict["title"])
    test_titles = my_dict["title"]
    for tils in range(len(test_titles)):
        my_til = test_titles[tils]
        if my_til not in title_yrview.keys():
            my_dict["pagevs"][tils] = 0
        else:
            my_dict["pagevs"][tils] = title_yrview[my_til]
    return my_dict