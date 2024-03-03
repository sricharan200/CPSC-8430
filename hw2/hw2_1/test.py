#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
import torch
import json
from model import test_data, test, MODELS, Decoder, Encoder, attention 
from torch.utils.data import DataLoader
from bleu_eval import BLEU
import pickle

model = torch.load('model.h5', map_location=lambda storage, loc: storage)
data = test_data('{}'.format(sys.argv[1]))
test_load = DataLoader(data, batch_size=32, shuffle=True)

with open('i2w.pickle', 'rb') as handle:
    i2w = pickle.load(handle)

# model = model.cuda()
res = test(test_load, model, i2w)

with open(sys.argv[2], 'w') as file:
    for id, s in res:
        file.write('{},{}\n'.format(id, s))


test = json.load(open("testing_label.json"))
output = sys.argv[2]
result = {}
with open(output,'r') as file:
    for line in file:
        line = line.rstrip()
        c = line.index(',')
        test_id = line[:c]
        cap = line[c+1:]
        result[test_id] = cap

bleu=[]
for item in test:
    score = []
    captions = [x.rstrip('.') for x in item['caption']]
    score.append(BLEU(result[item['id']],captions,True))
    bleu.append(score[0])
average = sum(bleu) / len(bleu)
print("Average bleu score is " + str(average))

