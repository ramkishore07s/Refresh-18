#!/usr/bin/env python
# coding: utf-8

# In[72]:

import os
import shutil
import nltk
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_folder', type=str)
parser.add_argument('--output_folder', type=str, default='./summaries/')
parser.add_argument('--summary_len', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--weights', type=str, default='./cnn.params')
parser.add_argument('--glove', type=str, default='../data/glove.pkl')
parser.add_argument('--gpu_no', type=int, default=0)
args = parser.parse_args()

# In[33]:

get_ipython().magic('run dataset.ipynb')
get_ipython().magic('run model.ipynb')

cpu = False
if args.gpu_no == -1:
    cpu = True


if not cpu: torch.cuda.set_device(args.gpu_no)

# In[79]:


batch_size = 20
input_folder = args.input_folder
output_folder = args.output_folder
summary_len = args.summary_len
filenames = [f for f in os.listdir(input_folder) if ".txt" in f]


# In[17]:


g = GloveEmbeddings()
g.load_dump(args.glove)

d = DataHandler()


# In[58]:


m = EncoderDecoder(torch.cuda.FloatTensor(g.vectors), word_emb_size=g.dim, 
                       sen_emb_size=350, doc_emb_size=600, sen_len=50, batch_size=20, output_dim=2,
                       reverse=True)

m.load_state_dict(torch.load(args.weights))
#m.cuda()


# In[68]:


for i in range(0, len(filenames), batch_size):
    summaries = []

    for filename in filenames[i:i+batch_size]:
        d.data = []

        text = open(os.path.join(input_folder, filename)).read()
        sentences = nltk.tokenize.sent_tokenize(text)
        words = [nltk.tokenize.word_tokenize(sentence) for sentence in sentences]
        d.data.append(g.convert_to_indices(words))
        
        d.pad()

        with torch.no_grad():
            if not cpu: pred = m(torch.cuda.LongTensor(d.lines))
            else: pred = m(torch.LongTensor(d.lines))
            for i, lines in enumerate(pred.data):
                pos = [l[1].data for l in lines]
                selected_lines = list(zip(*sorted(zip(pos, range(len(pos))), key=lambda x: x[0], reverse=True)))[1][0:summary_len]
                selected_lines = [sentences[i] for i in selected_lines if i < len(words[i])]
                summaries.append(selected_lines)


# In[80]:


if os.path.exists(output_folder): shutil.rmtree(output_folder)
os.mkdir(output_folder)


# In[78]:


for filename, summary in zip(filenames, summaries):
    with open(os.path.join(output_folder, filename), 'w+') as f:
        f.writelines(summary)

