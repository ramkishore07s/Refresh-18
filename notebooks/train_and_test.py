#!/usr/bin/env python
# coding: utf-8

# In[26]:
import logging

get_ipython().run_line_magic('run', 'dataset.ipynb')
get_ipython().run_line_magic('run', 'model.ipynb')
get_ipython().run_line_magic('run', 'utils.ipynb')
get_ipython().run_line_magic('run', 'rouge.ipynb')


# In[2]:

logging.basicConfig(format="[ %(asctime)-12s ] %(message)s", level=logging.DEBUG)
torch.cuda.set_device(1)


# In[5]:


g = NeuralSumToEmbedding(glove_filename='/home/ramkishore.s/meta/glove/glove.6B.200d.txt')
g.vectors = g.vectors.reshape(-1, g.dim)

logging.info('Loaded Glove embeddings')

# In[17]:


train = DataHandler()
train.load_dump('../data/processed/cnn/training.pkl')
train.pad(padding_idx=g.padding_idx)
train.make_batches(20)

logging.info('Loaded train data')


# In[23]:


test = DataHandler()
test.load_dump('../data/processed/cnn/test.pkl')
test.pad(padding_idx=g.padding_idx)
test.make_batches(20)

logging.info('Loaded test data')

# In[11]:


summary_scores = pickle.load(open('../data/processed/cnn/refresh_scores.pkl', 'rb'))


# In[16]:


m = EncoderDecoder(torch.cuda.FloatTensor(g.vectors), word_emb_size=g.dim, 
                       sen_emb_size=350, doc_emb_size=600, sen_len=50, batch_size=20, output_dim=2,
                       reverse=True)
logging.info(str(m.cuda()))


# In[24]:


iterations = 10
if not os.path.exists('../temp'):  os.mkdir('../temp')


# In[ ]:


for i in range(iterations):
    logging.info('iteration: ' + str(i))
    train_refresh(m, batches=train.batches, lines=train.lines, scores=summary_scores, 
                          iterations=1, max_=15, single_sample=True)
    
    get_summaries(m, test.lines, test.batches, 
                      doc_folder='../data/parsed_data/cnn/test/documents/', 
                      write_folder='../temp/', 
                      output_dim=2, summary_len=3)
    
    logging.info('computing rouge on test data')
    scores = computeRouge('../data/parsed_data/cnn/test/summaries/', '../temp/')
    
    torch.save(m.state_dict(), 'R1_f1_' + str(scores[0]) + '.params')

