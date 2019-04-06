#!/usr/bin/env python
# coding: utf-8

# In[12]:
import os
get_ipython().run_line_magic('run', 'embeddings.ipynb')
get_ipython().run_line_magic('run', 'dataset.ipynb')
get_ipython().run_line_magic('run', 'label_generator.ipynb')


# **Parse raw data into different folders**

# In[23]:
if not os.path.exists: os.make_dirs('../data/parsed_data/cnn/')
if not os.path.exists: os.make_dirs('../data/parsed_data/dailymail')

parse_all('../data/neuralsum/cnn/', '../data/parsed_data/cnn/')
parse_all('../data/neuralsum/dailymail/', '../data/parsed_data/dailymail/')


# **Convert parsed training documents to embedding indices**

# In[24]:
if not os.path.exists: os.make_dirs('../data/processed/cnn/')
if not os.path.exists: os.make_dirs('../data/processed/dailymail')

data_converter = NeuralSumToEmbedding('/home/ramkishore.s/meta/glove/glove.6B.200d.txt')
data_converter.root_convert('../data/parsed_data/cnn/', '../data/processed/cnn/')
data_converter.root_convert('../data/parsed_data/dailymail/', '../data/processed/dailymail/')


# **Create Greedy summary labels**

# In[ ]:


greedy_label_generator('../data/parsed_data/cnn/training/documents/', 
                       '../data/parsed_data/cnn/training/summaries/',
                       '../data/processed/cnn/greedy_train_summary_labels.pkl')
greedy_label_generator('../data/parsed_data/dailymail/training/documents/', 
                       '../data/parsed_data/dailymail/training/summaries/',
                       '../data/processed/dailymail/greedy_train_summary_labels.pkl')

# Generate multiple summary scores

top_k_summaries_folder('../data/parsed_data/cnn/training/documents/', 
                            '../data/parsed_data/cnn/training/summaries',
                            dump_file='../data/processed/cnn/refresh_scores.pkl',
                            max_sum_len=4, max_sum_select=15)
top_k_summaries_folder('../data/parsed_data/dailymail/training/documents/', 
                            '../data/parsed_data/dailymail/training/summaries',
                            dump_file='../data/processed/dailymail/refresh_scores.pkl',
                            max_sum_len=5, max_sum_select=15)


