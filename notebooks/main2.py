#!/usr/bin/env python
# coding: utf-8

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, help='any of cnn/dm/both')
parser.add_argument('--gpu', type=int, help='torch.cuda.set_device(<..>)')
parser.add_argument('--variable_train_summary_len', type=str, help='use variable summary lengths for training refresh')
parser.add_argument('--log_file', type=str, help='name of the log file')
parser.add_argument('--iterations', type=int, help='no of iterations')
parser.add_argument('--testset', type=str, help='any of cnn/dm/both')
parser.add_argument('--single_sample', type=int, help='0/1. 0-> use average of samples')
parser.add_argument('--encoder_type', type=str, help='lin/rev/tree')

args = parser.parse_args()
print(args)

args.single_sample = bool(args.single_sample)


# **Training**

# In[2]:

get_ipython().run_line_magic('run', 'paths.ipynb')
get_ipython().run_line_magic('run', 'embeddings.ipynb')
get_ipython().run_line_magic('run', 'dataset.ipynb')
get_ipython().run_line_magic('run', 'Unsupervised-Tree-LSTM/treeLSTM.ipynb')
get_ipython().run_line_magic('run', 'model.ipynb')
get_ipython().run_line_magic('run', 'utils.ipynb')
get_ipython().run_line_magic('run', 'rouge.ipynb')


from rouge import Rouge


# In[ ]:


torch.cuda.set_device(args.gpu)


# In[3]:

g = NeuralSumToEmbedding(dump_filename=CONFIG.GLOVE_CACHE_FILENAME)
g.vectors = g.vectors.reshape(-1, g.dim)


d = NeuralSumDataHandler()
r = RougeNeuralSum()

if args.dataset in ['cnn', 'both']:
    d.load_padded_dump(CONFIG.CNN_PADDED_TRAIN_DUMP)
    if args.variable_train_summary_len=='true': 
        r.load(CONFIG.CNN_TRAIN_R1_R2_RL_DUMP_V, max_=5)
    else:
        r.load(CONFIG.CNN_TRAIN_R1_R2_RL_DUMP, max_=5)
        
if args.dataset in ['dm', 'both']:
    
    if args.dataset == 'dm': d.load_padded_dump(CONFIG.DAILYMAIL_PADDED_TRAIN_DUMP)
    else: d.extend_padded_dump(CONFIG.DAILYMAIL_PADDED_TRAIN_DUMP)
        
    if args.variable_train_summary_len=='true': 
        r.load(CONFIG.DAILYMAIL_TRAIN_R1_R2_RL_DUMP_V, max_=15)
    else:
        r.load(CONFIG.DAILYMAIL_TRAIN_R1_R2_RL_DUMP, max_=15)

d.make_batches(20)

print('loaded dataset: ' + str(len(d.lines)) + ' elements')


torch.cuda.set_device(args.gpu)


# In[9]:

if args.encoder_type == 'lin':
    m = EncoderDecoder(torch.cuda.FloatTensor(g.vectors), word_emb_size=g.dim, 
                       sen_emb_size=350, doc_emb_size=600, sen_len=50, batch_size=20, output_dim=2)
if args.encoder_type == 'rev':
    m = EncoderDecoder(torch.cuda.FloatTensor(g.vectors), word_emb_size=g.dim, 
                       sen_emb_size=350, doc_emb_size=600, sen_len=50, batch_size=20, output_dim=2,
                       reverse=True)
if args.encoder_type == 'tree':
    m = NeuralDiscourseSum(torch.cuda.FloatTensor(g.vectors), word_emb_size=g.dim, 
                       sen_emb_size=350, doc_emb_size=600, sen_len=50, batch_size=20, output_dim=2)

print('\n', m.cuda(), '\n')


# In[13]:

if args.testset in ['dm', 'both']:
    d2 = NeuralSumDataHandler()
    d2.load_padded_dump(CONFIG.DAILYMAIL_PADDED_TEST_DUMP)
    d2.make_batches(20)

if args.testset in ['cnn', 'both']:
    d3 = NeuralSumDataHandler()
    d3.load_padded_dump(CONFIG.CNN_PADDED_TEST_DUMP)
    d3.make_batches(20)


# In[14]:

files = os.listdir(CONFIG.PREDICTED_SUMMARY_FOLDER)
for filename in files:
    os.remove(os.path.join(CONFIG.PREDICTED_SUMMARY_FOLDER, filename))

iterations = args.iterations


# In[ ]:

with open(args.log_file, 'w+') as log:
    for i in range(0, iterations):
        log.write('iteration: ' + str(i + 1) + '\n')
        m.documentEncoder.training = True
        m.training = True
        d.make_batches(20)
        train_refresh(m, batches=d.batches, lines=d.lines, scores=r.summary_scores, 
                      iterations=1, max_=15, single_sample=args.single_sample)
        m.documentEncoder.training = False
        m.training = False
        
        get_summaries(m, d3.lines, d3.batches, 
                      doc_folder=CONFIG.CNN_TEST_DOCUMENTS_FOLDER, 
                      write_folder=CONFIG.PREDICTED_SUMMARY_FOLDER, 
                      output_dim=2, summary_len=3)


        filenames = os.listdir(CONFIG.PREDICTED_SUMMARY_FOLDER)
        hyps = [open(CONFIG.PREDICTED_SUMMARY_FOLDER + filename).read().replace('\n', ' ') for filename in filenames]
        refs = [open(CONFIG.CNN_TEST_SUMMARY_FOLDER + filename).read().replace('\n', ' ') for filename in filenames]
        rouge = Rouge()
        # or
        scores = rouge.get_scores(hyps, refs, avg=True)
        log.write('\n' + str(scores['rouge-1']['f']) + ' ' + str(scores['rouge-2']['f']) + ' ' + 
                  str(scores['rouge-l']['f']) + '\n' )
        torch.save(m.state_dict(), '../parameters/' + args.dataset + args.variable_train_summary_len + str(args.single_sample) + '_' + str(i))
        print(' result: ', str(scores['rouge-1']['f']) + ' ' + str(scores['rouge-2']['f']) + ' ' + 
              str(scores['rouge-l']['f']))
        #output = computeRouge(CONFIG.PREDICTED_SUMMARY_FOLDER, CONFIG.CNN_TEST_SUMMARY_FOLDER)
        #print(output[1])



# In[16]:


#o
