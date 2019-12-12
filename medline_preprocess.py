#!/usr/bin/env python
# coding: utf-8

# In[1]:


import xml.etree.ElementTree as et
import pandas as pd
import glob
from itertools import chain
from nltk import word_tokenize


# In[ ]:


#given root of each .xml file, select the articles with both title and abstract available
#we also select those with keywords for further application if possible
def text_kw(root):
    result_title = []
    result_abst = []
    result_kw = []
    for article in root:
        tmp_title = 0
        tmp_abst = 0
        tmp_kw = []
        for title in article.iter('ArticleTitle'):
            tmp_title = title.text
        for abst in article.iter('AbstractText'):
            tmp_abst = abst.text
        for kw in article.iter('Keyword'):
            tmp_kw.append(kw.text)
        if tmp_abst != 0 and tmp_title != 0 and tmp_kw != []:
            result_title.append(tmp_title)
            result_abst.append(tmp_abst)
            result_kw.append(tmp_kw)
    return [result_title, result_abst, result_kw]


# In[ ]:


#parse multiple .xml raw files from the MedLine database
title = []
abst = []
kw = []
count = 0
for name in glob.glob('*.xml'):
    print(count)
    tree = et.parse(name)
    root = tree.getroot()
    tmp_result = text_kw(root)
    title.append(tmp_result[0])
    abst.append(tmp_result[1])
    kw.append(tmp_result[2])
    count += 1


# In[ ]:


#the steps above are performed in several different batches
#output text from a batch of .xml files to csv for storage
#load csv files later to filter out certain articles
title_flat = list(chain.from_iterable(title))
abst_flat = list(chain.from_iterable(abst))
kw_flat = list(chain.from_iterable(kw))
df = pd.DataFrame()
df['title'] = title_flat
df['abst'] = abst_flat
df.to_csv('Text_941-972.csv', index = None, header = None)
df2 = pd.DataFrame(kw_flat)
df2.to_csv('KW_941-972.csv', index = None, header = None)


# In[ ]:


#filter titles and abstracts < 10 char
text_namelst = glob.glob('Text*.csv')
kw_namelst = glob.glob('KW*.csv')
for i in range(len(kw_namelst)):
    text = pd.read_csv(text_namelst[i], header = None, dtype = str)
    kw = pd.read_csv(kw_namelst[i], header = None, dtype = str)
    text.columns = ['title', 'abst']
    cond = (text['title'].str.len() >= 10) & (text['abst'].str.len() >= 10)
    text = text.loc[cond]
    kw = kw.iloc[text.index]
    text.to_csv('Filtered_' + text_namelst[i], index = None, header = None)
    kw.to_csv('Filtered_' + kw_namelst[i], index = None, header = None)


# In[ ]:


#filter for <500
text_namelst = glob.glob('Filtered_Text*.csv')
kw_namelst = glob.glob('Filtered_KW*.csv')
for i in range(len(kw_namelst)):
    text = pd.read_csv(text_namelst[i], header = None, dtype = str)
    kw = pd.read_csv(kw_namelst[i], header = None, dtype = str)
    text.columns = ['title', 'abst']
    cond = ((text['abst'].str.len() <= 500))
    text = text.loc[cond]
    kw = kw.iloc[text.index]
    text.to_csv(text_namelst[i], index = None, header = None)
    kw.to_csv(kw_namelst[i], index = None, header = None)


# In[32]:


#filter titles with '['
#title with [] are translated titles
text_namelst = glob.glob('Filtered_Text*.csv')
kw_namelst = glob.glob('Filtered_KW*.csv')
for i in range(len(kw_namelst)):
    text = pd.read_csv(text_namelst[i], header = None, dtype = str)
    kw = pd.read_csv(kw_namelst[i], header = None, dtype = str)
    text.columns = ['title', 'abst']
    cond = ((text['title'].str[0] != '['))
    text = text.loc[cond]
    kw = kw.iloc[text.index]
    text.to_csv(text_namelst[i], index = None, header = None)
    kw.to_csv(kw_namelst[i], index = None, header = None)


# In[49]:


#concat output from all batches
full_text = pd.DataFrame()
full_kw = pd.DataFrame()
text_namelst = glob.glob('Filtered_Text*.csv')
kw_namelst = glob.glob('Filtered_KW*.csv')
for i in range(len(kw_namelst)):
    text = pd.read_csv(text_namelst[i], header = None, dtype = str)
    kw = pd.read_csv(kw_namelst[i], header = None, dtype = str)
    full_text = pd.concat([full_text, text])
    full_kw = pd.concat([full_kw, kw])
full_text.to_csv('full_text', index = None, header = None)
full_kw.to_csv('full_kw', index = None, header = None)

