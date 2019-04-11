# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 19:24:08 2017

@author: Kim.Vuong
"""

import pandas as pd
import numpy as np
import spacy
from statistics import mean
import itertools, collections
from itertools import chain
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.svm import LinearSVC, SVC
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, f_classif, SelectFromModel,chi2, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler, Normalizer, LabelBinarizer,LabelEncoder
from selecttextitem import TextSelector #custom script
from selectnumitem import NumberSelector #custom script

import nltk
from nltk.corpus import stopwords

englishstopwords = stopwords.words('english')

#%%

f = 'C:/Users/Kim.Vuong/Documents/Kaggle/Spooky Author Identification/train.csv'

t = 'C:/Users/Kim.Vuong/Documents/Kaggle/Spooky Author Identification/test.csv'

df = pd.read_csv(f)
dft = pd.read_csv(t)

#%% look at the data

print(df.shape)

df.info()

df['author'].value_counts()
df['author'].hist()


#look at segments of the data
df[df['author'] == 'EAP']['text'].head()#.sort_values('id')

#any missing values?

df['text'].isnull().value_counts()
df['author'].isnull().value_counts()

#count tags

L = [pos for pos in df['pos']]#eval(label)
counter_label = collections.Counter(itertools.chain(*L))
print(counter_label)

#%% spacy

nlp = spacy.load('en_core_web_md') # 'en_default' = "en_core_web_md" but stop word isn't working.
#add stop words from en_default
nlp.vocab.add_flag(lambda s: s in spacy.en.word_sets.STOP_WORDS, spacy.attrs.IS_STOP)

df['docs'] = [doc for doc in nlp.pipe(df['text'], batch_size=1000, n_threads=4)]

dft['docs'] = [doc for doc in nlp.pipe(dft['text'], batch_size=1000, n_threads=4)]

#%% explore spacy tokens, ents etc.
dir(df['docs'][0]) # docs - descriptions of what we have for the document
dir(df['docs'][0][1]) # tokens  e.g. first doc- information we have for the tokens
[(e, e.label_) for e in df['docs'][13].ents] # named entites for doc 1
[(tok,tok.lemma_, tok.pos_,tok.is_punct,tok.tag_, tok.is_stop, tok.lower_) for tok in df['docs'][19]] # returns token, Pos and whether stop word
df['tokens'][1]

#sentences in a doc
t = [sents for sents in df['docs'][1000].sents]


doc1 = nlp(u'London is a big city in the United Kingdom.')
for ent in doc1.ents:
    print(ent.label_, ent.text)

#%% simple tokenisation first

def sense2vec(token):
    text = token.text.replace(' ', '_')
    tag = token.ent_type_ or token.pos_
    #return '{0}|{1}'.format(text, tag)
    return '{0}'.format(text)

#filter nouns only
def filter_noun(tok):
    return tok.tag_ in ["NN","NNP","NNPS","NNS"]

#entity
EntityType = ['PERSON','NORP','FACILITY','ORG','GPE','LOC','PRODUCT','EVENT','WORK_OF_ART','LANGUAGE']
def select_text(t):
    return t.ent_type_ if (t.ent_type_ and t.ent_type_ not in EntityType) else t.text
    
def entity_or_lemma(t):
    text = t.text if t.ent_type else t.lemma_
    return text.replace(' ', '_')

def preprocess(doc):
    # merge entities
    for ent in doc.ents:
        if len(ent) > 1:
            ent.merge(ent.root.tag_, ent.text, ent.label_)
    
    # return our tokens
    return [entity_or_lemma(tok) for tok in doc if not filter_token(tok)]

#filter if stop or punctuation ,"SYM", "NUM"
def filter_token(tok):
    return tok.is_stop or tok.is_punct or tok.pos_ in ["PUNCT"]\
            or tok.lower_ in englishstopwords\
            #or tok.text in ["'s"]
            
def lowering(article):
    return [t.text.lower() for t in article] #t.text
#%% Sentence tokenisation

#number of words per sentence     
df['num_sent'] = df['docs'].apply(lambda doc: [len(s) for s in doc.sents])
dft['num_sent'] = dft['docs'].apply(lambda doc: [len(s) for s in doc.sents])

#average number of words per sentence
df['avg_sent'] = df['num_sent'].apply(lambda sents: mean(sents))
dft['avg_sent'] = dft['num_sent'].apply(lambda sents: mean(sents))

#%%

#average word length
#number of commas
#no. words not stop words
#length?
#function word count?
#noun phrases?

#%% PoS 

df['pos'] = [[tok.pos_ for tok in doc] for doc in df['docs']] #if not filter_token(tok)
dft['pos'] = [[tok.pos_ for tok in doc] for doc in dft['docs']] #if not filter_token(tok)


#%%
    
all_words = chain.from_iterable([words for rownum, words in df['tokens'].iteritems()])
words = pd.Series(list(all_words)).value_counts()
words.to_csv('C:/Users/Kim.Vuong/Documents/Kaggle/Spooky Author Identification/full_freq_tokens.csv',header=True, index=True, encoding='utf-8')

#%%
### applying the tokenisation

#just tokens - removed stop words
df['tokens'] = df['docs'].apply(lambda doc: [tok for tok in doc if not filter_token(tok)])
dft['tokens'] = dft['docs'].apply(lambda doc: [tok for tok in doc if not filter_token(tok)])

#%%

#all words - inc stop

df['tokens'] = [[tok for tok in doc] for doc in df['docs']] #if not filter_token(tok)
dft['tokens'] = [[tok for tok in doc] for doc in dft['docs']] #if not filter_token(tok)


#%%

#nouns only

df['tokens'] = df['docs'].apply(lambda doc: [tok for tok in doc if filter_noun(tok)])
dft['tokens'] = dft['docs'].apply(lambda doc: [tok for tok in doc if filter_noun(tok)])

#%%
#just tokens and using sense2vec
df['tokens'] = df['docs'].apply(lambda doc: [sense2vec(t) for t in doc if not filter_token(t)])
dft['tokens'] = dft['docs'].apply(lambda doc: [sense2vec(t) for t in doc if not filter_token(t)])

#%%
#replace where there is lemma or if not entities (not working yet)

df['tokens'] = df['docs'].apply(preprocess)
dft['tokens'] = dft['docs'].apply(preprocess)
#%%
#entities only if not then just token

df['tokens'] = df['docs'].apply(lambda doc: [select_text(toks) for toks in doc if not filter_token(toks)])
dft['tokens'] = dft['docs'].apply(lambda doc: [select_text(toks) for toks in doc if not filter_token(toks)])
#%%
#what about just lemma and text?
df['tokens'] = [[tok.lemma_ for tok in doc if not filter_token(tok)]
                for doc in df['docs']]
dft['tokens'] = [[tok.lemma_ for tok in doc if not filter_token(tok)]
                for doc in dft['docs']]
#%% entity and text

df['tokens'] = df['docs'].apply(lambda doc: [select_text(toks) for toks in doc if not filter_token(toks)])
dft['tokens'] = dft['docs'].apply(lambda doc: [select_text(toks) for toks in doc if not filter_token(toks)])

#%%
#lower
df['tokens'] = df['tokens'].apply(lowering)
dft['tokens'] = dft['tokens'].apply(lowering)

#%%
#encode the author


le = LabelEncoder()
Y_train = le.fit_transform(df['author'])

#%% Train test split

X_train = df[['tokens','pos','avg_sent']]

#Y_train = df['author']

XT = dft[['tokens','pos','avg_sent']]


docs_train, docs_test, labels_train, labels_test = train_test_split(
        X_train, Y_train, test_size=0.1, random_state=42)

#%% train model

def tok(x):
    return x

def prep(x):
    return x

tfidfvectorizer = TfidfVectorizer(tokenizer=tok, preprocessor=prep, 
                                 ngram_range=(1,2)) # to act as countvectorizer add use_idf=False, norm=None
countvectorizer = CountVectorizer(tokenizer=tok, preprocessor=prep, 
                                  ngram_range=(1,2))

#create individual feature extraction pipelines
tokens = Pipeline([
        ('selector', TextSelector(key='tokens')),
        ('tfidf', tfidfvectorizer),
        #('counter', countvectorizer)
        ])
        
pos = Pipeline([
        ('selector', TextSelector(key='pos')),
        ('counter', countvectorizer)
        ])
    
avgsent = Pipeline([
        ('selector', NumberSelector(key='avg_sent')),
        #('standard', StandardScaler())
        ('normalize',Normalizer())
        ])

#create a feature union of the individual pipelines
features = FeatureUnion([
        ('tokens', tokens),
        #('pos', pos),
        #('avgsent',avgsent)
        ])

#creating a pipeline for the feature creation to see if it works 
#feature_processing = Pipeline([('feats', features)])
#feature_processing.fit_transform(X_train)

#creating pipeline for modelling.
    
clf0 = CalibratedClassifierCV(LinearSVC(), cv=10)
    
model = Pipeline([
        ('features', features),
        ('classifier',clf0)
        #('NB',MultinomialNB())
        ])
    
#%%

#print(sorted(model.get_params()))
    
# gridsearch params
#look for the best hyperparameters for SVC
#make one list and not multiples dict
parameters = [
        {#'classifier__estimator__base_estimator__C':[10,50,100],
         #'features__pos__counter__ngram_range':[(1,1),(1,2),(1,3)],
         'features__tokens__tfidf__ngram_range': [(1,1),(1,2),(1,3)],
         'features__tokens__tfidf__use_idf': [True,False],
         'features__tokens__tfidf__norm': ['l1', 'l2',None]
         #'ovr__estimator__feats__k':[50, 100, 500]}
        #,{'vectorizer__max_df':[0.5,0.4,0.3]}
        #,{'vectorizer__min_df':[3,4,5]}
        #{'vectorizer__max_features':[10000, 25000, 50000, 100000,150000]}
        #{'classifier__estimator__C':[0.1,1,10],'classifier__estimator__kernel':['rbf'],'classifier__estimator__gamma':[0.000001,0.00001,0.0001,0.001]}
        }]
    
gs = GridSearchCV(estimator=model, param_grid=parameters, cv=5, scoring='neg_log_loss')     
gs.fit(X_train, Y_train) #mod to use grid search or "model" for the pipeline only

docs_test_pred = gs.predict_proba(docs_test)
logloss = log_loss(labels_test,docs_test_pred)
    
    # distance of your data points from the hyperplane that separates the data
    #print(mod.decision_function(docs_test))
    
print(gs.grid_scores_)
print(gs.cv_results_)
print(gs.best_params_)    
#%%    


# train
model.fit(docs_train, labels_train) #mod to use grid search or "model" for the pipeline only
    
####evaluation
# test  
docs_test_pred = model.predict_proba(docs_test)

logloss = log_loss(labels_test,docs_test_pred)

labels_predict = model.predict(docs_test)

#%% model on whole dataset

model.fit(X_train,Y_train)

#using gridsearch

gs.fit(X_train,Y_train) 
proba = gs.predict_proba(XT)

   
#%% predicting test file
proba = model.predict_proba(XT)

proba = pd.DataFrame(proba,columns=['EAP','HPL','MWS'])

proba['id']= dft['id']
proba.set_index('id', inplace=True)

proba.to_csv('C:/Users/Kim.Vuong/documents/Kaggle/Spooky Author Identification/test submissions/021.csv')

