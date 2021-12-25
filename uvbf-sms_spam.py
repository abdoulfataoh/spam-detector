#!/usr/bin/env python
# coding: utf-8

# ![logo uvbf](https://drive.google.com/uc?export=download&id=1eP-0JTAV3p7a_mhAPHyVdFiB9pPckKr6)

# # Projet: Determiner si un sms est un spam ou pas

# ## Membres du groupe:
# - KABORE Abdoul Fataoh
# - OUEDRAOGO Ibrahim Alassane
# - ROUAMBA Fadilatou
# - ROBGO Karima

# <hr style="height:1px;" />

# # Importation & Description des donnees

# In[24]:


import pandas as pd


# - **Importation**

# In[25]:


df = pd.read_csv("sms_spams_collection.txt", sep="\t", header = None)
df.columns = ["label", "content"]


# In[26]:


df.head()


# - **Description**

# In[27]:


df.groupby('label').count()


# # Netoyage des donnees

# **Le netoyage consiste a supprimer des informations de notre dataset qui sont pas utile a notre modele**

# - **Supprimer les points de ponctuations**

# In[28]:


import string
string.punctuation


# In[29]:


def remove_punctuations(text):
    text = "".join([c for c in text if c not in string.punctuation])
    return text


# In[30]:


df["content_clean"] = df["content"].apply(lambda x: remove_punctuations(x))


# In[31]:


df


# - **Supprimer les stopwords** <br>
# Les stopwords sont des mots qui n'ont pas de valeurs ajouter a notre modele

# In[32]:


import re


# In[33]:


def tokeniser(text):
    regex = re.compile("\w+")
    words = re.findall(regex, text)
    return words


# In[34]:


df["content_tokeniser"] = df["content_clean"].apply(lambda x: tokeniser(x))
df.head()


# In[35]:


import nltk


# In[36]:


def remove_stopwords(content_list):
    result = [word for word in content_list if word not in nltk.corpus.stopwords.words("english")]
    return result


# In[37]:


df["no_stopwords"]= df["content_tokeniser"].apply(lambda x: remove_stopwords(x))
# df.head()


# In[38]:


df


# 3. Le steamming <br>
# Le steamming est une operation qui consiste a contracter du texte dans le but de reduire son poids et de le traiter de facon plus rapide

# In[39]:


ps = nltk.PorterStemmer()
wn = nltk.WordNetLemmatizer()


# In[167]:


ps.stem("racines")


# In[168]:


wn.lemmatize("racines")


# In[41]:


def stemming(content_list: list) -> list:
    result = [ps.stem(sword) for sword in content_list]
    return result


# In[42]:


def lemmatizing(content_list: list) -> list:
    result = [wn.lemmatize(lword) for lword in content_list]
    return result


# ## **(1) Notre fonction final de netoyage et contraction**

# In[60]:


def clean_sms(text: str):
    result = remove_punctuations(text)
    tokens = tokeniser(result)
    text = stemming(tokens)
    return text


# # Vectorisation

# Apres l'etape de preparation des donnees, et avant d'appliquer les algorithme de machine learning pour entrainer notre modele, nous devrons proceder a la vectorisation des donnees.
# Il s'agit de denombre le nombre d'ocurance d'un mot donnee pour chaque document

# - **CountVectorizer**

# In[53]:


from sklearn.feature_extraction.text import CountVectorizer


# In[80]:


cell = ["Nous avons un presentation de nlp. nlp est le traitement auto de langage"]


# In[105]:


full_vectorisation = CountVectorizer(analyzer=clean_sms)
vect_final = full_vectorisation.fit_transform(cell)


# In[118]:


f = full_vectorisation.get_feature_names()
d = pd.DataFrame(vect_final.toarray())
d.columns = f
d


# - **TF-IDS**

# ### TF-IDS: Term Frequency-Inverse Frequency Document Frequency
# Il s'agit de donner un poids a un **mot** dans un document
# <p>Le poids d'un mot determine le degree d'utilisation de ce mots dans un document donne par rapport aux autres documents</p>

# ## W<sub>i,j</sub> = tf<sub>i,j</sub> * log(N/df<sub>i</sub>)
# 
# - ### W<sub>i,j</sub> = Le poids du mot i dans le document j
# - ### tf<sub>i,j</sub>  = Frquence du mot i dans le document j
# - ### N = Nombre de documents total
# - ### df = Nombre de document contenant i

# In[119]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[120]:


cell = ["Nous avons un presentation de nlp. nlp est le traitement auto de langage"]


# In[121]:


vectorizer_obj = TfidfVectorizer()
vectorizer_data = vectorizer_obj.fit_transform(cell)


# In[123]:


f = full_vectorisation.get_feature_names()
d = pd.DataFrame(vectorizer_data.toarray())
d.columns = f
d


# ## **(2) Choix de la technique de vectorisation**

# In[126]:


data = pd.read_csv("sms_spams_collection.txt", sep="\t", header = None)
data.columns = ["label", "content"]
data.head()


# In[138]:


vectorisation_full = TfidfVectorizer(analyzer=clean_sms)
vect_final = vectorisation_full.fit_transform(data['content'])


# # Feature engenering

# <p>Le feature engenering est une etape crutiale dans le machine learning. il permet d'augmenter la puissance explicative d'un jeux de donnees de donnees</p>

# Methode:
#  - soit on ajoute des nouvelles variables
#  - soit on transforme les variables existantes

# ## **(3) Injecter le pourcentage de ponctuation et la longueur du text**

# In[145]:


def count_punctuation(text: str):
    nb_punc =  sum([1 for c in text if c in string.punctuation])
    return round(nb_punc/(len(text)), 4)*100


# In[143]:


def message_lenght(text: str):
    return len(text) - text.count(" ")


# In[146]:


data['content_len'] = data['content'].apply(lambda x: message_lenght(x))
data['punctuation_rate'] = data['content'].apply(lambda x: count_punctuation(x))


# In[147]:


data


# In[148]:


final_data = pd.concat([pd.DataFrame(vect_final.toarray()), data['content_len'], data['punctuation_rate']], axis=1)
final_data


# ## 4. Model & entrainement

# In[159]:


from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import train_test_split


# In[160]:


X_train, X_test, Y_train, Y_test = train_test_split(final_data, data['label'], test_size=0.2)


# In[161]:


from sklearn  import svm


# In[162]:


alg_svm= svm.SVC(kernel = 'linear')


# In[163]:


alg_svm.fit(X_train, Y_train)


# In[164]:


predictions = alg_svm.predict(X_test)


# In[165]:


precision, recall, fscore, _ = score(Y_test, predictions, pos_label='spam', average='binary')


# In[166]:


print('Precision: {} / Recall: {} / Accuracy: {}'.format(round(precision, 3),
                                                        round(recall, 3),
                                                        round((predictions==Y_test).sum() / len(predictions),3)))


# In[ ]:




