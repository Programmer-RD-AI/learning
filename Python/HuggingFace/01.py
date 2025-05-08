#!/usr/bin/env python
# coding: utf-8

# # The pipeline function

# In[1]:


from transformers import pipeline


# In[2]:


classifier = pipeline("sentiment-analysis")


# In[3]:


classifier("I've been waiting for a HuggingFace Course my whole life")


# In[4]:


classifier([
    "I've been waiting for a HuggingFace Course my whole life",
    "I hate this so much!"
])


# In[5]:


classifier = pipeline('zero-shot-classification')
classifier(
    "This is a course about the Transformers library",
    candidate_labels=['education', 'politics', 'buisness']
)


# In[6]:


generator = pipeline('text-generation')
generator("In this course, we will teach you how to")


# In[7]:


generator = pipeline('text-generation', model="distilgpt2")
generator("In this course, we will teach you how to", max_length=30, num_return_sequences=2) # no. of words and no. of like generated texts


# In[8]:


unmasker = pipeline('fill-mask')
unmasker('This course will teach you all about <mask> modoels', top_k=2)


# In[9]:


ner = pipeline('ner')
ner("My name is Sylavin and I work at Hugging Face in Brookyln")


# In[10]:


question_answerer = pipeline("question-answering")
question_answerer(question="Where do I work?", context="My name is Sylavin and I work at Hugging Face in Brooklyn")


# In[11]:


summarizer = pipeline("summarization")
summarizer("A critical phase in Roman history was the Punic Wars against Carthage (264–146 BC). The First Punic War (264–241 BC) established Rome as a naval power, while the Second Punic War (218–201 BC) saw the rise of Hannibal, whose daring crossing of the Alps remains legendary. Despite early Carthaginian victories, Rome eventually triumphed, culminating in the Third Punic War (149–146 BC), which led to the complete destruction of Carthage.")


# In[12]:


translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
translator("Ce cours est produit par Hugging Face")


# # What happens inside the pipeline function? (PyTorch)

# In[13]:


from transformers import AutoTokenizer


# In[14]:


checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


# In[15]:


raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!"
]


# In[16]:


inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")


# In[17]:


inputs


# In[18]:


from transformers import AutoModel


# In[19]:


checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModel.from_pretrained(checkpoint)


# In[20]:


outputs = model(**inputs)


# In[21]:


print(outputs.last_hidden_state.shape)


# In[22]:


from transformers import AutoModelForSequenceClassification


# In[23]:


checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)


# In[24]:


outputs = model(**inputs)


# In[25]:


outputs.logits


# In[26]:


import torch
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)


# In[27]:


predictions


# In[28]:


model.config.id2label


# # Instantiate a Transformers model (PyTorch)
# 

# In[29]:


from transformers import AutoModel


# In[30]:


bert_model = AutoModel.from_pretrained("bert-base-cased")
bert_config = AutoModel.from_pretrained("bert-base-cased")


# In[31]:


gpt_model = AutoModel.from_pretrained("gpt2")
gpt_config = AutoModel.from_pretrained("gpt2")


# In[32]:


bart_model = AutoModel.from_pretrained("facebook/bart-base")
bart_config = AutoModel.from_pretrained("facebook/bart-base")


# In[38]:


from transformers import BertConfig, BertModel
bert_config = BertConfig.from_pretrained("bert-base-cased")
bert_model = BertModel(bert_config)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




