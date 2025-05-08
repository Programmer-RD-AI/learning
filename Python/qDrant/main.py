#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system("pip install 'qdrant-client[fastembed]' --upgrade")


# In[2]:


QDRANT_HOST="https://3bdbdcd2-6854-48ab-b5c4-c22a0da98431.europe-west3-0.gcp.cloud.qdrant.io"
QDRANT_API_KEY="eR-0gtWq9ljNyq1b4dFK0sGn9xXpbOmp-UR7J_fEl2ia11ckAWzijw"
QDRANT_PORT=6333


# In[3]:


from qdrant_client import AsyncQdrantClient, QdrantClient

# client = QdrantClient(path="path/to/db")  # Persists changes to disk
# or
client = QdrantClient(url=QDRANT_HOST, port=QDRANT_PORT, api_key=QDRANT_API_KEY)


# In[4]:


# Prepare your documents, metadata, and IDs
docs = ["Qdrant has Langchain integrations", "Qdrant also has Llama Index integrations"]

client.add(
    collection_name="demo_collection",
    documents=docs,
)


# In[12]:


search_result = client.query_points(
    collection_name="test_collection",
)
print(search_result)


# In[13]:


search_result.points


# In[7]:


type(search_result)


# In[6]:


from qdrant_client.http.models import Distance, VectorParams

if not client.collection_exists("test_collection"):
    client.create_collection(
            collection_name="test_collection",
            vectors_config=VectorParams(size=8, distance=Distance.DOT),
    )


# In[7]:


from qdrant_client.http.models import PointStruct

operation_info = client.upsert(
    collection_name="test_collection",
    wait=True,
    points=[
        PointStruct(id=1, vector=[0.05, 0.61, 0.76, 0.74, 0.05, 0.61, 0.76, 0.74], payload={"city": "Berlin"}),
        PointStruct(id=2, vector=[0.05, 0.61, 0.76, 0.74, 0.05, 0.61, 0.76, 0.74], payload={"city": "London"}),
        PointStruct(id=3, vector=[0.05, 0.61, 0.76, 0.74, 0.05, 0.61, 0.76, 0.74], payload={"city": "Moscow"}),
        PointStruct(id=4, vector=[0.05, 0.61, 0.76, 0.74, 0.05, 0.61, 0.76, 0.74], payload={"city": "New York"}),
        PointStruct(id=5, vector=[0.05, 0.61, 0.76, 0.74, 0.05, 0.61, 0.76, 0.74], payload={"city": "Beijing"}),
        PointStruct(id=6, vector=[0.05, 0.61, 0.76, 0.74, 0.05, 0.61, 0.76, 0.74], payload={"city": "Mumbai"}),
    ]
)
print(operation_info)


# In[9]:


search_result = client.search(
    collection_name="test_collection",
    query_vector=[0.18, 0.81, 0.75, 0.12, 0.12, 0.69, 0.58, 0.54],
    limit=1
)
print(search_result)


# In[11]:


from qdrant_client.http.models import Filter, FieldCondition, MatchValue

search_result = client.search(
    collection_name="test_collection",
    query_vector=[0.18, 0.81, 0.75, 0.12, 0.12, 0.69, 0.58, 0.54],
    query_filter=Filter(
        must=[
            FieldCondition(
                key="city",
                match=MatchValue(value="London")
            )
        ]
    ),
    limit=1
)
print(search_result)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




