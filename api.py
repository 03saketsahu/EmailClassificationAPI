#!/usr/bin/env python

# coding: utf-8

# In[1]:


# pip install numpy pandas scikit-learn flask fastapi spacy nltk re
#


# In[2]:


import pandas as pd
df = pd.read_csv(r"C:\Users\saket\Downloads\combined_emails_with_natural_pii.csv") 




# In[3]:


df['type'].unique()


# In[4]:


category_mapping = {
    "Incident": "Technical Support",
    "Problem": "Billing Issues",
    "Change": "Account Management",
    "Request": "General Inquiry"  # or map based on its context
}

df["category"] = df["type"].map(category_mapping)


# In[5]:


df.loc[0:20]


# In[6]:


print(df[0:20].to_string())


# In[7]:


import re

def mask_pii(text):
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    phone_pattern = r'\b\d{10}\b'
    name_pattern = r'(?i)(?<=my name is\s)([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)\b'
    dob_pattern = r"\b\d{4}-\d{2}-\d{2}\b"
    aadhar_num_pattern = r"\b\d{12}\b"
    credit_debit_no_pattern = r"\b(?:\d[ -]*?){13,19}\b"
    cvv_no_pattern = r"\b\d{3,4}\b"
    expiry_no_pattern = r"\b(0[1-9]|1[0-2])\/\d{2,4}\b" 



    



    text = re.sub(email_pattern, '[email]', text)
    text = re.sub(phone_pattern, '[phone_number]', text)
    text = re.sub(name_pattern, '[full_name]', text)
    text = re.sub(dob_pattern, '[dob]', text)
    text = re.sub(aadhar_num_pattern, '[aadhar_num]', text)
    text = re.sub(credit_debit_no_pattern, '[credit_debit_no]', text)
    text = re.sub(cvv_no_pattern, '[cvv_no]', text)
    text = re.sub(expiry_no_pattern, '[expiry_no]', text)


    return text



masked_df = df.copy()  # Preserve original structure
masked_df['text'] = masked_df['email'].apply(mask_pii)
      
  


# In[8]:


print(masked_df[0:10].to_string)


# In[9]:


masked_df.loc[1,'email']


# In[10]:


masked_df.loc[1,'text']


# In[11]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Prepare features
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(masked_df['text'])  # Convert emails to numerical features
y = masked_df['category']  # Labels for classification

# Train model

model = MultinomialNB()
model.fit(X, y)



# In[12]:

import joblib

# Save trained model & vectorizer
joblib.dump(model, "classification_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

model = joblib.load("classification_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import json
import nest_asyncio
nest_asyncio.apply()

app = FastAPI()

@app.post("/classify")
async def classify_email(request: Request):
    data = await request.json()
    email_body = data["email_body"]

    if not email_body:
        return JSONResponse(content={"error": "Invalid request: missing email_body"}, status_code=400)

    # Apply PII Masking
    masked_email = mask_pii(email_body)

    transformed_email = vectorizer.transform([masked_email]).toarray()  

    # Classify Email
    category = model.predict(transformed_email)[0]

    return {
        "input_email_body": email_body,
        "masked_email": masked_email,
        "category_of_the_email": category
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000) 
    


# you can check it by running this prompt in another python interface or notebook

# import requests

# url = "http://127.0.0.1:8000/classify"
# data = {"email_body": "Hello, my name is John Doe. My email is johndoe@example.com."}
# """Hello, my name is John Doe. 
#     My email is johndoe@example.com, 
#     my phone number is 9876543210, 
#     my date of birth is 1995-06-15, 
#     my Aadhar number is 123456789012, 
#     my credit card number is 4111-1111-1111-1111, 
#     my CVV is 123, 
#     and my expiry date is 09/25."""
# response = requests.post(url, json=data)

# print("Status Code:", response.status_code)
# print("Raw Response:", response.text)  # Check if the API returns valid JSON


