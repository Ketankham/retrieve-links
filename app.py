from gpt_index import GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain import OpenAI
import sys
import os
import requests
import pandas as pd
import openai
from dotenv import load_dotenv


load_dotenv()  

import numpy as np
from openai.embeddings_utils import get_embedding, cosine_similarity
from flask import Flask, request, jsonify
import json
app = Flask(__name__)
datafile_path = "documents1-with-embeddings1-removed-longer.csv"
df = pd.read_csv(datafile_path)
df["embedding"] = df.embedding.apply(eval).apply(np.array)
#os.environ['OPENAI_API_KEY'] = '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'
openai.api_key = os.environ['OPENAI_API_KEY']

datafile_path = "documents1-with-embeddings1-removed-longer.csv"
print(df.head())
#df = pd.read_csv(datafile_path)
#df["embeddings"] = df.embeddings.apply(eval).apply(np.array)
def search_reviews(df, product_description, n=3, pprint=True):
    product_embedding = get_embedding(
        product_description,
        engine="text-embedding-ada-002"
    )
    df["similarity"] = df.embedding.apply(lambda x: cosine_similarity(x, product_embedding))

    results = (
        df.sort_values("similarity", ascending=False)
        .head(n)
        .metadata
    )
    if pprint:
        for r in results:
            print(r)
            print()
    return results
@app.route('/search', methods=['POST'])
def search():
    product_description = request.json['product_description']
    results = search_reviews(df, product_description, n=4)
    results_dict = results.to_dict()
    return jsonify(results_dict)  
    





'''@app.route('/query', methods=['POST'])
def query():
    query_text = request.json['query']
    os.environ['OPENAI_API_KEY'] = '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'
    index = GPTSimpleVectorIndex.load_from_disk('bubble-index.json')
    response = index.query(query_text, response_mode="compact")
    return jsonify({'response': response})
    #print(response.response)'''

if __name__ == '__main__':
    app.run(debug=True)

