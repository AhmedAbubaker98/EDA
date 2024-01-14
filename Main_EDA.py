import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import google.generativeai as genai
from openai import OpenAI
from PIL import Image
from pathlib import Path
import os 

generation_config = {
  "temperature": 0,
  "top_p": 1,
  "top_k": 1,
  "max_output_tokens": 2048,
}
 
safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  }
]

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
model = genai.GenerativeModel(model_name="gemini-pro-vision",
                              generation_config=generation_config,
                              safety_settings=safety_settings)

def GraphPromptBuilder(PS, df1, df2, df3, f1, f2, f3):
    prompt = f"""You are Cora, an automated AI-powered data analysis software. You, Gemini, are at the heart of this system, responsible for providing code for simple readable graphs. 
    based on the following problem statement '{PS}' provide 3 graphs that would extract relevant information from the data files: {f1}, {f2}, {f3}.
    
    {f1}:
    1. head is\n {df1.head()}
    2. the data types are\n {df1.dtypes}
    3. the null values are\n {df1.isnull().sum()}
    4. the summary is\n {df1.describe()}

    {f2}:
    1. head is\n {df2.head()}
    2. the data types are\n {df2.dtypes}
    3. the null values are\n {df2.isnull().sum()}
    4. the summary is\n {df2.describe()}

    {f3}:
    1. head is\n {df3.head()}
    2. the data types are\n {df3.dtypes}
    3. the null values are\n {df3.isnull().sum()}
    4. the summary is\n {df3.describe()}
    
    remember:
    1. only write the code for the graphs, not the graphs themselves.
    2. your reply in its entirety will be executed in a python environment, therefore, write all of the code at once.
    3. available libraries are pandas, numpy, seaborn, matplotlib.
    4. write '#1 start' and '#1 end' at the start end of the first graph, '#2 start' and '#2 end' at the start end of the second graph, '#3 start' and '#3 end' at the start end of the third graph.
     """
    return prompt

def AnalysisPromptBuilder(PS):
    prompt = f"""You are Cora, an automated AI-powered data analysis software. You, Gemini, are at the heart of this system, responsible for reading given graphs 
    and providing analysis based on the problem statement, '{PS}' here are the graph(s) that you should extract relevant information from the provide image(s).
     """
    return prompt


Cab_Data = pd.read_csv('Cab_Data.csv')
City = pd.read_csv('City.csv')
Customer_ID = pd.read_csv('Customer_ID.csv')
Transaction_ID = pd.read_csv('Transaction_ID.csv')

PS = "I want to know which company is the best for investing, Pink Cab or Yellow Cab."

prompt1 = GraphPromptBuilder(PS, Cab_Data, City, Transaction_ID, "Cab_Data.csv" , "City.csv", "Transaction_ID.csv")

#OPENAI Send the prompt to the model for completion
chat_completion = client.chat.completions.create(
    messages=[{'role': 'user', 'content': prompt1}],
    model="gpt-3.5-turbo"
    )

#clean the response from anything that is not code
a = chat_completion.choices[0].message.content.strip()
a = a[a.find("import"):]
a = a.replace("```python", "")
a = a.replace("```", "")
a = a.replace("plt.show()", "")

a = a.replace("#1 end", 'plt.savefig("graph1.png")')
a = a.replace("#2 end", 'plt.savefig("graph2.png")')
a = a.replace("#3 end", 'plt.savefig("graph3.png")\nggg')
a = a[:a.find("ggg")]

print(a)
exec(a)

#GOOGLE GMEINI
prompt2 = AnalysisPromptBuilder(PS)

#convert graph to base64
graph1 = Image.open("graph1.png")
graph2 = Image.open("graph2.png")
graph3 = Image.open("graph3.png")

response1 = model.generate_content([prompt2, graph1, graph2, graph3], generation_config={"temperature": 0.4, "max_output_tokens": 4096})

print({response1.text})

