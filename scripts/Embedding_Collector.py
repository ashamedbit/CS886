import pandas as pd
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import numpy as np
from threading import Thread
from pathlib import Path
import math
import torch
import xml.etree.ElementTree as ET

def save_output(hidden, pooler, attention, folder, title):
    Path(folder + "/" + "pooler/").mkdir(parents=True, exist_ok=True)
    Path(folder + "/" + "hidden/").mkdir(parents=True, exist_ok=True)
    Path(folder + "/" + "attention/").mkdir(parents=True, exist_ok=True)
    with open(folder + "/" + "pooler/" + title + ".npy", "wb") as f:
        np.save(f, pooler.detach().cpu().numpy())
    with open(folder + "/" + "hidden/" + title + ".npy", "wb") as f:
        np.save(f, hidden.detach().cpu().numpy())
    with open(folder + "/" + "attention/" + title + ".npy", "wb") as f:
        np.save(f, attention.detach().cpu().numpy())



def get_window( **kwargs):
    for window_start in range(0, math.ceil(MAX_LEN / WINDOW_SIZE) * WINDOW_SIZE,
                              WINDOW_SIZE):
        temp = dict()
        for key, value in kwargs.items():
            temp[key] = value[:, window_start: window_start + EMBEDDING_SIZE].to(DEV)
        yield temp


def get_trial_embedding():
    tree = ET.parse('../data/Baseline Corpus/pubmed19n0001.xml')
    root = tree.getroot()

    Journal = []
    Title = []
    Abstract = []
    PubYear = []
    id = 0
    for articles in tree.iter('PubmedArticle'):
        #print("Journal "+str(id)+" data ")

        Journal += ["NA"]
        Title += ["NA"]
        Abstract += ["NA"]
        PubYear += ["NA"]

        for journal in articles.iter('Title'):
            #print("Journal is " + journal.text)
            Journal[-1] = [journal.text]

        for title in articles.iter('ArticleTitle'):
            #print("Article title is "+ title.text)
            Title[-1] = [title.text]

        for title in articles.iter('AbstractText'):
            #print("Article abstract is "+ title.text)
            Abstract[-1] = [title.text]
        
        for title in articles.iter('Journal'):
            for year in title.iter('Year'):
                #print("Article year is "+ year.text)
                PubYear[-1] = [year.text]
            break

        id= id + 1

        #if id >5:
        #    break

    df = pd.DataFrame(
    {'Journal': Journal,
     'Title': Title,
     'Abstract': Abstract,
     'PubYear': PubYear
    })
    
    print(df)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    model = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    model.to(DEV)
    for row in tqdm(df.iterrows(),total=df.shape[0]):
        output = tokenizer.batch_encode_plus([row[1].Journal[0],row[1].Title[0],row[1].Abstract[0]], return_tensors='pt', padding=True,
                                             pad_to_multiple_of=2048, truncation=True, max_length=2048)
        hidden_part = []
        pooler_part = []
        attention_part = []
        for part in get_window(**output):
            with torch.no_grad():
                m_o = model(**part)
            attention_part.append(part['attention_mask'])
            hidden_part.append(m_o.last_hidden_state)
            pooler_part.append(m_o.pooler_output)
        # save_output(m_o, "Trial_Embedding", row[1].trial_id)
        hidden_output = torch.hstack(hidden_part)
        pooler_output = torch.vstack(pooler_part)
        attention_mask = torch.concat(attention_part, dim=1)
        Thread(target=save_output, args=(hidden_output, pooler_output, attention_mask, "Trial_Embedding", "Trial")).start()

    return
    # df = pd.read_csv("Data/Trial_Data.csv")
    # df['gender'].fillna("Both", inplace=True)
    # df['maximum_age'].fillna("No Limit", inplace=True)
    # df['minimum_age'].fillna("No Limit", inplace=True)
    # df['brief_title'].fillna("", inplace=True)
    # df['detailed_description'].fillna("", inplace=True)
    # df['criteria'] = df['criteria'].map(lambda x: str(x).replace("\n", " "))
    # df['criteria'] = df['criteria'] + " " + "gender: " + df['gender'] + " " + "maximum age: " + df[
    #     'maximum_age'] + " " + "minimum age: " + df['minimum_age']
    df['abstract'] = df['brief_title'] + " " + df['detailed_description'] + " " + df['criteria']
    tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    model = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    model.to(DEV)
    for row in tqdm(df.iterrows(),total=df.shape[0]):
        output = tokenizer.batch_encode_plus([row[1].temp], return_tensors='pt', padding=True,
                                             pad_to_multiple_of=2048, truncation=True, max_length=2048)
        hidden_part = []
        pooler_part = []
        attention_part = []
        for part in get_window(**output):
            with torch.no_grad():
                m_o = model(**part)
            attention_part.append(part['attention_mask'])
            hidden_part.append(m_o.last_hidden_state)
            pooler_part.append(m_o.pooler_output)
        # save_output(m_o, "Trial_Embedding", row[1].trial_id)
        hidden_output = torch.hstack(hidden_part)
        pooler_output = torch.vstack(pooler_part)
        attention_mask = torch.concat(attention_part, dim=1)
        Thread(target=save_output, args=(hidden_output, pooler_output, attention_mask, "Trial_Embedding", row[1].trial_id)).start()

def get_patient_embedding():
    df =pd.read_csv("Data/Filtered_Topics.csv")
    print(df.head())
if __name__ == "__main__":
    MAX_LEN = 2048
    WINDOW_SIZE = 500
    EMBEDDING_SIZE = 512
    DEV = "cpu:0"

    get_trial_embedding()
