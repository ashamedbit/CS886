from fileinput import filename
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import numpy as np
from threading import Thread
from pathlib import Path
import math
import torch
import xml.etree.ElementTree as ET
import re
from os import walk
import zipfile


def obtain_folder_min_max_file(path):
    filenames = next(walk(path), (None, None, []))[2]
    minfileval =len(filenames) * 3
    maxfileval = -1
    for file in filenames:
        if re.search('xml$', file): 
            numstring = re.findall(r'\d+', file)[1]
            num = int(numstring)
            
            if num < minfileval:
                minfileval = num

            if num > maxfileval:
                maxfileval = num

    return minfileval, maxfileval

def initialize_pmids(dir):
    filenames = next(walk(path), (None, None, []))[2]
    firstpmidlist = [-1 for i in range(len(filenames))] 
    lastpmidlist = [-1 for i in range(len(filenames))] 
    return firstpmidlist,lastpmidlist
            
def obtain_file_min_max_pubmed(path):
    tree = ET.parse(path)
    root = tree.getroot()
    firstpmid = -1
    lastpmid = -1
    for articles in tree.iter('PMID'):
        lastpmid = int(articles.text)

        if firstpmid == -1:
            firstpmid = int(articles.text)

    #print(firstpmid)
    #print(lastpmid)

    return firstpmid, lastpmid

def populate_metadata(dir):

    filenames = next(walk(dir), (None, None, []))[2]
    minfileval,maxfileval = obtain_folder_min_max_file(dir)

    firstpmidlist = [-1 for i in range(1,maxfileval+2)] 
    lastpmidlist = [-1 for i in range(1,maxfileval+2)] 

    file1 = open('output.txt', 'r')
    Lines = file1.readlines()
    linecount = 0
    filled = 0
    # Strips the newline character
    for line in Lines:
        linecount = linecount+1
        temp = line.split(",")
        if linecount == 1:
            for i in range(len(temp) - 1):
                firstpmidlist[i] = int(temp[i])
                if firstpmidlist[i] != -1:
                    #print("Stored value: Index "+str(i)+ " first pmid is " + str(firstpmidlist[i]))
                    filled = filled + 1

        if linecount == 4:
            for i in range(len(temp) - 1):
                lastpmidlist[i] = int(temp[i])

                #if lastpmidlist[i] != -1:
                #    print("Stored value: Index "+str(i)+ " last pmid is " + str(lastpmidlist[i]))
  

    count = 0
    for file in filenames:
        count = count +1
        fullpath = dir +"/"+file
        numstring = re.findall(r'\d+', file)[1]
        num = int(numstring)

        if firstpmidlist[num] != -1:
            print("Skipped iteration "+str(count)+" with index " + str(num)+ " of " + str(len(filenames)))
            continue

        firstpmid,lastpmid = obtain_file_min_max_pubmed(fullpath)
       
        firstpmidlist[num] = firstpmid
        lastpmidlist[num] = lastpmid
        #print(firstpmidlist)
        #print(lastpmidlist)
        
        print("At iteration "+str(count)+"with index " + str(num)+ " of " + str(len(filenames)))
        #if count > 5:
        #    break

        with open("output.txt", "w") as txt_file:
            for line in firstpmidlist:
                txt_file.write(str(line)+ ",")
            txt_file.write("\n")
            txt_file.write("\n")
            txt_file.write("\n")
            for line in lastpmidlist:
                txt_file.write(str(line) + ",")
    
        
def binary_search_training_data(low, high, key):
    if high == low:
        return high
    
    mid = int((low+high)/2)

    length = len(str(mid))
    midstring = ""

    for i in range(0,4-length):
        midstring= "0"+midstring
        
    midstring = midstring + str(mid)
    filemid = "pubmed19n" + midstring + ".xml"

    first,last = obtain_file_min_max_pubmed("../data/Baseline Corpus/"+filemid)

    if first > key:
        high = mid -1
    elif last < key:
        low = mid +1
    elif first == key:
        return mid
    elif last == key:
        return mid
    elif first < key and last > key:
        return mid

    return binary_search_training_data(low,high,key)
    

def read_training_data():
    #data = pd.read_csv('../data/training/qrels-treceval-abstracts.2019.txt', sep=" ", header=None)
    #minfile,maxfile = obtain_folder_min_max_file("../data/Baseline Corpus")
    firstpmidlist = []
    lastpmidlist = []
    populate_metadata("../data/Baseline Corpus/Baseline Corpus-20220224T050357Z-001/Baseline Corpus")


    #binary_search_training_data(minfile, maxfile,5000)
    #obtain_file_min_max_pubmed("../data/Baseline Corpus/pubmed19n0001.xml")
    return
    for row in tqdm(data.iterrows(),total=data.shape[0]):
        queryno = row[1][0]
        pubmed = row[1][2]
        relevance = row[1][3]
        print(pubmed)


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
    read_training_data()
    return

    tree = ET.parse('../data/topics/topics2019.xml')
    root = tree.getroot()

    disease = []
    gene = []
    demographic = []
    query = []
  
    id = 0
    for articles in tree.iter('topic'):
        #print("Journal "+str(id)+" data ")

        disease += ["NA"]
        gene += ["NA"]
        demographic += ["NA"]
        query += ["NA"]

        for node in articles.iter('disease'):
            #print("Journal is " + journal.text)
            disease[-1] = [node.text]

        for node in articles.iter('gene'):
            #print("Article title is "+ title.text)
            gene[-1] = [node.text]

        for node in articles.iter('demographic'):
            #print("Article abstract is "+ title.text)
            demographic[-1] = [node.text]

        query[-1] = disease[-1] +  gene[-1] + demographic[-1]
        id= id + 1

        #if id >5:
        #    break

    #df = pd.DataFrame(
    #{'disease': Journal,
    # 'gene': Title,
    # 'demographic': Abstract
    #})

    df = pd.DataFrame(
    {'query': query
    })
    
    print(df)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    model = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    model.to(DEV)
    for row in tqdm(df.iterrows(),total=df.shape[0]):
        output = tokenizer.batch_encode_plus(row[1].query, return_tensors='pt', padding=True,
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
        #Thread(target=save_output, args=(hidden_output, pooler_output, attention_mask, "Trial_Embedding", "Trial")).start()


    return

def get_patient_embedding():
    df =pd.read_csv("Data/Filtered_Topics.csv")
    print(df.head())
if __name__ == "__main__":
    MAX_LEN = 2048
    WINDOW_SIZE = 500
    EMBEDDING_SIZE = 512
    DEV = "cpu:0"

    get_trial_embedding()
