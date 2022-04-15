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
import os.path
from os import walk
from bisect import bisect_left

def obtain_pmid_from_file(dir,pmid):
    if not(os.path.exists(dir)):
        print("File "+dir+" does not exist")
        return "","","","","",""


    tree = ET.parse(dir)
    root = tree.getroot()
    firstpmid = -1
    lastpmid = -1
    print(dir)
    Journal = ""
    Title = ""
    Abstract = ""
    Keyword = ""
    Chemicals = ""
    Meshdescriptor = ""
    for articles in tree.iter('PubmedArticle'):
        for node in articles.iter('PMID'):
            articlepmid = (int)(node.text)
            if pmid == articlepmid:
                print("Found "+str(pmid))
                for journal in articles.iter('Title'):
                    #print("Journal is " + journal.text)
                    Journal = journal.text

                for title in articles.iter('ArticleTitle'):
                    #print("Article title is "+ title.text)
                    Title = title.text

                for title in articles.iter('AbstractText'):
                    #print("Article abstract is "+ title.text)
                    Abstract = title.text

                for  keyword in articles.iter('KeywordList'):
                    for node in keyword.iter('Keyword'):
                        Keyword = Keyword + ";"+ node.text

                for chemical in articles.iter('ChemicalList'):
                    for node in chemical.iter('NameOfSubstance'):
                        Chemicals = Chemicals + ";"+ node.text

                for mesh in articles.iter('MeshHeadingList'):
                    for node in mesh.iter('DescriptorName'):
                        Meshdescriptor = Meshdescriptor + ";"+ node.text


    #print(firstpmid)
    #print(lastpmid)

    return Journal,Title,Abstract,Keyword,Chemicals,Meshdescriptor


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


def read_metadata(dir):
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


    data = pd.read_csv('../data/training/qrels-treceval-abstracts.2019.txt', sep=" ", header=None)
    
    Journal = []
    Title = []
    Abstract = []
    PMID = []
    Keyword = []
    Chemical = []
    Mesh = []
    
    if os.path.exists('final.csv'):
        df = pd.read_csv('final.csv', sep = '#')
        df.fillna("", inplace=True)
        Journal = df['Journal'].to_list()
        Title = df['Title'].to_list()
        Abstract = df['Abstract'].to_list()
        PMID = df['PMID'].to_list()
        Keyword = df['Keyword'].to_list()
        Chemical = df['Chemical'].to_list()
        Mesh = df['Mesh'].to_list()
    PMIDDICT = set(PMID)

    count = 0
    for row in tqdm(data.iterrows(),total=data.shape[0]):
        count = count + 1
        queryno = row[1][0]
        pmid = row[1][2]
        relevance = row[1][3]

        if pmid in PMIDDICT:
            print("Skipped pmid "+str(pmid))
            continue

    
        pos = bisect_left(firstpmidlist, pmid)
    
        if (pos == len(firstpmidlist)) or (firstpmidlist[pos] != pmid):
            pos = pos - 1

        length = len(str(pos))
        midstring = ""

        for i in range(0,4-length):
            midstring= "0" + midstring
            
        midstring = midstring + str(pos)
        filename = "pubmed19n" + midstring + ".xml"
        filepath = dir + "/"+filename
        Journalentry,Titleentry,Abstractentry,keywordentry,chemicalentry,meshentry=obtain_pmid_from_file(filepath,pmid)


        Journal += [Journalentry]
        Title += [Titleentry]
        Abstract += [Abstractentry]
        PMID += [pmid]
        Keyword += [keywordentry]
        Chemical += [chemicalentry]
        Mesh += [meshentry]

        #if count>2:
        #    break

        df = pd.DataFrame(
        {'PMID' : PMID,
        'Journal': Journal,
        'Title': Title,
        'Abstract': Abstract,
        'Keyword': Keyword,
        'Chemical': Chemical,
        'Mesh': Mesh,
        })
        
        df.to_csv('final.csv', sep = '#')
    #print(filepath)
    #print(firstpmidlist[956])


def store_metadata():
    read_metadata("../data/Baseline Corpus/Baseline Corpus-20220224T050357Z-001/Baseline Corpus")




def get_patient_embedding():
    df =pd.read_csv("Data/Filtered_Topics.csv")
    print(df.head())
if __name__ == "__main__":
    MAX_LEN = 2048
    WINDOW_SIZE = 500
    EMBEDDING_SIZE = 512
    DEV = "cpu:0"
    store_metadata()
