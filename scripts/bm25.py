from rank_bm25 import BM25Okapi
import pandas as pd
import os.path
import numpy
from scipy.stats import percentileofscore
import xml.etree.ElementTree as ET
import csv

def ret_pos(ele,list):
    print(ele)
    for i,x in enumerate(list):
        if ele == x:
            print("found")
            return i
    return -1

def plot_graph(arr):
    # pre-sort array
    arr_sorted =  sorted(arr)

    # calculate percentiles using scipy func percentileofscore on each array element
    s = pd.Series(arr)
    percentiles = s.apply(lambda x: percentileofscore(arr_sorted, x))
    df = pd.DataFrame({'data': s, 'percentiles': percentiles})    
    df.sort_values(by='data')

    print(df)


def obtain_querys(dir):
    if not(os.path.exists(dir)):
        print("File "+dir+" does not exist")
        return "","","","","",""


    tree = ET.parse(dir)
    root = tree.getroot()
 
    Querys = []

    for topic in tree.iter('topic'):      
        query  = ""
        for disease in topic.iter('disease'):
            query += disease.text
        query += " "
        for gene in topic.iter('gene'):
            query += gene.text
        query += " "
        for demographic in topic.iter('demographic'):
            query += demographic.text

        if query == "   ":
            print("Empty query")
            continue

        Querys += [query]
    return Querys

output = pd.read_csv('../data/training/relevancylist.txt', sep=" ", header=None)
query = obtain_querys('../data/query/query2019.xml')

datapath = '../data/training/articles.csv'
df = pd.read_csv(datapath,sep="#")
df['category'] = output[3][0:len(df['Title'])]
df.fillna("", inplace=True)
df['combined']=df['Title']+' '+df['Abstract']+' '+df['Keyword']+' '+df['Chemical']+' '+df['Mesh']


i=0
droplist = []
for row in df.iterrows():
    if row[1].Title == "" and row[1].Abstract == "" and row[1].Keyword == "" and row[1].Chemical == "" and row[1].Mesh == "":
        droplist += [i]
    #print('PMID is '+str(df['PMID'][i]))
    #print(row[1].combined)
    #print(df['category'][i])
    i=i+1
    #print("\n")

print(len(df))
df = df.drop(droplist)
print(len(df))
corpus = df['combined']

tokenized_corpus = [doc.split(" ") for doc in corpus]



relevancy2list = []
relevancy1list = []
relevancy0list = []
for i,q in enumerate(query):
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = q.split(" ")
    doc_scores = bm25.get_scores(tokenized_query)
    doc_scores = bm25.get_scores(tokenized_query)

    #print(len(doc_scores))
    #print(len(output))

    #new_list = []
    #pos = 0
    #for score in doc_scores:
        #new_list.append([score,pos])
        #pos = pos + 1
    #new_list= sorted(new_list)

    #training_records = []
    #training_results = []
    for j,record in enumerate(output[0]):
        if (j<len(doc_scores)) and record == i:
            #training_records += [output[2][j]]
            #training_results += [output[3][j]]

    #print(training_records)
    #print(training_results)
            #print(output[3][j])

            if output[3][j] == 2:
                relevancy2list += [doc_scores[j]]
            elif output[3][j] == 1:
                relevancy1list += [doc_scores[j]]
            elif output[3][j] == 0:
                relevancy0list += [doc_scores[j]]
    print(i)
    print("###############")

list2 = pd.DataFrame(relevancy2list)
list1 = pd.DataFrame(relevancy1list)
list0 = pd.DataFrame(relevancy0list)

with open('percentile.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(relevancy2list)
    writer.writerow(relevancy1list)
    writer.writerow(relevancy0list)
   
#print(tokenized_query)
#print(bm25.get_top_n(tokenized_query, corpus, n=1))
# array([0.        , 0.93729472, 0.        ])