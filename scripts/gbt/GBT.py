import pandas as pd
import numpy as np
import torch
import os.path
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from transformers import BertTokenizer, BertModel
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')

def obtain_array_from_text(textarray):
    X_train = [tokenizer(text.replace('_',' ').replace(';',' '), padding='max_length', max_length = 300, truncation=True, return_tensors="pt") for text in list(textarray)]
    array= []
    for ele in X_train:
        array += [ele['input_ids'].numpy()[0]]
    return array

def match_arrays(a,b):
    #print(a)
    #print(b)
    dict = {}
    for x in a:
        if x != 0:
            dict[x]=1
    score = 0
    for y in b:
        if y in dict.keys():
            score = score+1
    
    #print(score)
    return score




def create_final_query_vector(output,querylist):
    finalquerys = []
    for row in output[1]:
        queryno = row
        finalquerys += [querylist[queryno]]
    return finalquerys

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
        query += "_"
        for gene in topic.iter('gene'):
            query += gene.text
        query += "_"
        for demographic in topic.iter('demographic'):
            query += demographic.text

        if query == "___":
            print("Empty query")
            continue

        Querys += [query]
    return Querys

queryexpansion = []
with open('queryexpansion.txt') as file:
    for line in file:
        queryexpansion += [line]



output = pd.read_csv('../../data/training/relevancylist.txt', sep=" ", header=None)
query = obtain_querys('../../data/query/query2019.xml')
finalqueryvector = create_final_query_vector(output,query)
df = pd.read_csv("../../data/training/articles.csv",sep = "#")


df['category'] = output[3][0:len(df['Title'])]
df['query'] = finalqueryvector[0:len(df['Title'])]
df['expansion'] = queryexpansion
df.fillna("", inplace=True)

#train_data, validate_data, test_data = np.split(df.sample(frac=1, random_state=42),[int(.8*len(df)), int(.9*len(df))])

train_data = df

y_train = train_data['category'].to_numpy()

for i,x in enumerate(y_train):
    if x == 2:
        y_train[i] = 1

#train_data.drop(labels="Survived", axis=1, inplace=True)

#full_data = train_data.append(test_data)

#drop_columns = ["Name", "Age", "SibSp", "Ticket", "Cabin", "Parch", "Embarked"]
#full_data.drop(labels=drop_columns, axis=1, inplace=True)

#full_data = pd.get_dummies(full_data, columns=["Sex"])
#full_data.fillna(value=0.0, inplace=True)

#X_train = full_data.values[0:891]
#X_test = full_data.values[891:]
#scaler = MinMaxScaler()
#X_train = scaler.fit_transform(X_train)
#X_test = scaler.transform(X_test)





state = 12  
test_size = 0.30  

print(train_data)

Titlearray = obtain_array_from_text(train_data['Title'])
Abstractarray = obtain_array_from_text(train_data['Abstract'])
Queryarray = obtain_array_from_text(train_data['query'])
Mesharray = obtain_array_from_text(train_data['Mesh'])
Chemicalarray = obtain_array_from_text(train_data['Chemical'])
Keywordarray = obtain_array_from_text(train_data['Keyword'])
queryexpansionarray = obtain_array_from_text(train_data['expansion'])


score1list = []
score2list = []
score3list = []
score4list = []
score5list = []
score6list = []
for i,x in enumerate(Titlearray):
    score1 = match_arrays(Titlearray[i],Queryarray[i])
    score2 = match_arrays(Abstractarray[i],Queryarray[i])
    score3 = match_arrays(Keywordarray[i],Queryarray[i])
    score4 = match_arrays(Chemicalarray[i],Queryarray[i])
    score5 = match_arrays(Mesharray[i],Queryarray[i])
    score6 = match_arrays(queryexpansionarray[i],Queryarray[i])
    score1list.append(score1)
    score2list.append(score2)
    score3list.append(score3)
    score4list.append(score4)
    score5list.append(score5)
    score6list.append(score6)

    #X_train += [np.concatenate((Titlearray[i],Abstractarray[i],Queryarray[i]),axis = 0)]

#score1array=np.array(score1list).reshape(-1,1)
X_train = []


for i in range(0,len(score1list)):
    #X_train.append([score1list[i],score2list[i],score3list[i],score4list[i],score5list[i],score6list[i]])
    X_train.append([score4list[i]])
   
#print(X_train)

#for i,x in enumerate(y_train):
#    if y_train[i] == 2: 
#        print("First is")
#        print(i)
#        print(Titlearray[i])
#        print(train_data['Title'].to_numpy()[i])
#        print(Queryarray[i])
#        print(train_data['query'].to_numpy()[i])
        #print(X_train[i])
        #quit()
      

#X_train = np.stack((Titlearray,Abstractarray,Queryarray),axis=1)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,  
    test_size=test_size, random_state=state)

lr_list = [0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1]

for learning_rate in lr_list:
    gb_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=learning_rate, max_features=1, max_depth=5, random_state=0)
    gb_clf.fit(X_train, y_train)


    #importances = gb_clf.feature_importances_
    #std = np.std([tree for tree in gb_clf.estimators_], axis=0)
    #forest_importances = pd.Series(importances, index=["Title","Abstract"])
    #fig, ax = plt.subplots()
    #forest_importances.plot.bar(yerr=1, ax=ax)
    #ax.set_title("Feature importances using MDI")
    #ax.set_ylabel("Mean decrease in impurity")
    #fig.tight_layout()

    result = permutation_importance(
        gb_clf, X_val, y_val, n_repeats=10, random_state=42, n_jobs=2
    )
    #forest_importances = pd.Series(result.importances_mean, index=["Title","Abstract","Keyword","Chemical","Mesh","Expansion",])
    forest_importances = pd.Series(result.importances_mean, index=["Title"])
    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
    ax.set_title("Feature importances using permutation on full model")
    ax.set_ylabel("Mean accuracy decrease")
    fig.tight_layout()
    plt.show()


    plt.show()
    print("Learning rate: ", learning_rate)
    print("Accuracy score (training): {0:.3f}".format(gb_clf.score(X_train, y_train)))
    print("Accuracy score (validation): {0:.3f}".format(gb_clf.score(X_val, y_val)))

gb_clf2 = GradientBoostingClassifier(n_estimators=20, learning_rate=0.5, max_features=1, max_depth=2, random_state=0)
gb_clf2.fit(X_train, y_train)
predictions = gb_clf2.predict(X_val)

print("Confusion Matrix:")
print(confusion_matrix(y_val, predictions))

print("Classification Report")
print(classification_report(y_val, predictions))
