import pandas as pd
import re
import math

df = pd.read_csv("../final.csv",sep = "#")
queryexpansiondata = pd.read_csv("../../data/queryexpansion/scrapped.csv",sep = ",")

expansiondict = {}
for i,url in enumerate(queryexpansiondata['web-scraper-start-url']):
    #print(url)
    #print(queryexpansiondata['expansion'][i])
    expansiondict[url] = queryexpansiondata['expansion'][i]
    
    
array = df['Mesh'].to_numpy()
df.fillna("", inplace=True)

list = []
for x in array:
    arr = re.split(';|,| ',x)
    for text in arr:
        if text != '':
            list += [text]
    list += ['#']

dict ={}
final = []
for x in list:
    if x in dict.keys():
        continue
    final.append(x)
    dict[x] = 1

print(len(list))
print(len(final))
print(list)

webstring = ""
for i,x in enumerate(final):
    webstring += "\"https://www.ncbi.nlm.nih.gov/mesh/?term="+x+"\""
    if i !=len(final):
        webstring += ","

f = open("queryexpansion.txt", "w")
tostorestring = ""
for x in list:
    #print(x)
    tmpstring = "https://www.ncbi.nlm.nih.gov/mesh/?term="+x
    if x == '#':
        #print(tostorestring[:100])
        f.write(tostorestring+ "\n")
        tostorestring = ""
    elif tmpstring in expansiondict.keys():
        if isinstance(expansiondict[tmpstring],str):
            #print(expansiondict[tmpstring])
            tostorestring += expansiondict[tmpstring]
            tostorestring += ". "

f.close()
#print(webstring)
