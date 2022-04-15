from scipy.stats import percentileofscore
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv


def plot_graph(arr,color, relevancy):
    # pre-sort array
    arr_sorted =  sorted(arr)

    # calculate percentiles using scipy func percentileofscore on each array element
    s = pd.Series(arr)
    percentiles = s.apply(lambda x: percentileofscore(arr_sorted, x))
    df = pd.DataFrame({'data': s, 'percentiles': percentiles})    

    df = df.drop_duplicates(subset = ["percentiles"])
    df = df.sort_values(by='data')
    plt.plot(df['percentiles'], df['data'],color=color,label = relevancy)
    plt.title("BM25 scores by percentile grouped by relevancy")
    plt.legend(loc="upper left")
    plt.ylabel("BM25 score")
    plt.xlabel("Percentile")
    #print(df.to_string())

with open("percentile.csv", "r") as f:
    reader = csv.reader(f, delimiter=",")
    no = 0
    for line in reader:
        no = no+1
        strarr = list(line)
        arr = []
        for x in strarr:
            arr += [float(x)]
        
        if no == 1:
            list1 = arr
        elif no ==2:
            list2 = arr
        elif no == 3:
            list3 = arr
list12 = list1 +list2
print(len(list1))
print(len(list2))
print(len(list12))
print(len(list3))
plot_graph(list12,'r',"relevant")
#plot_graph(list2,'g')
plot_graph(list3,'b',"not relevant")
plt.show()


