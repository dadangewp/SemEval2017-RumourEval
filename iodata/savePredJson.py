'''
description: save prediction to json-file
parameter: DATA_DIR (path to file)
'''
#imports
import json 

#variables
DATA_DIR = "D:\\PhD\\RumourEval\\Small Project on Stance Detection in Rumour\\"


def savePredToFile(tweet,label,filename):
  
    data = dict()
    comment = 0
    support = 0
    deny = 0 
    query = 0
  
    for i in range(tweet.__len__()):
        data.update({tweet[i]:label[i]})
        if (label[i]=="comment"):
            comment += 1
        if (label[i]=="support"):
            support += 1
        if (label[i]=="deny"):
            deny += 1
        if (label[i]=="query"):
            query += 1
    
    print ("label statistics: " + str(comment) + " comments; " + str(support) + " supports; " + str(deny) + " deny; " + str(query) + " query;")
    
    with open(DATA_DIR+filename, 'w') as toFile:
        json.dump(data, toFile)