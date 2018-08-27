'''
description: reading ground thruth data
parameter: DATA_DIR (path to dataset)
'''

#imports
import json 

#variables
DATA_DIR = "D:\\PhD\\RumourEval\\Small Project on Stance Detection in Rumour\\"

def readGT(data):
    
    path = DATA_DIR
    if data == "trainA":
        path = path + "semeval2017-task8-dataset/traindev/rumoureval-subtaskA-train.json"
        
    if data == "testA":
        path = path + "gt_a.json"
    
    with open(path) as data_file_GT:    
        json_GT = json.load(data_file_GT)
        data_file_GT.close()
    
    '''
    # for binary classification (deny/notdeny)
    for k, v in json_GT.items():      
        if v == 'support' or v == 'comment' or v == 'query':
            json_GT[k] = unicode('notDeny')           
    '''
        
    return json_GT