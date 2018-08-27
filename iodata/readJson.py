'''
description: reading json tweets into a python dictionary
parameter: DATA_DIR (path to dataset)
'''

#imports
import json 
import glob
import os

#variables
DATA_DIR = "D:\\PhD\\RumourEval\\Small Project on Stance Detection in Rumour\\"

def readJson(data):
      
    dataPath = DATA_DIR
    listOfEvents = ""
    count = 0
    
    json_tweets = dict()
    tweetHierarchy = dict()
    
    if data == "train":
        dataPath = dataPath + "semeval2017-task8-dataset/rumoureval-data/"
        listOfEvents = os.listdir(dataPath)
        for event in listOfEvents:
            json_tweets.update(readStructure(dataPath+event))
            #print (dataPath+event)
            #print (event)
            tweetHierarchy.update(readTweetHierarchy((dataPath+event)))
            
    if data == "test":
        dataPath = dataPath + "semeval2017-task8-test-data"
        json_tweets = readStructure(dataPath)
        tweetHierarchy = readTweetHierarchy(dataPath)
        
    if data == "debug":
        dataPath = dataPath + "semeval2017-task8-dataset/rumoureval-data/charliehebdo"  
        json_tweets = readStructure(dataPath)    
        tweetHierarchy = readTweetHierarchy(dataPath)
    
    #print str(count) + " tweets read"
    return json_tweets, tweetHierarchy


def readStructure(dataPath):
    
    json_repTweets = dict()
    json_tweets = dict()
    
    #read for all tweets
    listOfTweets = os.listdir(dataPath)
    for SourceTweetId in listOfTweets:
        
        currentPath = dataPath + "/" + SourceTweetId
        
        #source tweet
        with open(currentPath+"/source-tweet/"+SourceTweetId+".json") as data_file_charlieHebdo:    
            json_tweet = json.load(data_file_charlieHebdo)
            #print (json_tweet)
            data_file_charlieHebdo.close()
            SourceTweetId = json_tweet["id"]
            json_repTweets.update({SourceTweetId:json_tweet})
                
        #reply tweets
        for file_name in glob.glob(currentPath+"/replies/*.json"):
            with open(file_name) as data_file_charlieHebdo:    
                json_tweet = json.load(data_file_charlieHebdo)
                data_file_charlieHebdo.close()
                json_repTweets.update({json_tweet["id"]:json_tweet})
                #count = count + 1
    
        json_tweets.update({SourceTweetId:json_repTweets})
        json_repTweets = dict()
    
    return json_tweets

def readTweetHierarchy(dataPath):
    
    tweetHierarchy = dict()
    tweetReplyHierarchy = dict()
    
    #read for all tweets
    listOfTweets = os.listdir(dataPath)
    for SourceTweetId in listOfTweets:
        
        currentPath = dataPath + "/" + SourceTweetId
        
        #source tweet
        with open(currentPath+"/structure.json") as data_file:    
            json_tweetHierarchy = json.load(data_file)
            data_file.close()
            tweetReplyHierarchy.update(json_tweetHierarchy)

    
        tweetHierarchy.update(tweetReplyHierarchy)
        tweetReplyHierarchy = dict()
    
    return tweetHierarchy