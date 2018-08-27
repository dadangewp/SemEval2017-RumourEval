'''
description: implementation of feature selection, preprocessing and scaling
'''
from features import userBased, contentBased
from features import featureEvaluation
import sklearn.preprocessing as pp
from iodata.saveToFile import saveMatrixToCSVFile
from iodata.saveToFile import saveTweetToCSVFile
import numpy as np
from collections import Counter
from scipy.sparse import hstack, csr_matrix

def featureExtraction(data,gt,hierarchy,isTest): 
    
    featureMatrix = []
    featurePrint = []
    labelMatrix = [] 
    tweetMatrix = []
    combinedMatrix = []
    features = [#"Tweet",
                "Source-Reply-Sim",
                "questMarkCount",
                "questMark",
                "hashTagsPresence",
                "textLengh",
                "LinkCount",
                "Reply-To-Sim",
                "TweetHierarchy",
                #"AFINN_Sentimen",
                #"DAL_Pleasantness"]
                "DAL_Activation",
                #"DAL_Imagery"]
                #"ANEW_Pleasantness",
                #"ANEW_Arrousal"]
                "ANEW_Dominance",
                #"EmolexSurprise",
                #"EmolexAnger",
                #"EmolexTrust"]
                #"EmolexPositive"]
                "EmolexNegative",
                #"EmolexAnticipation"]
                #"EmolexJoy",
                "EmolexFear",
                #"EmolexDisgust",
                #"EmolexSadness",
                #"LIWCAffect",
                "LIWCAssent",
                "LIWCCause",
                "LIWCCertain",
                #"LIWCCogmech"]
                #"LIWCFuture",
                #"LIWCInhib",
                #"LIWCInsight",
                #"LIWCNegate",
                "LIWCSad"]
                #"LIWCYou"]
                #"LIWC_Pos",
                #"LIWC_Neg"]
                #"EmoSNAnger",
                #"EmoSNDisgust",
                #"EmoSNFear",
                #"EmoSNJoy",
                #"EmoSNSad",
                #"EmoSNSurprise"]
    replyToTweetText = "-"
    clients = list()
    #if isTest == 0:
    #    balance = 332
    #else :
    #    balance = 70
    #comment = 0
    #support = 0
    #deny = 0
    #query = 0
    for sourceTweet, tweets in data.items():
            for tweet, tweetContent in tweets.items():
                #save SourceTweet
                for iTweet, iTweetContent in tweets.items():         
                    if (sourceTweet==iTweet):
                        sourceTweetContent = iTweetContent
                        break
                    
                #save ReplyToTweet
                for jTweet, jTweetContent in tweets.items():         
                    if (tweetContent["in_reply_to_status_id"]==jTweet):
                        replyToTweetContent = jTweetContent
                        replyToTweetText = jTweetContent["text"]
                        break    
                
                #tokenized and stemmed
                PPSourceTweetContent = contentBased.stemmer((contentBased.bowTokenizer(sourceTweetContent["text"])))
                PPTweetContent = contentBased.stemmer((contentBased.bowTokenizer(tweetContent["text"])))
                PPTweetContentDeny = contentBased.stemmer((contentBased.bowTokenizer(tweetContent["text"].lower())))
                PPReplyToTweetContent = contentBased.stemmer((contentBased.bowTokenizer(replyToTweetText)))
                
                #creating class labels
                label = ""
                for t, l in gt.items():
                    if int(t) == tweet:
                        label = str(l)
                
                clients.append(tweetContent["source"])                
                
                #tweet console output
                '''
                print PPSourceTweetContent
                print PPTweetContent
                print tweetContent["source"]
                
                if label == "deny":
                    try:
                        print tweetContent["text"]
                    except:
                        print "error"
                        
                if(label=="deny"):
                        try:
                            print str(tweet)+": "+ sourceTweetContent["text"] + "::==::" + tweetContent["text"]
                        except:
                            print "print not possible"
                '''
                        
                #if label != "" and ((label == "comment" and comment <= balance) or (label == "query" and query <= balance) or (label == "support" and support <= balance) or (label == "deny" and deny <= balance)):
                #if label == "query":
                 #   if label == "comment" :
                  #      comment = comment + 1
                  #  elif label == "query" :
                  #      query = query + 1
                  #  elif label == "support":
                  #      support = support + 1
                  #  else :
                  #      deny = deny + 1
                if label != "" :        
                    labelMatrix.append(label)
                    tweetMatrix.append(tweet) 
                    #creating feature vector    
                    featureVector = ([#str(tweetContent["text"]),
                                          contentBased.textSimToSource([PPSourceTweetContent,PPTweetContent]),
                                          contentBased.retweetCount(tweetContent),
                                          contentBased.questionMarksCount(tweetContent["text"]),
                                          contentBased.questionMark(tweetContent["text"]),
                                          contentBased.hashTagsPresence(tweetContent["text"]),
                                          contentBased.textLenght(tweetContent["text"]),
                                          contentBased.linksCount(tweetContent["text"]),
                                          contentBased.textSimToSource([PPReplyToTweetContent,PPTweetContent]),
                                          contentBased.tweetLevel(sourceTweet,tweet,hierarchy),
                                          #contentBased.getAffectiveAFINN(tweetContent["text"]),
                                          #contentBased.getAffectiveDALPleasantness(tweetContent["text"])
                                          contentBased.getAffectiveDALActivation(tweetContent["text"]),
                                          #contentBased.getAffectiveDALImagery(tweetContent["text"])
                                          #contentBased.getAffectiveANEWPleasantness(tweetContent["text"])
                                          #contentBased.getAffectiveANEWArrousal(tweetContent["text"])
                                          contentBased.getAffectiveANEWDominance(tweetContent["text"]),
                                          #contentBased.getSurpriseEmotionEmolex(tweetContent["text"]),
                                          #contentBased.getAngerEmotionEmolex(tweetContent["text"]),
                                          #contentBased.getTrustEmotionEmolex(tweetContent["text"])
                                          #contentBased.getPositiveEmotionEmolex(tweetContent["text"])
                                          contentBased.getNegativeEmotionEmolex(tweetContent["text"]),
                                          #contentBased.getAnticipationEmotionEmolex(tweetContent["text"]),
                                          #contentBased.getJoyEmotionEmolex(tweetContent["text"])
                                          contentBased.getFearEmotionEmolex(tweetContent["text"]),
                                          #contentBased.getDisgustEmotionEmolex(tweetContent["text"]),
                                          #contentBased.getSadnessEmotionEmolex(tweetContent["text"]),
                                          #contentBased.getAffectiveLIWCAffect(tweetContent["text"])
                                          contentBased.getAffectiveLIWCAssent(tweetContent["text"]),
                                          contentBased.getAffectiveLIWCCause(tweetContent["text"]),
                                          contentBased.getAffectiveLIWCCertain(tweetContent["text"]),
                                          #contentBased.getAffectiveLIWCCogmech(tweetContent["text"])
                                          #contentBased.getAffectiveLIWCFuture(tweetContent["text"])
                                          #contentBased.getAffectiveLIWCInhib(tweetContent["text"])
                                          #contentBased.getAffectiveLIWCInsight(tweetContent["text"])
                                          #contentBased.getAffectiveLIWCNegate(tweetContent["text"]),
                                          contentBased.getAffectiveLIWCSad(tweetContent["text"])
                                          #contentBased.getAffectiveLIWCYou(tweetContent["text"])
                                          #contentBased.getAffectiveLIWCPos(tweetContent["text"])
                                          #contentBased.getAffectiveLIWCNeg(tweetContent["text"])
                                          #contentBased.getAngerESN(tweetContent["text"])
                                          #contentBased.getDisgustESN(tweetContent["text"]),
                                          #contentBased.getFearESN(tweetContent["text"]),
                                          #contentBased.getJoyESN(tweetContent["text"])
                                          #contentBased.getSadESN(tweetContent["text"])
                                          #contentBased.getSurpriseESN(tweetContent["text"])
                                          #contentBased.sentimentScore(tweetContent["text"])
                                        ])
                    #array = contentBased.addVector()
                    #for value in array:
                    #    featureVector.append(value)
                    featureMatrix.append(featureVector[:len(features)]) #number of features
                    featurePrint.append(tweetContent["text"])
                    #print (featureVector)
                    #print("\n")
                    featureVector.append(label)
                    featureVector.append(str(tweet))
                    combinedMatrix.append(featureVector)               
 
    #print ("features:")                
    #print (features) 
    
    nrClients = Counter(clients)
                                                
    # standardization (zero mean, variance of one)
    stdScale = pp.StandardScaler().fit(featureMatrix)
    featureMatrixScaled = stdScale.transform(featureMatrix)
    
    #file output
    saveMatrixToCSVFile(featureMatrix,"featureMatrix.csv")
    saveTweetToCSVFile(featurePrint,"tweetTextTrain.txt")
    #saveTweetToCSVFile(labelMatrix,"LabelMatrixTrain.txt")
    #print ("the size is :" + str(len(featurePrint)))
    #saveMatrixToCSVFile(featureMatrixScaled,"featureScaleMatrix.csv")
    #saveMatrixToCSVFile(labelMatrix,"labelMatrix.csv")
    #saveMatrixToCSVFile(combinedMatrix,"featureLabelMatrix.csv")
     
    featureEvaluation.featureClassCoerr(featureMatrix,labelMatrix) 
    #print(features)
                              
    #return pp.normalize(featureMatrixScaled), labelMatrix, tweetMatrix                    
    return featureSelection(featureMatrixScaled), labelMatrix, tweetMatrix, features        

def featureSelection(featureMatrix):
    return featureMatrix