'''
description: implementations of content-based features
'''
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
import re
import numpy as np
import datetime as dt
from nltk import word_tokenize, pos_tag
from nltk.parse import stanford
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from affective.linguisticResourceDAL import DAL
from affective.lingusticResourceANEW import ANEW
from affective.linguisticResourceAFINN import AFINN
from affective.linguisticResourceLIWCPos import LIWCPos
from affective.linguisticResourceLIWCNeg import LIWCNeg
from affective.linguisticResourceLIWCAffect import LIWCAffect
from affective.linguisticResourceLIWCAssent import LIWCAssent
from affective.linguisticResourceLIWCCause import LIWCCause
from affective.linguisticResourceLIWCCertain import LIWCCertain
from affective.linguisticResourceLIWCCogmech import LIWCCogmech
from affective.linguisticResourceLIWCFuture import LIWCFuture
from affective.linguisticResourceLIWCInhib import LIWCInhib
from affective.linguisticResourceLIWCInsight import LIWCInsight
from affective.linguisticResourceLIWCNegate import LIWCNegate
from affective.linguisticResourceLIWCSad import LIWCSad
from affective.linguisticResourceLIWCYou import LIWCYou
from emotion.emotionEmolex import Emolex
from emotion.emotionEmoSenticNetAnger import ESNAnger
from emotion.emotionEmoSenticNetDisgust import ESNDisgust
from emotion.emotionEmoSenticNetFear import ESNFear
from emotion.emotionEmoSenticNetJoy import ESNJoy
from emotion.emotionEmoSenticNetSad import ESNSad
from emotion.emotionEmoSenticNetSurprise import ESNSurprise

parser = stanford.StanfordParser(model_path="D:\PhD\RumourEval\Small Project on Stance Detection in Rumour\stanford-parser-full-2017-06-09\model\englishPCFG.ser.gz")
#model = gensim.models.Word2Vec.load('brown_model')
dal = DAL()
sid = SentimentIntensityAnalyzer()
anew = ANEW()
afinn = AFINN()
liwcpos = LIWCPos()
liwcneg = LIWCNeg()
liwcaffect = LIWCAffect()
liwcassent = LIWCAssent()
liwccause = LIWCCause()
liwccertain = LIWCCertain()
liwccogmech = LIWCCogmech()
liwcfuture = LIWCFuture()
liwcinhib = LIWCInhib()
liwcinsight = LIWCInsight()
liwcnegate = LIWCNegate()
liwcsad = LIWCSad()
liwcyou = LIWCYou()
emolex = Emolex()
esnanger = ESNAnger()
esndisgust = ESNDisgust()
esnfear = ESNFear()
esnjoy = ESNJoy()
esnsad = ESNSad()
esnsurprise = ESNSurprise()

def textSimToSource(tweetTexts):
      
    jaccard = float(len(tweetTexts[0].intersection(tweetTexts[1]))/float(len(tweetTexts[0].union(tweetTexts[0])))) 
    
    # count word occurrences
    #a_vals = Counter(tweetTexts[0])
    #b_vals = Counter(tweetTexts[1])

    # convert to word-vectors
    #words  = list(a_vals.keys() | b_vals.keys())
    #a_vect = [a_vals.get(word, 0) for word in words]        # [0, 0, 1, 1, 2, 1]
    #b_vect = [b_vals.get(word, 0) for word in words]        # [1, 1, 1, 0, 1, 0]

    # find cosine
    #len_a  = sum(av*av for av in a_vect) ** 0.5             # sqrt(7)
    #len_b  = sum(bv*bv for bv in b_vect) ** 0.5             # sqrt(4)
    #dot    = sum(av*bv for av,bv in zip(a_vect, b_vect))    # 3
    #cosine = dot / (len_a * len_b)  
    return jaccard

def getAffectiveDALActivation(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    pleasantness, activation, imagery,pleasantness_sum, activation_sum, imagery_sum=dal.get_dal_sentiment(cleanedTweet)
    return activation

def getAffectiveDALPleasantness(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    pleasantness, activation, imagery,pleasantness_sum, activation_sum, imagery_sum=dal.get_dal_sentiment(cleanedTweet)
    return pleasantness

def getAffectiveDALImagery(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    pleasantness, activation, imagery,pleasantness_sum, activation_sum, imagery_sum=dal.get_dal_sentiment(cleanedTweet)
    return imagery

def getAffectiveANEWPleasantness(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    pleasantness, arrousal, dominance, pleasantness_sum, arrousal_sum, dominance_sum=anew.get_anew_sentiment(cleanedTweet)
    return pleasantness

def getAffectiveANEWArrousal(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    pleasantness, arrousal, dominance, pleasantness_sum, arrousal_sum, dominance_sum=anew.get_anew_sentiment(cleanedTweet)
    return arrousal

def getAffectiveANEWDominance(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    pleasantness, arrousal, dominance, pleasantness_sum, arrousal_sum, dominance_sum=anew.get_anew_sentiment(cleanedTweet)
    return dominance

def getAffectiveAFINN(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    sentiment=afinn.get_afinn_sentiment(cleanedTweet)
    return sentiment

def getAffectiveLIWCPos(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    sentiment=liwcpos.get_liwcpos_sentiment(cleanedTweet)
    return sentiment

def getAffectiveLIWCNeg(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    sentiment=liwcneg.get_liwcneg_sentiment(cleanedTweet)
    return sentiment

def getAffectiveLIWCAffect(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    sentiment=liwcaffect.get_liwcaffect_sentiment(cleanedTweet)
    return sentiment

def getAffectiveLIWCAssent(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    sentiment=liwcassent.get_liwcassent_sentiment(cleanedTweet)
    return sentiment

def getAffectiveLIWCCause(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    sentiment=liwccause.get_liwccause_sentiment(cleanedTweet)
    return sentiment

def getAffectiveLIWCCertain(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    sentiment=liwccertain.get_liwccertain_sentiment(cleanedTweet)
    return sentiment

def getAffectiveLIWCCogmech(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    sentiment=liwccogmech.get_liwccogmech_sentiment(cleanedTweet)
    return sentiment

def getAffectiveLIWCFuture(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    sentiment=liwcfuture.get_liwcfuture_sentiment(cleanedTweet)
    return sentiment

def getAffectiveLIWCInhib(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    sentiment=liwcinhib.get_liwcinhib_sentiment(cleanedTweet)
    return sentiment

def getAffectiveLIWCInsight(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    sentiment=liwcinsight.get_liwcinsight_sentiment(cleanedTweet)
    return sentiment

def getAffectiveLIWCNegate(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    sentiment=liwcnegate.get_liwcnegate_sentiment(cleanedTweet)
    return sentiment

def getAffectiveLIWCSad(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    sentiment=liwcsad.get_liwcsad_sentiment(cleanedTweet)
    return sentiment

def getAffectiveLIWCYou(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    sentiment=liwcyou.get_liwcyou_sentiment(cleanedTweet)
    return sentiment

def getSurpriseEmotionEmolex(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    cleanedTweet.lower()
    surprise = emolex.get_emotion(cleanedTweet)
    count = 0
    for lst in surprise:
        count = count + lst.count("surprise")
    return count

def getAngerEmotionEmolex(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    cleanedTweet.lower()
    surprise = emolex.get_emotion(cleanedTweet)
    count = 0
    for lst in surprise:
        count = count + lst.count("anger")
    return count

def getAnticipationEmotionEmolex(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    cleanedTweet.lower()
    surprise = emolex.get_emotion(cleanedTweet)
    count = 0
    for lst in surprise:
        count = count + lst.count("anticipation")
    return count

def getDisgustEmotionEmolex(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    cleanedTweet.lower()
    surprise = emolex.get_emotion(cleanedTweet)
    count = 0
    for lst in surprise:
        count = count + lst.count("disgust")
    return count

def getFearEmotionEmolex(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    cleanedTweet.lower()
    surprise = emolex.get_emotion(cleanedTweet)
    count = 0
    for lst in surprise:
        count = count + lst.count("fear")
    return count

def getJoyEmotionEmolex(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    cleanedTweet.lower()
    surprise = emolex.get_emotion(cleanedTweet)
    count = 0
    for lst in surprise:
        count = count + lst.count("joy")
    return count

def getSadnessEmotionEmolex(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    cleanedTweet.lower()
    surprise = emolex.get_emotion(cleanedTweet)
    count = 0
    for lst in surprise:
        count = count + lst.count("sadness")
    return count

def getPositiveEmotionEmolex(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    cleanedTweet.lower()
    surprise = emolex.get_emotion(cleanedTweet)
    count = 0
    for lst in surprise:
        count = count + lst.count("positive")
    return count

def getNegativeEmotionEmolex(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    cleanedTweet.lower()
    surprise = emolex.get_emotion(cleanedTweet)
    count = 0
    for lst in surprise:
        count = count + lst.count("negative")
    return count

def getTrustEmotionEmolex(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    cleanedTweet.lower()
    surprise = emolex.get_emotion(cleanedTweet)
    count = 0
    for lst in surprise:
        count = count + lst.count("trust")
    return count

def getAngerESN(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    cleanedTweet.lower()
    score = esnanger.get_esnanger_sentiment(cleanedTweet)
    return score

def getDisgustESN(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    cleanedTweet.lower()
    score = esndisgust.get_esndisgust_sentiment(cleanedTweet)
    return score

def getFearESN(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    cleanedTweet.lower()
    score = esnfear.get_esnfear_sentiment(cleanedTweet)
    return score

def getJoyESN(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    cleanedTweet.lower()
    score = esnjoy.get_esnjoy_sentiment(cleanedTweet)
    return score

def getSadESN(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    cleanedTweet.lower()
    score = esnsad.get_esnsad_sentiment(cleanedTweet)
    return score

def getSurpriseESN(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    cleanedTweet.lower()
    score = esnsurprise.get_esnsurprise_sentiment(cleanedTweet)
    return score


def polarityContrastLIWC(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    cleanedTweet.lower()
    pos = liwcpos.get_liwcpos_sentiment(cleanedTweet)
    neg = liwcneg.get_liwcneg_sentiment(cleanedTweet)
    if(pos != 0 and neg != 0):
        return 1
    else :
        return 0

def polarityContrastEmolex(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    cleanedTweet.lower()
    pos = getPositiveEmotionEmolex(cleanedTweet)
    neg = getNegativeEmotionEmolex(cleanedTweet)
    if(pos != 0 and neg != 0):
        return 1
    else :
        return 0

def repeatedChar(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    cleanedTweet.lower()
    repeat = re.findall(r'((\w)\3{3,})', cleanedTweet)
    if len(repeat) > 0:
        return 1
    else :
        return 0


def polarityShiftAFINN(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    cleanedTweet.lower()
    score = afinn.getAfinnShift(cleanedTweet)
    return score

def retweetCount(tweet): 
    return tweet["retweet_count"]

def avgWordLength(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    sentenceLength = len(cleanedTweet)
    tweetTokenList = tokenizer(cleanedTweet)
    wordCount = len(tweetTokenList)
    if (sentenceLength > 0 and wordCount > 0):
        avg = float(sentenceLength/wordCount)
        avg = format (avg,'.2f')
        return float(avg)
    else :
        return 0 

def questionMarksCount(tweetText):
    return len(re.findall("\?", tweetText))

def questionMark(tweetText):
    count = len(re.findall("\?", tweetText))
    if (count > 0):
        return 1
    else:
        return 0
    
def exclamationMarks(tweetText):
    count = len(re.findall("\!\!\!", tweetText))
    if count > 0:
        return 1
    else:
        return 0

def textLenght(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",tweetText).split())
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    tweetTokenList = tokenizer(cleanedTweet) 
    return len(tweetTokenList)

def linksCount(tweetText):
    return len(re.findall("http", tweetText))

def linkPresence(tweetText):
    if(len(re.findall("http", tweetText)) > 0):
        return 1
    else:
        return 0
    
def hashTagsCount(tweetText):
    return len(re.findall("#", tweetText))

def hashTagsPresence(tweetText):
    if(len(re.findall("#", tweetText)) > 0):
        return 1
    else:
        return 0

def bowTokenizer(text):
    #nltk.download("stopwords")
    stopTerms = set(stopwords.words("english"))
    tokens = set(text.split()).difference(stopTerms)  
    return tokens

def tokenizer(text):   
    vectorizer = CountVectorizer(min_df=1)
    analyze = vectorizer.build_analyzer()
    return analyze(text)
    
def stemmer(terms):
    
    stemmer = PorterStemmer()
    stemmedTerms = set([]) 
     
    try: 
        for term in terms:
            stemmedTerm = stemmer.stem(term)
            stemmedTerms.add(str(stemmedTerm))
        
        return stemmedTerms
    except:
        return terms
    
def replyTimeToSource(sourceDate,replyDate):
    
    sourceTime = dt.datetime.strptime(sourceDate[:sourceDate.__len__()-11],'%a %b %d %H:%M:%S')
    replyTime = dt.datetime.strptime(replyDate[:replyDate.__len__()-11],'%a %b %d %H:%M:%S')
    timeDelta = replyTime - sourceTime
    replyTime = (timeDelta.days * 24 * 60) + (timeDelta.seconds/60)
    replyTime = format (replyTime,'.2f')
    replyTime = float(replyTime)
    return replyTime

    
def possiblySensitive(value):
    
    if value == "True": # no boolean -> text
        return 1
    else:
        return 0
    
def tweetLevel(sourceId,targetId,hierarchy):
    
    count = 0
    tweethierachy = ""

    if sourceId == targetId:
        return 0
    else:
        for i, t in hierarchy.items():
            if int(i) == sourceId:
                tweethierachy = str(t).replace("u", " ").split()
                break
    
        for x in tweethierachy:
            if (x.find("{") != -1):
                count = count + x.count('{')
            if (x.find("}") != -1):
                count = count - x.count('}')
            if (x.find(str(targetId)) != -1):
                return count
    
    return 0

def sentimentScore(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",tweetText).split())
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    sid = SentimentIntensityAnalyzer()
    ss = sid.polarity_scores(cleanedTweet)
    return ss["compound"]

def addVector():
    vector = np.full((3,1),7)
    return vector
