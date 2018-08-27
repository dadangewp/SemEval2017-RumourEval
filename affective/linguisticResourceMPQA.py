# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 15:53:11 2018

@author: dadangewp
"""

import codecs
import re
import os

class MPQA(object):

    mpqaSentiment={}
    mpqaSubjectivity={}

    def __init__(self):
        dirname = os.path.dirname(__file__)
        self.mpqaSentiment = {}
        self.mpqaSubjectivity = {}
        file=codecs.open(dirname + '//resources//mpqa.txt', encoding='UTF-8')
        for line in file:
            line = line.rstrip()
            word = line.split(" ")[2]
            sentiment = line.split(" ")[5]
            subjectivity = line.split(" ")[0]
            self.mpqaSentiment[word] = sentiment
            self.mpqaSubjectivity[word] = subjectivity

        self.pattern_split = re.compile(r"\W+")

        return

    def get_mpqa_sentiment(self,text):

        sentiments=0.0
        words = self.pattern_split.split(text.lower())
        for word in words:
            if word in self.mpqaSentiment:
                if self.mpqaSentiment[word] == "positive":
                    sentiments = sentiments + 1
                elif self.mpqaSentiment[word] == "negative":
                    sentiments = sentiments - 1
                else :
                    sentiments = sentiments + 0
        sentiments = format(sentiments, '.2f')
        sentiments = float(sentiments)
        return sentiments
    
    def get_mpqa_subjectivity(self,text):
        sentiments=0.0
        words = self.pattern_split.split(text.lower())
        for word in words:
            if word in self.mpqaSubjectivity:
                if self.mpqaSubjectivity[word] == "strongsubj":
                    sentiments = sentiments + 2
                elif self.mpqaSubjectivity[word] == "weaksubj":
                    sentiments = sentiments + 1
                else :
                    sentiments = sentiments + 0
        sentiments = format(sentiments, '.2f')
        sentiments = float(sentiments)
        return sentiments


if __name__ == '__main__':
    mpqa = MPQA()
    sentiment = mpqa.get_mpqa_sentiment("abandon abase Abhor")
    print(sentiment)
    subj = mpqa.get_mpqa_subjectivity("abandon abase Abhor")
    print (subj)