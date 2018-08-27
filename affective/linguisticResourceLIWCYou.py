# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 22:51:26 2018

@author: dadangewp
"""

import codecs
import re
from nltk.stem.porter import PorterStemmer
import os

class LIWCYou(object):

    liwcyou=[]

    def __init__(self):
        self.liwcyou = []
        dirname = os.path.dirname(__file__)
        stemmer = PorterStemmer()
        #http://www2.imm.dtu.dk/pubdb/views/publication_details.php?id=6010
        file=codecs.open(dirname + '//resources//LIWC-YOU.txt', encoding='UTF-8')
        for line in file:
            word = line.strip("\r\n")
            word = stemmer.stem(word)
            self.liwcyou.append(word)
        #print(self.liwcpos)    
        self.pattern_split = re.compile(r"\W+")
        return

    def get_liwcyou_sentiment(self,text):
        
        stemmer = PorterStemmer()
        counter=0
        words = self.pattern_split.split(text.lower())
        words = text.split(" ")
        #print (words)
        #print (self.liwcpos)
        for word in words:
            stemmed = stemmer.stem(word)
            #print(stemmed)
            if stemmed in self.liwcyou:
                counter = counter + 1


        return counter


if __name__ == '__main__':
    liwcyou = LIWCYou()
    sentiment=liwcyou.get_liwcyou_sentiment("absolutely")
    print(sentiment)