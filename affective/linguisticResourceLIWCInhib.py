# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 19:53:31 2018

@author: dadangewp
"""

import codecs
import re
from nltk.stem.porter import PorterStemmer
import os

class LIWCInhib(object):

    liwcinhib=[]

    def __init__(self):
        self.liwcinhib = []
        dirname = os.path.dirname(__file__)
        stemmer = PorterStemmer()
        #http://www2.imm.dtu.dk/pubdb/views/publication_details.php?id=6010
        file=codecs.open(dirname + '//resources//LIWC-INHIB.txt', encoding='UTF-8')
        for line in file:
            word = line.strip("\r\n")
            word = stemmer.stem(word)
            self.liwcinhib.append(word)
        #print(self.liwcpos)    
        self.pattern_split = re.compile(r"\W+")
        return

    def get_liwcinhib_sentiment(self,text):
        
        stemmer = PorterStemmer()
        counter=0
        words = self.pattern_split.split(text.lower())
        words = text.split(" ")
        #print (words)
        #print (self.liwcpos)
        for word in words:
            stemmed = stemmer.stem(word)
            #print(stemmed)
            if stemmed in self.liwcinhib:
                counter = counter + 1


        return counter


if __name__ == '__main__':
    liwcinhib = LIWCInhib()
    sentiment=liwcinhib.get_liwcinhib_sentiment("absolutely")
    print(sentiment)