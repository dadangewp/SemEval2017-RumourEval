# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 19:31:22 2018

@author: dadangewp
"""

import codecs
import re
from nltk.stem.porter import PorterStemmer
import os

class LIWCAffect(object):

    liwcaffect=[]

    def __init__(self):
        self.liwcaffect = []
        dirname = os.path.dirname(__file__)
        stemmer = PorterStemmer()
        #http://www2.imm.dtu.dk/pubdb/views/publication_details.php?id=6010
        file=codecs.open(dirname + '//resources//LIWC-AFFECT.txt', encoding='UTF-8')
        for line in file:
            word = line.strip("\r\n")
            word = stemmer.stem(word)
            self.liwcaffect.append(word)
        #print(self.liwcpos)    
        self.pattern_split = re.compile(r"\W+")
        return

    def get_liwcaffect_sentiment(self,text):
        
        stemmer = PorterStemmer()
        counter=0
        words = self.pattern_split.split(text.lower())
        words = text.split(" ")
        #print (words)
        #print (self.liwcpos)
        for word in words:
            stemmed = stemmer.stem(word)
            #print(stemmed)
            if stemmed in self.liwcaffect:
                counter = counter + 1


        return counter


if __name__ == '__main__':
    liwcaffect = LIWCAffect()
    sentiment=liwcaffect.get_liwcaffect_sentiment("bad")
    print(sentiment)