# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 21:27:49 2017

"""

import codecs
import re
from nltk.stem.porter import PorterStemmer
import os

class LIWCPos(object):

    liwcpos=[]

    def __init__(self):
        dirname = os.path.dirname(__file__)
        self.liwcpos = []
        stemmer = PorterStemmer()
        #http://www2.imm.dtu.dk/pubdb/views/publication_details.php?id=6010
        file=codecs.open(dirname + '//resources//LIWC-POS.txt', encoding='UTF-8')
        for line in file:
            word = line.strip("\r\n")
            word = stemmer.stem(word)
            self.liwcpos.append(word)
        #print(self.liwcpos)    
        self.pattern_split = re.compile(r"\W+")
        return

    def get_liwcpos_sentiment(self,text):
        
        stemmer = PorterStemmer()
        counter=0
        #words = self.pattern_split.split(text.lower())
        words = text.split(" ")
        #print (words)
        #print (self.liwcpos)
        for word in words:
            stemmed = stemmer.stem(word)
            #print(stemmed)
            if stemmed in self.liwcpos:
                counter = counter + 1


        return counter


if __name__ == '__main__':
    liwcpositive = LIWCPos()
    sentiment=liwcpositive.get_liwcpos_sentiment("wonderful opportunity pride beloved")
    print(sentiment)