#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Stefano Giagu'
__email__ = 'stefano.giagu@roma1.infn.it'
__version__ = '1.1.0'
__status__ = 'prod'
#
# usage: ./msgDecoder.py -i input -o output
#
# read italian words/id db and text input files convert it in a 
# vector of integers ids and save it as text files
#
import os, sys
import string
import numpy as np

conteggi = {}

# conversion algorithm
def convert( ifile ):
   global conteggi
   convertita = []

   

   filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'

   file = open(ifile,newline='', encoding='utf-8')

   result = file.read()
   result = result.lower()
   for c in filters:
       result = result.replace(c," ")

   result = result.replace("'"," ").replace("-"," ").replace("“"," ").replace("”"," ").replace("â","a")
   result = result.replace("à","a").replace("á","a").replace("à","a")
   result = result.replace("è","e").replace("é","e").replace("è","e").replace("é","e").replace("è","e")
   result = result.replace("ì","i").replace("í","i").replace("ì","i")
   result = result.replace("ò","o").replace("ó","o").replace("ò","o")
   result = result.replace("ù","u").replace("ú","u").replace("ù","u")

   words = word_tokenize(result)
   for i in words:
      skypch = (
       i == " " or i == "," or i == ":" or i == "." or i == "," or i == "(" or i == ")" or i == ";" or
       i == "!" or i == "?" or i == "’" or i == "%" or i == "#" or i == "@" or i == "+" or
       i == "-" or i == "/" or i == "*" or i == "&" or i == "<" or i == ">" or i == "~" or
       i == "`" or i == "'"
      )
      if skypch == False:
        par = i.lower()
        lpar = len(par)
        if lpar < 2:
           lpar = lpar -1

        try: #search word
           idx = parole.index(par)
           tmp_idx = indici[idx]
           trovata = 1
        except:
           trovata = 0

        if trovata == 0: #if failed search word in dictionary w/o last character words
           try:
              idx = paroleX.index(par)
              tmp_idx = indici[idx]
              trovata = 1
           except:
              trovata = 0

        if trovata == 0: #if failed search word w/o last character
           try:
              idx = parole.index(par[0:lpar])
              tmp_idx = indici[idx]
              trovata = 1
           except:
              trovata = 0

        if trovata > 0:
           convertita.append(tmp_idx)
           conteggi[tmp_idx] = conteggi[tmp_idx]+1
           f3.write(i+' id: '+tmp_idx+'\n')
        else:
           tmp_idx = 0
           convertita.append(tmp_idx)
           conteggi[tmp_idx] = conteggi[tmp_idx]+1
           f3.write(i+' NON TROVATA\n')

   outvect = np.asarray(convertita).astype('i4') 
   print(outvect)

   print(conteggi[0], ' : Conteggi 0')

   return

# Import and download Natural Language Toolkt 
import nltk 
nltk.download('punkt')
from nltk.tokenize import word_tokenize 

# Import and Load Italian Words + ID DB
conteggi[0] = 0
parole = []
paroleX = []
indici = []
with open('DB/DB_parole_uniche_IT_withID.dat', encoding='utf-8') as f:
   for line in f:
      fields = line.split()
      parole.append(fields[0])
      xp = fields[0]
      xl = len(fields[0])
      paroleX.append(xp[0:xl-1])
      indici.append(fields[1])
      #print(fields[0], '  ', fields[1])
      conteggi[fields[1]] = 0
f.close()

#print(parole.index('zuzzerellona'))
#print(indici[parole.index('zuzzerellona')])
f3=open('risultatiNEW.dat','w')

convert('corpus_HouseMD_stefano.txt')

f3.close()
f2=open('frequenze.dat','w')

with open('DB/DB_parole_uniche_IT_withID.dat', encoding='utf-8') as f:
   for line in f:
      fields = line.split()
      f2.write(fields[0]+'  '+fields[1]+'  '+str(conteggi[fields[1]])+'\n')
f.close()
f2.close()
