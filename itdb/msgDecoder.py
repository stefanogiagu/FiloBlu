#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Stefano Giagu'
__email__ = 'stefano.giagu@roma1.infn.it'
__version__ = '1.2.0'
__status__ = 'dev'
#
# usage: ./msgDecoder.py -i input -o output
#
# read italian words/id db and text input files convert it in a 
# vector of integers ids and save it as text files
#
import os, sys
import string
import numpy as np

# command line args
import argparse
parser = argparse.ArgumentParser(description='msgDecoder.py: txt msg to integer ids conversion.')
print (__author__)
print (__version__)
parser.add_argument('-d',action='store_true',default=False, dest='directory',help='if present interpret as dirs the following args', required=False)
parser.add_argument('-i','--input', help='Input file or dir',required=True)
parser.add_argument('-o','--output',help='Output file or dir', required=True)
args = parser.parse_args()

# input and output file names
## show values ##
if args.directory:
   print ("Input dir: %s" % args.input )
   print ("Output dir: %s" % args.output )
else:
   print ("Input file: %s" % args.input )
   print ("Output file: %s" % args.output )
idir=args.input
odir=args.output

# conversion algorithm
def convert( ifile, ofile ):
   convertita = [];

   filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'

   file = open(ifile,newline='', encoding='utf-8')

   result = file.read()
   result = result.lower()
   for c in filters:
       result = result.replace(c," ")

   result = result.replace("'"," ").replace("-"," ").replace("“"," ").replace("”"," ").replace("â","a")
   result = result.replace("à","a").replace("á","a").replace("à","a")
   result = result.replace("è","e").replace("é","e").replace("è","e").replace("é","e").replace("è","e")
   result = result.replace("ì","i").replace("í","i").replace("ì","i")
   result = result.replace("ò","o").replace("ó","o").replace("ò","o")
   result = result.replace("ù","u").replace("ú","u").replace("ù","u")

   words = word_tokenize(result)
   for i in words:
      skypch = (
       i == " " or i == "," or i == ":" or i == "." or i == "," or i == "(" or i == ")" or i == ";" or
       i == "!" or i == "?" or i == "’" or i == "%" or i == "#" or i == "@" or i == "+" or
       i == "-" or i == "/" or i == "*" or i == "&" or i == "<" or i == ">" or i == "~" or
       i == "`" or i == "'"
      )
      if skypch == False:
        trovata = 0
        tmp_idx = -1
        tmp_dis = 999999999999
        for j in parole:
           par0 = j[0].lower()
           lpar0 = len(par0)
           par = i.lower()
           lpar = len(par)
           if par[0:lpar] == par0[0:lpar0]:
              trovata = 1
              tmp_idx = j[1]
              break
           else:
              if lpar0 > 1:
                 lpar0 = lpar0-1
              if lpar > 1:
                 lpar = lpar-1
              if par[0:lpar] == par0[0:lpar0]:
                 trovata = 1
                 if abs(lpar-lpar0) <= tmp_dis:
                    tmp_idx = j[1]
                    tmp_dis = int(tmp_idx)

        if trovata > 0:
           convertita.append(tmp_idx)
           print(i+' id: '+tmp_idx)
        else:
           tmp_idx = 0
           convertita.append(tmp_idx)
           print(i+' NON TROVATA')

   outvect = np.asarray(convertita).astype('i4') 
   print(outvect)

   #save converted vector
   f=open(ofile,'w')
   for ele in convertita:
      f.write(ele+' ')
   f.close()

   return

# Import and download Natural Language Toolkt 
import nltk 
nltk.download('punkt')
from nltk.tokenize import word_tokenize 

# Import and Load Italian Words + ID DB
parole = [];
with open('DB/DB_parole_uniche_IT_withID.dat', encoding='utf-8') as f:
   for line in f:
      fields = line.split()
      parole.append(fields)
f.close()

# Conversion algorithm
if args.directory:
  import os
  for filename in os.listdir(idir):
     print("file: %s" % filename)
     print("file: %s" % os.path.splitext(filename)[0])
     ifile= idir+"/"+filename
     ofile= odir+"/"+os.path.splitext(filename)[0]+".dat"

     convert(ifile, ofile)
else:
  convert(idir,odir)
