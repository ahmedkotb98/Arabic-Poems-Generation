#!/usr/bin/env python
# -*-coding: utf-8 -*-
from __future__ import print_function

import os
import re
import argparse

from tqdm import tqdm


FARSI_ORIGINAL_CHARACHTERS = "٠١٢۲٣۳٤٥Ƽ٦٧۷٨۸٩۹ﺀٴٱﺂﭑﺎﺈﺄιٲﺍٳίﺃٵﺇﺁﺑپﮨﺒٻﺐﺏﭘﭒﭗﭖﭚٮﭛٺﺗﭠﺘټﺖﺕﭡٹﭞٿﭟﭤﮢﭥﭨﭢﭣﮣﭧﺛﺜﮆﺚﺙٽﮇچﺟﭴﺠﭼڄڇﭸﺝڃﺞﭽﮀﭵﭹﭻﭾﭿﭺﺣﺤﺡﺢځﺧﺨڅڂﺦﺥڿډﺩڍﺪڊڈﮃﮂڋﮈڌﮉڐﮄﺫﺬڎڏۮڕړﺮﺭڒڔږڑژﮌڗﮍڙﺯﺰﮊﺳڛﺴﺲﺱښﺷڜﺸﺶﺵۺﺻﺼڝﺺﺹﺿﻀﺽڞﺾۻﻃﻁﻄﻂﻈﻇﻅڟﻆﻋ۶ﻌﻊﻉﻏﻐڠۼﻍﻎﻓڤﻔﭬڣﭰﻒﻑڦڢڡﭫڥﭪﭭﭯﭮﻗﻘڨﻖﻕڧﭱگڳکڪڱﮔﻛﮘڰﮐﮖﻜﮜڲﻚڴﮗڭﻙﮓﮙګڮﮕﮛڬﮎﮝﮚﮑﮒﮏﯖﯕﻟڵڷﻠڶﻞﻝڸﻣﻤﻢﻡﻧﻥڼﻨﻦڻڽﮠڹﮞںטּﮡﮟھہۃﮬﮪﮧۂﻫﮫﺔﻪﻬﮭﺓۿﻩەۀﮤﮥﮦۆۈۅﯙۉﻭﻮۄۋۇۊﯚٷٶﯛﯠﺆﯜۏﺅﯡﯝﯘﯢﯞﯣﯗﯟﯾےﻳۓېێﮱﻴﮯﭔﻲۑۍﯿﻱﻰﭜڀﺋﻯﭕﮮﺌﭓﯼﭝ༦ﺊﯽﮰﭙﯥﺉﯦﯧﯤیٸ"
ARABIC_NORMALIZED_CHARACHTERS = "0122334556778899ءءاااااااااااااااببببببببببببببتتتتتتتتتتتتتتتتتتتتثثثثثثثجججججججججججججججججججحححححخخخخخخخددددددددددددددذذذذذرررررررررررررزززسسسسسسششششششصصصصصضضضضضضططططظظظظظعععععغغغغغغفففففففففففففففففقققققققككككككككككككككككككككككككككككككككككللللللللممممننننننننننننننهههههههههههههههههههههوووووووووووووووووووووووووووويييييييييييييييييييييييييييييييييييييي"

farsi_to_arabic = str.maketrans(FARSI_ORIGINAL_CHARACHTERS, ARABIC_NORMALIZED_CHARACHTERS)


def replace_Farsi_characters(s):
	s = s.replace("ﻻ","لا")
	s = s.replace("ﻵ","لآ")
	s = s.replace("ﻷ","لأ")
	s = s.replace("ע","لا")
	s = s.replace("ﻹ","لإ")
	s = s.replace("ﻼ","لا")
	s = s.replace("ﻶ","لآ")
	s = s.replace("ﻸ","لأ")
	s = s.replace("ﬠ","لا")

	s = s.translate(farsi_to_arabic)
	return s


# In[ ]:

def normalize_text(t):
	s = t

	s = s.replace("أ", "ا")
	s = s.replace("إ", "ا")
	s = s.replace("آ", "ا")
	#s = s.replace("ى", "ي")
	#s = s.replace("ة", "ه")

	# diac
	s = s.replace("َ", "")
	s = s.replace("ُ", "")
	s = s.replace("ِ", "")
	s = s.replace("ّ", "")
	s = s.replace("ْ", "")
	s = s.replace("ٌ", "")
	s = s.replace("ً", "")
	s = s.replace("ٍ", "")

	s = s.replace("ـ", "")

	s = replace_Farsi_characters(s)
	s =  re.sub(r'[ًٌٍَُِّْـ]+',r'',s)
	return s

def clean_ara(txt):
	s = normalize_text(txt)
	s = re.sub(r':',r'\n',s)
	s = re.sub(r'\r',r'\n',s)
	s = re.sub(r'[^ء-ي٠-٩0-9\.]+',' ',s)
	s = re.sub(r'\. +',r'.\n',s)
	s = re.sub(r'([ء-ي]+)',r' \1 ',s)
	s = re.sub(r'\ +',' ',s)
	lines = s.split('\n')
	lines1 = []
	for l in lines:
		if(len(l.split())>80):
			words = s.split()
			for i in range(int(len(words)/80)+1):
				words.insert((i+1)*80,'\n')
			l = ' '.join(words)
		lines1.append(l)
	s = '\n'.join(lines1)
	s = re.sub(r'\n\n+',r'\n',s)
	return s


def main(args):
	input_file = open(args.input_file, "r")
	output_path = os.path.join(args.output_dir, args.input_file + '.cleaned')
	output_file = open(output_path, "w")
	lines = input_file.read().split('\n')
	for line in tqdm(lines):
		line = line.strip()
		line = clean_ara(line)
		for l in line.split('\n'):
			l = re.sub(r'^[0-9 ]*\ ','',l)
			nl = re.sub('[0-9 ]+',' ',l)
			if len(nl) > 1 and len(nl.split()) > 4:
				output_file.write(l + '\n')

	output_file.close()
	input_file.close()


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="clean dataset")
	parser.add_argument(
		"-i", "--input_file", help="input data file path", required=True, type=str
	)
	parser.add_argument(
		"-s",
		"--output_dir",
		help="Path to save the cleaned dataset",
		required=True,
		type=str,
	)
	args = parser.parse_args()
	main(args)
