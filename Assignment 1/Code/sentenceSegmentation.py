from util import *

# Add your import statements here

import nltk.data


class SentenceSegmentation():

	def naive(self, text):
		"""
		Sentence Segmentation using a Naive Approach

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each string is a single sentence
		"""
		
		segmentedText = []

		#Fill in code here

		sentence = ''
		for character in text:
			sentence += character
			if character  == '.' or character == '?' or character == '!':
				segmentedText.append(sentence)
				sentence = ''

		return segmentedText





	def punkt(self, text):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each strin is a single sentence
		"""

		segmentedText = []

		#Fill in code here

		sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
		segmentedText = sent_detector.tokenize(text.strip())
		
		return segmentedText