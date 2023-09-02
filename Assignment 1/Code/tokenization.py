from util import *

# Add your import statements here
from nltk.tokenize import TreebankWordTokenizer


class Tokenization():

	def naive(self, text):
		"""
		Tokenization using a Naive Approach

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		tokenizedText = []

		#Fill in code here
		
		punctuations = [" ", ".", "?", "!", ",", ";", ":", "'", '"']
		for string in text:
			token_list = []
			token = ''
			for character in string:
				token += character
				if character in punctuations:
					if token[-1] in punctuations:
						token = token[:len(token)-1]
					token_list.append(token)
					token_list.append(character)
					token = ''
			tokenizedText.append(token_list)
		
		return tokenizedText



	def pennTreeBank(self, text):
		"""
		Tokenization using the Penn Tree Bank Tokenizer

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		tokenizedText = []

		#Fill in code here
		for string in text:
			tokenizedText.append(TreebankWordTokenizer().tokenize(string))

		return tokenizedText