from util import *

# Add your import statements here

from nltk.stem import WordNetLemmatizer


class InflectionReduction:

	def reduce(self, text):
		"""
		Stemming/Lemmatization

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of
			stemmed/lemmatized tokens representing a sentence
		"""

		reducedText = []
		
		punctuations = [" ", ".", "?", "!", ",", ";", ":", "'", '"']
		#Fill in code here
		lemmatizer = WordNetLemmatizer()
		for sublist in text:
			lemma_sublist = []
			for token in sublist:
				if token not in punctuations:
					lemma_sublist.append(lemmatizer.lemmatize(token))
			reducedText.append(lemma_sublist)
		
		return reducedText


