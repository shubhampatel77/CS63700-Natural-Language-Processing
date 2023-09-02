from util import *

# Add your import statements here
from nltk.corpus import stopwords


class StopwordRemoval():

	def fromList(self, text):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence with stopwords removed
		"""

		stopwordRemovedText = []
		

		#Fill in code here

		stop_words = set(stopwords.words('english'))
		punctuations = [" ", ".", "?", "!", ",", ";", ":", "'", '"']
		
		for sublist in text:
			new_sublist = []
			for token in sublist:
				if token not in punctuations:
					if token not in stop_words:
						new_sublist.append(token)
			stopwordRemovedText.append(new_sublist)

		return stopwordRemovedText




	