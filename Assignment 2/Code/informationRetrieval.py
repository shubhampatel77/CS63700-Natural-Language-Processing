from util import *

# Add your import statements here
import numpy as np
import math
import time


class InformationRetrieval():

	def __init__(self):
		self.index = None
		self.docs = None #added variable docs to ease access during ranking

	def buildIndex(self, docs, docIDs):
		"""
		Builds the document index in terms of the document
		IDs and stores it in the 'index' class variable

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is
			a document and each sub-sub-list is a sentence of the document
		arg2 : list
			A list of integers denoting IDs of the documents
		Returns
		-------
		None
		"""

		#Fill in code here
		index = {i: docIDs[i] for i in range(len(docs))}

		self.index = index
		self.docs = docs


	def rank(self, queries):
		"""
		Rank the documents according to relevance for each query

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is a query and
			each sub-sub-list is a sentence of the query
		

		Returns
		-------
		list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		"""

		doc_IDs_ordered = []

		#Fill in code here
		docQueryList = self.docs + queries

		docQueryDict = {i: docQueryList[i] for i in range(len(docQueryList))}
		docDict = {i: self.docs[i] for i in range(len(self.docs))}

		#STEP 1: Inverted Index calculation
		start = time.time()
		inv_idx = {}
		for key_idx, (key, doc) in enumerate(docDict.items()):
			for tokenizedSentence in doc:
				for token in tokenizedSentence:
					if token not in inv_idx.keys():
						doc_list = []
						doc_list.append(key)
						inv_idx.setdefault(token, doc_list)
					if key not in inv_idx[token]:
						inv_idx[token].append(key)
		end = time.time()
		print(f"Inverted Index calculation time: {end-start}s")
		
		#STEP 2: Term Frequency calculation
		start = time.time()
		tf_list = []
		for doc in docQueryDict.values():
			term_freq = {}
			for tokenizedSentence in doc:
				for token in tokenizedSentence: 
					if token not in term_freq.keys():
						term_freq.setdefault(token, 0)

					if token in term_freq.keys():
						term_freq[token] += 1
			tf_list.append(term_freq)
		

		for i in range(len(tf_list)):
			for token in inv_idx.keys():
				if token not in tf_list[i].keys():
					tf_list[i].setdefault(token, 0)
		end = time.time()
		print(f"Term Frequency calculation time: {end-start}s")

		#STEP 3: IDF and weight matrix calculation time:
		start = time.time()
		idf_dict = {}
		for key_idx, (key, index) in enumerate(inv_idx.items()):
			idf_dict.setdefault(key, math.log2(len(self.docs)/len(index)))

		weight_matrix = np.zeros((len(inv_idx.keys()), len(docQueryList)))
		for key_idx, (key, index) in enumerate(inv_idx.items()):
			for i in range(len(docQueryList)):
				weight_matrix[key_idx, i] = tf_list[i][key]*idf_dict[key]
		end = time.time()
		print(f"IDF and weight matrix calculation time: {end-start}s")

		np.savetxt('weight_matrix.txt', weight_matrix)

		#STEP 4: Similarity calculation and sorting
		start = time.time()
		cos_sim = np.zeros((len(self.docs), len(queries)))
		# dot_sim = np.zeros((len(self.docs), len(queries)))
		# proj_sim = np.zeros((len(self.docs), len(queries)))
		for i in range(len(self.docs)):
			for j in range(len(queries)):
				cos_sim[i, j] = np.dot(weight_matrix[:, i], weight_matrix[:, len(self.docs)+j])/(np.linalg.norm(weight_matrix[:, i])*np.linalg.norm(weight_matrix[:, len(self.docs)+j]))
				# dot_sim[i, j] = np.dot(weight_matrix[:, i], weight_matrix[:, len(self.docs)+j])
				# proj_sim[i, j] = np.dot(weight_matrix[:, i], weight_matrix[:, len(self.docs)+j])/np.linalg.norm(weight_matrix[:, len(self.docs)+j])
		sorted_sim = np.argsort(-cos_sim, axis=0)
		doc_IDs_ordered = [[self.index[sorted_sim[i, j]] for i in range(sorted_sim.shape[0])] for j in range(sorted_sim.shape[1])]
		end = time.time()
		print(f"Similarity and sorting calculation time: {end-start}s")

		
		return doc_IDs_ordered