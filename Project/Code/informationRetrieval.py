from util import *

# Add your import statements here
import numpy as np
import math
import time
import json
from rank_bm25 import BM25Okapi
from nltk.corpus import wordnet

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



	def expand_query(self,query):
		expanded_query = query.copy()
		for term in query:
			synonyms = []
			for syn in wordnet.synsets(term):
				for lemma in syn.lemmas():
					synonyms.append(lemma.name())
			expanded_query.extend(synonyms)
		return expanded_query
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

        # Query Expansion
		# exQ=[]
		# for query in queries:
		# 	query0=[]
		# 	for sentence in query:
		# 		expanded_query = self.expand_query(sentence)
		# 		query0.append(expanded_query)
		# 	exQ.append(query0)
		# queries  = exQ


		#Fill in code here

		docQueryList = self.docs + queries

		docQueryDict = {i: docQueryList[i] for i in range(len(docQueryList))}
		docDict = {i: self.docs[i] for i in range(len(self.docs))}

		#STEP 1: Inverted Index calculation
		start = time.time()
		inv_idx = {}
		for key_idx, (key, doc) in enumerate(docDict.items()):
			for processedSentence in doc:
				for token in processedSentence:
					if token not in inv_idx.keys():
						docIndices = []
						docIndices.append(key)
						inv_idx.setdefault(token, docIndices)
					if key not in inv_idx[token]:
						inv_idx[token].append(key)
		end = time.time()
		print(f"Inverted Index calculation time: {end-start}s")
		with open('inv_idx.json', 'w') as f:
			json.dump(inv_idx, f)
		print(len(inv_idx.keys()))

		#STEP 2: Term Frequency calculation
		start = time.time()
		tf_list = []
		for doc in docQueryDict.values():
			term_freq = {}
			for processedSentence in doc:
				for token in processedSentence:
					if token not in term_freq.keys():
						term_freq.setdefault(token, 0) #Plus 1 smoothing?

					if token in term_freq.keys():
						term_freq[token] += 1
			tf_list.append(term_freq)

		with open('tf1.json', 'w') as f:
			json.dump(tf_list, f)
		print(len(tf_list), len(tf_list[0].keys()))

		for i in range(len(tf_list)):
			for token in inv_idx.keys():
				if token not in tf_list[i].keys():
					tf_list[i].setdefault(token, 0) #Plus 1 smoothing?

		end = time.time()
		print(f"Term Frequency calculation time: {end-start}s")
		with open('tf2.json', 'w') as f:
			json.dump(tf_list, f)
		print(len(tf_list), len(tf_list[0].keys()))

		#STEP 3: IDF and weight matrix calculation time:
		start = time.time()
		idf_dict = {}
		for key_idx, (key, docIndices) in enumerate(inv_idx.items()):
			idf_dict.setdefault(key, math.log2(len(self.docs)/len(docIndices)))

		weight_matrix = np.zeros((len(inv_idx.keys()), len(docQueryList)))

		tf_max = np.zeros(len(docQueryList))
		length = np.zeros(len(docQueryList))
		for i in range(len(docQueryList)):
			if tf_list[i]:
				tf_max[i] = max(list(tf_list[i].values()))
			else:
				tf_max[i] = 1
			length[i] = len(list(tf_list[i].keys()))


		for key_idx, (key, docIndices) in enumerate(inv_idx.items()):
			for i in range(len(docQueryList)):
				weight_matrix[key_idx, i] = (math.log2(tf_list[i][key] + 1)/math.log2(length[i]))*idf_dict[key]
				# weight_matrix[key_idx, i] = tf_list[i][key]*idf_dict[key]
				 #max(tf_list[i].values()))
			# for i in range(len(queries)):
			# 	weight_matrix[key_idx, len(self.docs)+i] = 0.5*(1+tf_list[i][key]/tf_max[i])*idf_dict[key]  #max(tf_list[i].values()))

		end = time.time()
		print(f"IDF and weight matrix calculation time: {end-start}s")
		print(weight_matrix.shape)

		# # STEP 4: LSA
		start = time.time()
		k = 200
		u, s, v = np.linalg.svd(weight_matrix, full_matrices=True)
		S = np.zeros((u.shape[0], v.shape[0]))
		S[:k, :k] = np.diag(s[:k])
		weight_matrix = u@S@v
		end = time.time()
		print(f"LSA calculation time: {end-start}s")
		#STEP 5: Similarity calculation and sorting

		start = time.time()
		cos_sim = np.zeros((len(self.docs), len(queries)))

		# dot_sim = np.zeros((len(self.docs), len(queries)))
		# proj_sim = np.zeros((len(self.docs), len(queries)))

		for i in range(len(self.docs)):
			for j in range(len(queries)):
				cos_sim[i, j] = np.dot(weight_matrix[:, i], weight_matrix[:, len(self.docs)+j])/(np.linalg.norm(weight_matrix[:, i])*np.linalg.norm(weight_matrix[:, len(self.docs)+j]))

				# dot_sim[i, j] = np.dot(weight_matrix[:, i], weight_matrix[:, len(self.docs)+j])
				# proj_sim[i, j] = np.dot(weight_matrix[:, i], weight_matrix[:, len(self.docs)+j])/np.linalg.norm(weight_matrix[:, len(self.docs)+j])


		#STEP 6: BM25
		start = time.time()
		#processing for format of input of BM25
		tokenized_documents = []
		tokenized_queries = []

		for key_idx, (key, doc) in enumerate(docDict.items()):
			d = []
			for processedSentence in doc:
				d+=processedSentence
			tokenized_documents.append(d)

		for doc in (queries):
			d = []
			for processedSentence in doc:
				d+=processedSentence
			tokenized_queries.append(d)

		# Creating a BM25 object with tokenized documents
		bm25 = BM25Okapi(tokenized_documents)

		# Calculate BM25 scores for documents
		bm25_scores=[]
		for j in range(len(queries)):
			bm25_scores.append(bm25.get_scores(tokenized_queries[j]))
		bm25_sim = (np.array(bm25_scores)).T

		# Sorted documents

		sim = np.multiply(bm25_sim,cos_sim)
		sorted_sim = np.argsort(-sim, axis=0)
		doc_IDs_ordered =  [[self.index[sorted_sim[i, j]] for i in range(len(self.docs))] for j in range(len(queries))]

		end = time.time()
		print(f"BM25 calculation time: {end - start}s")
		#
		# np.savetxt("doc_IDs_ordered_BM25_multiply_100_exQ.txt", doc_IDs_ordered)

		return doc_IDs_ordered
