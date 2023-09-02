from util import *

# Add your import statements here
import numpy as np



class Evaluation():

	def get_true_docs(self, query_ids, qrels):
		true_doc_IDs = []
		for query_id in query_ids:
			query_true_doc_IDs = []
			for qrel in qrels:
				if query_id == int(qrel["query_num"]):
					query_true_doc_IDs.append(int(qrel["id"]))
			true_doc_IDs.append(query_true_doc_IDs)

		return true_doc_IDs
	
	def get_sorted_relevance(self, query_ids, qrels):
		
		query_qrel_len = np.zeros(len(query_ids))
		for qrel in qrels:
			for query_id in query_ids:
				if int(qrel["query_num"]) == query_id:
					query_qrel_len[int(query_id)-1] += 1
			
		true_relevance = []
		qrels_temp = qrels
		for i in query_qrel_len:
			query_sorted_rel = []
			for qrel in qrels_temp[:int(i)]:
				query_sorted_rel.append(qrel["position"])
			true_relevance.append(-np.sort(-np.array(query_sorted_rel)))
			qrels_temp = qrels_temp[int(i):]
				
		return true_relevance
		

	def queryPrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The precision value as a number between 0 and 1
		"""

		#Fill in code here
		
		RetRelIntersection = 0
		for doc in query_doc_IDs_ordered[:k]:
			if doc in true_doc_IDs[int(query_id)-1]:
				RetRelIntersection += 1
		precision = RetRelIntersection/k

		return precision


	def meanPrecision(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean precision value as a number between 0 and 1
		"""

		#Fill in code here

		true_doc_IDs = self.get_true_docs(query_ids, qrels)
		
		precisions = []
		for query_id in query_ids:
			precisions.append(self.queryPrecision(doc_IDs_ordered[int(query_id)-1], query_id, true_doc_IDs, k))
		meanPrecision = np.mean(precisions)

		return meanPrecision

	
	def queryRecall(self, doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The recall value as a number between 0 and 1
		"""

		#Fill in code here
		
		RetRelIntersection = 0
		for doc in doc_IDs_ordered[:k]:
			if doc in true_doc_IDs[int(query_id)-1]:
				RetRelIntersection += 1
		recall = RetRelIntersection/len(true_doc_IDs[int(query_id)-1])


		return recall


	def meanRecall(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean recall value as a number between 0 and 1
		"""

		#Fill in code here
		true_doc_IDs = self.get_true_docs(query_ids, qrels)
		
		recalls = []
		for query_id in query_ids:
			recalls.append(self.queryRecall(doc_IDs_ordered[int(query_id)-1], query_id, true_doc_IDs, k))

		meanRecall = np.mean(recalls)

		return meanRecall


	def queryFscore(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The fscore value as a number between 0 and 1
		"""

		#Fill in code here

		query_precision = self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
		query_recall = self.queryRecall(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
		if query_precision == 0 and query_recall == 0:
			return 0
		fscore = 2* query_precision*query_recall/(query_precision+query_recall)

		return fscore


	def meanFscore(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value
		
		Returns
		-------
		float
			The mean fscore value as a number between 0 and 1
		"""

		#Fill in code here
		true_doc_IDs = self.get_true_docs(query_ids, qrels)
		
		fscores = []
		for query_id in query_ids:
			fscores.append(self.queryFscore(doc_IDs_ordered[int(query_id)-1], query_id, true_doc_IDs, k))
		meanFscore = np.mean(fscores)

		return meanFscore
	

	def queryNDCG(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The nDCG value as a number between 0 and 1
		"""
		nDCG = 0

		# # #Fill in code here
		# for i in range(1, k+1):
		# 	nDCG += 

		return nDCG


	def meanNDCG(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean nDCG value as a number between 0 and 1
		"""

		meanNDCG = -1

		#Fill in code here

		return meanNDCG


	def queryAveragePrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of average precision of the Information Retrieval System
		at a given value of k for a single query (the average of precision@i
		values for i such that the ith document is truly relevant)

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The average precision value as a number between 0 and 1
		"""

		#Fill in code here

		ret_labels = []
		for doc in query_doc_IDs_ordered[:k]:
			if doc in true_doc_IDs[int(query_id)-1]:
				ret_labels.append(1)
			else:
				ret_labels.append(0)
		precisions = [np.sum(ret_labels[:(j+1)])/(j+1) for j in range(len(ret_labels))]
		precisions = [precision*ret_label for precision, ret_label in zip(precisions, ret_labels)]
		if np.sum(ret_labels) == 0:
			avgPrecision = 0
		else:
			avgPrecision = np.sum(precisions)/np.sum(ret_labels)

		return avgPrecision


	def meanAveragePrecision(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of MAP of the Information Retrieval System
		at given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The MAP value as a number between 0 and 1
		"""

		# meanAveragePrecision = -1

		#Fill in code here
		true_doc_IDs = self.get_true_docs(query_ids, qrels)
		
		avgPrecision = []
		for query_id in query_ids:
			avgPrecision.append(self.queryAveragePrecision(doc_IDs_ordered[int(query_id)-1], query_id, true_doc_IDs, k))

		meanAveragePrecision = np.mean(avgPrecision)

		return meanAveragePrecision

