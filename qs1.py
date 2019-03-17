import nltk
import numpy as np
import pandas as pd
import random
import sys, math

class Classifier:
	def __init__(self, featureGenerator):
		self.featureGenerator = featureGenerator
		self._C_SIZE = 0
		self._V_SIZE = 0
		self._classes_list = []
		self._classes_dict = {}
		self._vocab = {}

	def setClasses(self, trainingData):
		for(label, line) in trainingData:
			if label not in self._classes_dict.keys():
				self._classes_dict[label] = len(self._classes_list)
				self._classes_list.append(label)
		self._C_SIZE = len(self._classes_list)
		return
		
	def getClasses(self):
		return self._classes_list

	def setVocab(self, trainingData):
		index = 0;
		for (label, line) in trainingData:
			line = self.featureGenerator.getFeatures(line)
			for item in line:
				if(item not in self._vocab.keys()):
					self._vocab[item] = index
					index += 1
		self._V_SIZE = len(self._vocab)
		return

	def getVocab(self):
		return self._vocab

	def train(self, trainingData):
		pass

	def classify(self, testData, params):
		pass

	def getFeatures(self, data):
		return self.featureGenerator.getFeatures(data)
		

class FeatureGenerator:
	def getFeatures(self, text):
		text = text.lower()
		return text.split()


class NaiveBayesClassifier(Classifier):
	def __init__(self, fg, alpha = 0.05):
		Classifier.__init__(self, fg)
		self.__classParams = []
		self.__params = [[]]
		self.__alpha = alpha

	def getParameters(self):
		return (self.__classParams, self.__params)

	def train(self, trainingData):
		self.setClasses(trainingData)
		self.setVocab(trainingData)
		self.initParameters()

		for (cat, document) in trainingData:
			for feature in self.getFeatures(document):
				self.countFeature(feature, self._classes_dict[cat])

	def countFeature(self, feature, class_index):
		counts = 1
		self._counts_in_class[class_index][self._vocab[feature]] = self._counts_in_class[class_index][self._vocab[feature]] + counts
		self._total_counts[class_index] = self._total_counts[class_index] + counts
		self._norm = self._norm + counts

	def classify(self, testData):
		post_prob = self.getPosteriorProbabilities(testData)
		return self._classes_list[self.getMaxIndex(post_prob)]

	def getPosteriorProbabilities(self, testData):
		post_prob = np.zeros(self._C_SIZE)
		for i in range(0, self._C_SIZE):
			for feature in self.getFeatures(testData):
				post_prob[i] += self.getLogProbability(feature, i)
			post_prob[i] += self.getClassLogProbability(i)
		return post_prob

	def getFeatures(self, testData):
		return self.featureGenerator.getFeatures(testData)

	def initParameters(self):
		self._total_counts = np.zeros(self._C_SIZE)
		self._counts_in_class = np.zeros((self._C_SIZE, self._V_SIZE))
		self._norm = 0.0

	def getLogProbability(self, feature, class_index):
		return math.log(self.smooth(self.getCount(feature, class_index),self._total_counts[class_index]))

	def getCount(self, feature, class_index):
		if feature not in self._vocab.keys():
			return 0
		else:
			return self._counts_in_class[class_index][self._vocab[feature]]

	def smooth(self, numerator, denominator):
		return (numerator + self.__alpha) / (denominator + (self.__alpha * len(self._vocab)))

	def getClassLogProbability(self, class_index):
		return math.log(self._total_counts[class_index]/self._norm)

	def getMaxIndex(self, posteriorProbabilities):
		maxi = 0
		maxProb = posteriorProbabilities[maxi]
		for i in range(0, self._C_SIZE):
			if(posteriorProbabilities[i] >= maxProb):
				maxProb = posteriorProbabilities[i]
				maxi = i
		return maxi


class Dataset:
	def __init__(self, filename):
		fp = open(filename, "r")
		i = 0
		self.__dataset = []
		for line in fp:
			if(line != "\n"):
				line = line.split()
				cat = line[0]
				sent = ""
				for word in range(1, len(line)):
					sent = sent+line[word]+" "
				sent = sent.strip()
				self.__dataset.append([cat, str(sent)])
				i = i+1
		random.shuffle(self.__dataset)	
		self.__D_SIZE = i
		self.__trainSIZE = int(0.6*self.__D_SIZE)
		self.__testSIZE = int(0.3*self.__D_SIZE)
		self.__devSIZE = 1 - (self.__trainSIZE + self.__testSIZE)

	def setTrainSize(self, value):
		self.__trainSIZE = int(value*0.01*self.__D_SIZE)
		return self.__trainSIZE

	def setTestSize(self, value):
		self.__testSIZE = int(value*0.01*self.__D_SIZE)
		return self.__testSIZE

	def setDevelopmentSize(self):
		self.__devSIZE = int(1 - (self.__trainSIZE + self.__testSIZE))
		return self.__devSIZE

	def getDataSize(self):
		return self.__D_SIZE
	
	def getTrainingData(self):
		return self.__dataset[0:self.__trainSIZE]

	def getTestData(self):
		return self.__dataset[self.__trainSIZE:(self.__trainSIZE+self.__testSIZE)]

	def getDevData(self):
		return self.__dataset[0:self.__devSIZE]



#============================================================================================

if __name__ == "__main__":
	
	# This Naive Bayes classifier implementation 10% better accuracy than the NLTK 3.0 Naive Bayes classifier implementation
	# at the task of classifying questions in the question corpus distributed with the book "Taming Text".

	# The "questions-train.txt" file can be found in the source code distributed with the book at https://www.manning.com/books/taming-text.
	#ref : https://aiaioo.wordpress.com/tag/question-classification/	
	#https://www.geeksforgeeks.org/naive-bayes-classifiers/
	# To the best of our knowledge, the improvement in accuracy is owed to the smoothing methods described in our blog:
	# https://aiaioo.wordpress.com/2016/01/29/in-a-naive-bayes-classifier-why-bother-with-smoothing-when-we-have-unknown-words-in-the-test-set/
	



	###WH-word POS_Tag detector
	filename = "test.txt"
	df = pd.DataFrame(columns = ['Question_Classifier'])
	#df = pd.DataFrame(columns = ['Question_Classifier','wh_bi_gram','wh_nbor_pos', 'root_token'])
	#print(df.head())
	texts = open("test.txt" ,'r')
	for text in texts:
		sentences = nltk.sent_tokenize(text)
		for sentence in sentences:
			words = nltk.word_tokenize(sentence)
			tagged_words = nltk.pos_tag(words)
			ne_tagged_words = nltk.ne_chunk(tagged_words)
			#print(tagged_words)
			#print([i[1] for i in tagged_words],'\n')
			for x in [i[0] for i in tagged_words if i[1]=="WP" or i[1]=="WRB"]:
			    #WP for (Who,What) & WRB for (When,Where)
			    #print(x)
			    #row = [x,words[1],]
			    df.loc[len(df)] = x
	print(df.head())
	###



	if len(sys.argv) > 1:
		filename = sys.argv[1]
	
	data = Dataset(filename)
	
	data.setTrainSize(50)
	data.setTestSize(50)
	
	train_set = data.getTrainingData()
	test_set = data.getTestData()
	
	test_data = [test_set[i][1] for i in range(len(test_set))]
	actual_labels = [test_set[i][0] for i in range(len(test_set))]
	
	fg = FeatureGenerator()
	alpha = 0.6 #smoothing parameter

	nbClassifier = NaiveBayesClassifier(fg, alpha)
	nbClassifier.train(train_set)
	
	correct = 0;
	total = 0;
	for line in test_data:
		best_label = nbClassifier.classify(line)
		if best_label == actual_labels[total]:
			correct += 1	
		total += 1
	
	acc = 1.0*correct/total
	print("Accuracy of this Naive Bayes Classifier: "+str(acc))






