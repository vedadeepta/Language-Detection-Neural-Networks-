import os


import random
import numpy as np 



def constructDatatset(path):

	with open( os.path.join(path, "english.txt") ) as f:
		english = f.readlines()
	f.close()

	with open( os.path.join(path, "mandarin.txt") ) as f:
		mandarin = f.readlines()
	f.close()


	english = [ word.split(",")[0] for word in english]
	mandarin = [ word.split(",")[0] for word in mandarin]


	print "english ", len(english)
	print "mandarin ", len(mandarin)

	f = open( os.path.join(path, "datasetgs.txt"), "w" )

	for word in english:
		f.write(word + ",1\n")
	for word in mandarin:
		f.write(word + ",0\n")
	f.close()




def readfile(path):

	with open(path) as f:
		content = f.readlines()
	f.close()


	words = [ c.split(",")[0] for c in content ]


	labels = [ int( c.split(",")[1][0] )  for c in content ]

	return words, labels

def vectorize(bag):

	wordvector = []

	for word in bag:

		vector = []
		word = word.lower()

		for i in xrange(0, 15):

			if(i < len(word)):
				letter = word[i]
			else:
				letter = None

			for j in xrange(ord('a'), ord('z')+1):

				if(letter and ord(letter) == j):
					vector.append(1)
				else:
					vector.append(0)

			if(letter == None):
				vector.append(1)
			else:
				vector.append(0)
		wordvector.append(vector)

	return wordvector

def dataWrapper(path):


	#english

	words, labels = readfile(path)

	wordvector = vectorize(words)

	tinput = [ np.reshape( wv, (405,1) ) for wv in wordvector ]

	tdata = zip(tinput, labels)

	return tdata


#constructDatatset(str("datasets/Training"))
#dataWrapper(str("datasets/Training/datasetgs.txt"))




