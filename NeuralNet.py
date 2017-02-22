import random
import numpy as np

import loaddata



class CreateNet(object):

	def __init__(self, sizes):

		self.depth = len(sizes)
		self.sizes = sizes
		self.count = 0

		self.biases = [ np.random.randn(y,1) for y in sizes[1: ] ]
		self.weights = [ np.random.randn(x,y) for x,y in zip(sizes[1:], sizes[:-1]) ]

	def sigmoid(self, z):
		return 1.0/( 1.0 + np.exp(-z) )

	def sigmoidPrime(self, z):
		return self.sigmoid(z) * ( 1 - self.sigmoid(z) )


	def cost(self, output, label):
		return 0.5 * (output - label) * (output - label)

	def gradientCost(self, output, label):
		return output - label

	def feedForward(self, a):

		for b,w in zip(self.biases, self.weights):

			a = self.sigmoid( np.dot(w, a) + b)


		return a

	def testBatch(self, batch):

		correct = 0

		for a, label in batch:

			probl = self.feedForward(a)

			if( label == 1 and probl >= 0.5 ):
				correct = correct + 1
			elif( label == 0 and probl < 0.5 ):
				correct = correct + 1

		return correct


	def SGD(self, lrate, epochs, batch_size):

		data = loaddata.dataWrapper( str("datasets/Training/dataset.txt") )
		n = len(data)
		score = []

		for j in xrange(epochs):

			np.random.shuffle(data)

			minibatches = [ data[k : k + batch_size] for k in xrange(0 , n, batch_size) ]

			for batch in minibatches:

				score.append( self.testBatch( batch ) )

				self.count = self.count + 1

				if( self.count == 200 ):
					avgscore = ( sum(score) * 100 ) / (self.count * batch_size)
					print " pct average score of last 200 batches ", avgscore

					self.count = 0
					score = []  
 
				self.updateBatch(lrate, batch)

			print "epoch ", (j+1), " finished"
			lrate = lrate - (0.001 * lrate)

		print "training finished"



	def updateBatch(self,lrate, batch):

		nabla_w = [ np.zeros(w.shape) for w in self.weights ]
		nabla_b = [ np.zeros(b.shape) for b in self.biases  ]

		for a, label in batch:
			delta_b, delta_w = self.backprop(a, label)
			nabla_w = [ nw + dw for nw,dw in zip(nabla_w, delta_w) ]
			nabla_b = [ nb + db for nb,db in zip(nabla_b, delta_b) ]

		self.weights = [ w - ( lrate/len(batch) ) * nw 
								for w,nw in zip(self.weights, nabla_w)  ]

		self.biases = [ b - (lrate/len(batch) ) * nb
								for b,nb in zip(self.biases, nabla_b)   ]

		#nabla_w = self.backprop(a, label) # np matrix stores gradient of cost function with respect to weights

		#self.weights = [ w - lrate * nw for w, nw in zip(self.weights, nabla_w) ]

	def backprop(self, a, label):

		activations = []
		zl = []

		nabla_w = [ np.zeros(w.shape) for w in self.weights ]
		nabla_b = [ np.zeros(b.shape) for b in self.biases ]

		#forward pass
		activations = [a]

		for b,w in zip(self.biases, self.weights):

			zl.append( np.dot(w,a) + b)

			a = self.sigmoid( zl[-1] )
			activations.append(a)


		#print "error :" , self.cost(a, label)
		#print a , label

		#backward pass

		delta = np.multiply( self.gradientCost(a, label), self.sigmoidPrime( zl[-1] ) )
		nabla_b[-1] = delta
		nabla_w[-1] = np.dot( delta, activations[-2].transpose() ) 

		for layer in xrange(2, self.depth):

			delta = np.dot(self.weights[-layer + 1].transpose(), delta) * self.sigmoidPrime( zl[-layer] )
			nabla_b[-layer] = delta
			nabla_w[ -layer ] = np.dot(delta, activations[-layer - 1].transpose() )

		return (nabla_b, nabla_w)



net = CreateNet([405,100,1])
net.SGD(0.6, 4, 20)
	

data = loaddata.dataWrapper( str("datasets/Testing/emtest.txt") )

c = net.testBatch(data)

print ( c * 100) / len(data) 

#print len(wrong)

i=1

while ( i != 0):
	word = str(raw_input())
	wordvector = loaddata.vectorize([word])
	tinput = [ np.reshape( wv, (405,1) ) for wv in wordvector ]
	#print tinput
	print net.feedForward(tinput[0]) * 100


















