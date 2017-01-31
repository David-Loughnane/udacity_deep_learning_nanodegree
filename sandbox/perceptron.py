import numpy as numpy


class NeuralNetwork():
	def __init__(self):
		np.random.seed(1)

		# we model a single neuron with 3 input connections and one output. 
		# we assign uniform random weights to a 3x1 matrix with values in the range [-1,1]
		#	with a mean of zero
		self.synaptic_weights = 2 * np.random.random((3,1)) - 1


	# the sigmoid activation function, which describes an s shaped curve
	# converts the to a probability in the range [0,1]
	def __sigmoid(self, x):
		return 1 / (1 + np.exp(-x))


	#gradient of the sigmoid curve
	def __sigmoid_derivative(self, x):
		return x * (1 - x)


	def train(self, training_set_inputs, training_set_outputs, number_of_iterations):
		for i in xrange(number_of_iterations):
			# pass the training set through our NN
			output = self.predict(training_set_inputs)

			# calculate the error
			error = training_set_outputs - output

			# multiply the error by the input and again by the gradient of the sigmoid curve
			adjustment = np.dot(training_set_inputs, error * self.__sigmoid_derivative(output))

			self.synaptic_weights += adjustment


	def predict(self, inputs):
		# pass inputs through our NN
		return self.__sigmoid(np.dot(inputs, self.synaptic_weights))






if __name__ = '__main__':
	
	# initalise a single neuron neural network
	neural_network = NeuralNetwork()

	print('Random starting synaptic weights: ')
	print(neural_network.synaptic_weights)

	#The training set. We have 4 examples with 3 inputs and one output
	training_set_inputs = np.array([[0,0,1], [1,1,1], [1,0,1], [0,1,1]])
	training_set_outputs = np.array([[0,1,1,0]]).T

	# train NN using the training set
	# do it 10,000 times and make small adjustments each time
	neural_network.train(training_set_inputs, training_set_outputs, 10000)

	print('New synaptic weights after training: ')
	print(neural_network.synaptic_weights)

	# test the NN
	print('Predicting: ')
	print(neural_network.predict(np.array([1,1,0])))