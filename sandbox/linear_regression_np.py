import numpy as np


def compute_error_for_line_given_points(b,m,points):
	# initalise error
	totalError = 0
	for i in range(0, len(points)):
		x = points[i, 0]
		y = points[i, 1]
		# get the difference, sqaure it, add to total
		totalError += (y - (m * x + b)) ** 2

	# get the average
	return totalError / float(len(points))


def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
	b = starting_b
	m = starting_m

	for i in range(num_iterations):
		# update b and m with more accurate b and m 
		# 	by performing this gradient step
		b, m = step_gradient(b, m, np.array(points), learning_rate)
	return [b,m]


def step_gradient(b_current, m_current, points, learningRate):
	b_gradient = 0
	m_gradient = 0

	N = float(len(points))

	for i in range(0, len(points)):
		x = points[i, 0]
		y = points[i, 1]
		# direction wrt b and m
		# computing partial derivatives of our error function

		b_gradient += -(2/N) * (y - (m_current * x) + b_current)
		m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))

	# update our b and m values using our partial derivatives
	new_b = b_current - (learningRate * b_gradient)
	new_m = m_current - (learningRate * m_gradient)

	return [new_b, new_m]


def run():
	# collect data
	points = np.genfromtxt('data.csv', delimiter=',')

	# define hyperparameters
	# how fast should our model converge?
	learning_rate = 0.0001
	# y = mx + b
	initial_b = 0
	initial_m = 0
	num_iterations = 1000


	print('starting gradient descent as b = {0}, m = {1}, error = {2}'.format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, points)))
	[b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
	print('ending point at b = {0}, m = {1}, error = {2}'.format(b, m, compute_error_for_line_given_points(b, m, points)))


if __name__ == '__main__':
	run()