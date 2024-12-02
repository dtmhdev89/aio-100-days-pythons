import torch
from matplotlib import pyplot


def init_inputs(r_min=-2.0, r_max=2.0):
    inputs = torch.arange(r_min, r_max+0.1, 0.1)

    return inputs


def convex(x):
    return x ** 2


def objective_non_convex(x):
    return x**4 + x**3 - 2*x**2 - 2*x + 2


def objective(x):
    return convex(x)


def derivative(x):
    return 2 * x


def derivative_non_convex(x):
    return 4*x**3 + 3*x**2 - 4*x - 2


def sgd(input, epochs, lr):
    solutions, scores = [], []
    solution = input
    solution_eval = objective(solution)
    solutions.append(solution)
    scores.append(solution_eval)

    for epoch in range(epochs):
        gradient = derivative(solution)
        solution = solution - lr * gradient
        solution_eval = objective(solution)
        solutions.append(solution)
        scores.append(solution_eval)
        print('Epoch: %0.2d -- f(%0.3f) = %.5f' % (
            epoch, solution, solution_eval))

    return solutions, scores


def sgd_non_convex(input, epochs, lr):
    solutions, scores = [], []
    solution = input
    solution_eval = objective_non_convex(solution)
    solutions.append(solution)
    scores.append(solution_eval)

    for epoch in range(epochs):
        gradient = derivative_non_convex(solution)
        solution = solution - lr * gradient
        solution_eval = objective_non_convex(solution)
        solutions.append(solution)
        scores.append(solution_eval)
        print('Epoch: %0.2d -- f(%0.3f) = %.5f' % (
            epoch, solution, solution_eval))

    return solutions, scores


def sdg_with_momentum(input, epochs, lr, momentum):
    solutions, scores = [], []
    solution = input
    change = 0.0
    solution_eval = objective_non_convex(solution)
    solutions.append(solution)
    scores.append(solution_eval)

    for epoch in range(epochs):
        gradient = derivative_non_convex(solution)
        new_change = lr * gradient + momentum * change
        solution = solution - new_change
        change = new_change
        solution_eval = objective_non_convex(solution)
        solutions.append(solution)
        scores.append(solution_eval)
        print('Epoch: %0.2d -- f(%0.3f) = %.5f' % (
            epoch, solution, solution_eval))

    return solutions, scores


def nesterov_momentum(input, epochs, lr, momentum):
    solutions, scores = [], []
    solution = input
    change = 0.0
    solution_eval = objective_non_convex(solution)
    solutions.append(solution)
    scores.append(solution_eval)

    for epoch in range(epochs):
        projected = solution + momentum * change
        gradient = derivative_non_convex(projected)
        new_change = momentum * change - lr * gradient
        solution = solution + new_change
        change = new_change
        solution_eval = objective_non_convex(solution)
        solutions.append(solution)
        scores.append(solution_eval)
        print('Epoch: %0.2d -- f(%0.3f) = %.5f' % (
            epoch, solution, solution_eval))

    return solutions, scores


inputs = init_inputs()
result = objective(inputs)
epochs = 20
lr = 1e-1
input = inputs[0]
solutions, scores = sgd(input, epochs, lr)
# pyplot.plot(inputs, result)
# pyplot.plot(solutions, scores, '.-', color='red')
# pyplot.show()

non_convex_result = objective_non_convex(inputs)
solutions, scores = sgd_non_convex(input, epochs, lr)

# pyplot.plot(inputs, non_convex_result)
# pyplot.plot(solutions, scores, '.', color='red')
# pyplot.show()

momentum = 0.8
solutions, scores = sdg_with_momentum(input, epochs, lr, momentum)

pyplot.plot(inputs, non_convex_result)
pyplot.plot(solutions, scores, '.', color='red')
pyplot.show()

momentum = 0.9
solutions, scores = sdg_with_momentum(input, epochs, lr, momentum)

pyplot.plot(inputs, non_convex_result)
pyplot.plot(solutions, scores, '.', color='red')
pyplot.show()

momentum = 0.8
solutions, scores = nesterov_momentum(input, epochs, lr, momentum)

pyplot.plot(inputs, non_convex_result)
pyplot.plot(solutions, scores, '.', color='red')
pyplot.show()
