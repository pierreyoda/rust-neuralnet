###
### Python script to quickly compute the values and derivative values of a
### numerical function for arbitrary inputs (used in the Rust tests).
###

import math

# common functions
sigmoid = lambda x: 1 / (1 + math.exp(-x))
sigmoid_dx = lambda x: sigmoid(x) * (1 - sigmoid(x))

tanh = lambda x: math.tanh(x)
tanh_dx = lambda x: 1 - tanh(x) ** 2

# TO MODIFY
inputs = [-2.0, -1.0, 0.0, 1.0, 2.0]
inputs = [0.5 * v for v in inputs]
function, function_dx = tanh, tanh_dx

# compute
outputs = []
for input in inputs:
    output = [function(input), function_dx(input)]
    outputs.append(output)
# print to file
digits = 16
with open("functions_outputs.txt", "w") as f:
    for output in outputs:
        value, derivative = round(output[0], digits), round(output[1], digits)
        f.write("{} ||| {}\n".format(value, derivative))
    f.write("\n")
