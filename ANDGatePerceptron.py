# Input data for an AND gate
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# Desired output for an AND gate (using -1 for False, 1 for True)
y = np.array([-1, -1, -1, 1])

# Create and train the perceptron
perceptron = Perceptron(learning_rate=0.1, n_iterations=10)
perceptron.fit(X, y)

# Test the trained perceptron
print("Testing the Perceptron on AND gate inputs:")
for i in range(len(X)):
    prediction = perceptron.predict(X[i])
    print(f"Input: {X[i]}, Expected: {y[i]}, Predicted: {prediction}")

# We can also test with new inputs, if desired
# new_input = np.array([1, 0])
# print(f"New input {new_input}: Prediction {perceptron.predict(new_input)}")

output:
Input: [0,0], Expected: -1,Predicted: -1
Input: [0,1], Expected: -1,Predicted: -1
Input: [1,0], Expected: -1,Predicted: -1
Input: [1,1], Expected: 1,Predicted: 1
