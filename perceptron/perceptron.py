import pandas as pd
import matplotlib.pyplot as plt
import random

from sklearn.datasets import load_breast_cancer, load_diabetes

class Perceptron:
    def __init__(self, inputs, rate):
        #empty input vector
        self.inputs = [0 for i in range(inputs)]
        #randomise weights and bias from -1 to 1
        self.weights = [random.uniform(-1,1) for i in range(inputs)]
        self.bias = random.uniform(-1,1)
        #learning rate and epoch number
        self.learn_rate = rate
        self.epoch_no = 0

    def activation(self,val):
        #activation function
        #if value above 0 -> return 1 else 0
        if val >= 0:
            return 1
        else:
            return 0

    def predict(self, input):
        #predicted value = summation of the product of input and weight + bias
        predicted = 0
        for i in range(len(input)):
            predicted += input[i] * self.weights[i] + self.bias
        
        #return 1 or 0
        logistic_prediction = self.activation(predicted)

        return logistic_prediction

    def train(self, x, truth):
        #predict label from input
        predicted = self.predict(x)
        #update all weights
        for i in range(len(self.weights)):
            #w(t+1) = w(t) + r(d-y(t))x
            self.weights[i] = self.weights[i] + self.learn_rate*(truth - predicted)*x[i]
        #update bias with same equation
        self.bias = self.bias + self.learn_rate*(truth - predicted)
        #adaptive learning rate -> reduces as epochs continue
        self.learn_rate = self.learn_rate/(self.epoch_no+1)


class PerceptronRegression(Perceptron):
    def activation(self, val):
        return val

def load_cancer_binary_class():
    #loads in sklearn binary classification dataset
    #569 samples, 30 dimensions
    #target values = 1 (Malignant) or 0 (Benign)
    data = load_breast_cancer()

    y = data.target
    x = data.data

    return x, y

def load_diabetes_regression():
    #loads in diabetes data
    #442 samples, 11 dimensions
    # target values = quantitative measure of disease progression after 1 year
    data = load_diabetes()

    y = data.target
    x = data.data

    return x, y

if __name__ == '__main__':
    x, y = load_cancer_binary_class() #binary classification data
    #x, y = load_diabetes_regression() #regression data
    no_vars = len(x[0]) 

    clf = Perceptron(no_vars, rate=0.1) #uncomment for binary classification
    #clf = PerceptronRegression(no_vars, rate = 0.01) #uncomment for regression perceptron

    trainhist = {}

    sample_size = len(x)
    error = 1 #error always starts at 1??
    for n in range(1,100):
        #100 epochs
        correct_predictions = 0

        #stop training based on overall error
        overall_error = 1/n * error #can't work out error calc

        for i in range(len(x)):
            data = x[i]
            target = y[i]

            #predict value
            pred = clf.predict(data)
            if pred == target:
                correct_predictions += 1

            #error calc
            error += (pred - target)

            #train model
            clf.train(data, target)
           
        print(f'Epoch: {n}, Accuracy: {correct_predictions/sample_size}')
        trainhist[n] = correct_predictions/sample_size

    plt.plot(trainhist.keys(), trainhist.values())
    plt.show()

    