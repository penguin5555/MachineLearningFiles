# neural network that outputs the music genre that a person
# would like based on their age

# decision tree or K-mode clustering

# //////////////////////////////
# keys
# //////////////////////////////
# all numbers are the ages
# //////////////////////////////
# 0 is for man
# 1 is for woman
# //////////////////////////////
# 1 is for hip-hop
# 2 is for pop
# 3 is for jazz
# 4 is for k-pop
# 5 is for dance
# 6 is for classical

import json
import random as r
import os
import numpy as np
import math as m

# clear the terminal screen for ease on eyes
os.system("cls")
os.system("cls")

# opening json file
with open("trainingdata.json") as f:
    importedData = json.load(f)
    savedImportedData = dict(importedData)

with open("testingdata.json") as f:
    importedTestingData = json.load(f)

# start of data config


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 *
                                                     (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()

# convert the dictionary from the json file to a dictionary with inputs to the network


def jsonToDictionary(json):
    key = ""
    for i in range(1, 100):
        key = f"{key}{i:02d}"
    converted = []
    for i in range(1, len(json)*2):
        if i/2 == round(i/2):
            converted.append(json[str(key[i] + key[i+1])])
    return converted


def splitTestingData(data, decimalOfListForTesting):
    # make the decimal from 0-1
    r.shuffle(data)
    output = []
    lenData = len(data)
    testingAmount = round((1 - decimalOfListForTesting) * lenData)
    output = data[testingAmount:]
    return output


def sigmoid(val):
    return 1/(1 + np.exp(-val))


def sigmoidInput(list):
    # not doing anything right now bc i dont think it has to
    return list
    temp = []
    for sublist in list:
        tempSubList = []
        for item in sublist:
            tempSubList.append(sigmoid(item))
        temp.append(tempSubList)

    return temp

# separate the answers from the dataset


def seperateAnswers(list):
    global answers
    global inputNeurons
    answers = []
    for sublist in list:
        for i in range(len(sublist)):
            if i == 0:
                answers.append(sublist.pop(2))

# add the values from the dataset to the input layer


def setupInputNeurons(items):
    global inputNeurons
    if type(items) != list:
        quit(code="Input layer is not a list")
    # let each item be a value from 0 to 1
    inputNeurons = []
    for item in items:
        inputNeurons.append(item)
    inputNeurons[0] = inputNeurons[0] * 10

# training data
# importedData = savedImportedData
# importedData = jsonToDictionary(importedData)
# seperateAnswers(importedData)
# trainingData = importedData
# trainingData = sigmoidInput(trainingData)
# trainingDataSelection = rand.randrange(0, len(trainingData))


# testing data
importedTestingData = jsonToDictionary(importedTestingData)
seperateAnswers(importedTestingData)
testingData = splitTestingData(importedTestingData, 0.6)
testingData = sigmoidInput(testingData)
testingDataSelection = r.randrange(0, len(testingData))


# setup the input neurons with a random selection of the imported data


# code below moved down
# setupInputNeurons(trainingData[trainingDataSelection])

# end of data config

# start of layers


class neuron:
    def __init__(self, id, activationsOfPreviousLayer, fedInWeights, fedInBais):
        self.activationsOfPreviousLayer = activationsOfPreviousLayer
        self.fedInWeights = fedInWeights
        self.id = id
        self.innerWeights = fedInWeights
        self.bias = fedInBais

        self.weightedSum = 0
        idx = 0
        # calculate the weighted sum -> w1a1 + w2a2 + ... + wn an
        for item in self.innerWeights:
            self.weightedSum += (item * self.activationsOfPreviousLayer[idx])
            idx += 1
        # calculate the activation from 0 to 1 -> sigmoid(weightedSum + bias)
        self.activation = sigmoid(self.weightedSum + self.bias)

    def getActivation(self):
        return self.activation

    def getId(self):
        return self.id

    def getWeights(self):
        return self.innerWeights

    def getBias(self):
        return self.bias

    def changeWeightAndActivation(self, weightIndex, deltaWeight):
        self.innerWeights[weightIndex] = self.innerWeights[weightIndex] + deltaWeight

        self.weightedSum = 0
        idx = 0
        # calculate the weighted sum -> w1a1 + w2a2 + ... + wn an
        for item in self.innerWeights:
            self.weightedSum += (item * self.activationsOfPreviousLayer[idx])
            idx += 1
        # calculate the activation from 0 to 1 -> sigmoid(weightedSum + bias)
        self.activation = sigmoid(self.weightedSum + self.bias)




class layer:
    def __init__(self, amountOfNeurons, activationsOfPreviousLayer):
        self.activationsOfPreviousLayer = activationsOfPreviousLayer
        self.amountOfNeurons = amountOfNeurons
        self.neurons = []
        # make all the neurons -> all are empty as of now
        for _ in range(self.amountOfNeurons):
            self.neurons.append(0)
        # fill all the neurons up
        for idx in range(len(self.neurons)):
            # set some random numbers for the weights of the neuron
            feedWeights = []
            for _ in self.activationsOfPreviousLayer:
                feedWeights.append(r.uniform(0.0, 1.0))
            # set a random number for the bias
            feedBias = r.randrange(-5, 5)
            self.neurons[idx] = neuron(idx, self.activationsOfPreviousLayer, feedWeights, feedBias)

    def getAmountOfNeurons(self):
        return self.amountOfNeurons

    def getAllWeights(self):
        # get the weights from all of the neurons
        allWeights = [i.getWeights() for i in self.neurons]
        return allWeights

    def getAllBiases(self):
        # get the weights from all of the neurons
        allBiases = [i.getBias() for i in self.neurons]
        return allBiases

    def getAllActivations(self):
        activations = []
        for idx in self.neurons:
            activations.append(idx.getActivation())
        return activations

    def getAllActivationPercentages(self):
        activations = []
        for idx in self.neurons:
            percentage = idx.getActivation()*100
            activations.append(f"{percentage}%")
        return activations

    def getPrediction(self):
        activations = []
        for idx in self.neurons:
            percentage = idx.getActivation()*100
            activations.append(percentage)
        returnList = []
        # [0] is the answer it thinks is right
        returnList.append(activations.index(max(activations))+1)
        # [1] is the percentage value of its decision
        returnList.append(max(activations))
        return returnList

    def printPrediction(self):
        activations = []
        for idx in self.neurons:
            percentage = idx.getActivation()*100
            activations.append(percentage)
        # does same thing as function above, just formats it cool
        return f"Prediction: With {max(activations)}% certainty it is {activations.index(max(activations))+1}"

    def getCost(self):
        # only use for ouput layer
        actualOuput = []
        predictedOuput = []
        predictedOuput = self.getAllActivations()

        for i in range(1 , len(predictedOuput)+1):
            if float(i) == float(correctAnswer):
                actualOuput.append(1.0)
            else:
                actualOuput.append(0.01)
        
        # cross-entropy below
        def crossEntropyLoss(p, y):
            # where p is predicted and y is actual
            sum = 0
            for idx in range(len(p)):
                sum += y[idx]*m.log(p[idx])
            return -sum

        def meanSquaredError(p, y):
            sum = 0
            for idx in range(len(p)):
                sum += np.square(y[idx] - p[idx])
            return sum
        
        return meanSquaredError(predictedOuput, actualOuput)
        #return crossEntropyLoss(predictedOuput, actualOuput)


    def feedForwardAgain(self, amountOfNeurons, activationsOfPreviousLayer, weights, bias):
        self.activationsOfPreviousLayer = activationsOfPreviousLayer
        self.amountOfNeurons = amountOfNeurons
        self.neurons = []
        # make all the neurons -> all are empty as of now
        for _ in range(self.amountOfNeurons):
            self.neurons.append(0)
        # fill all the neurons up
        for idx in range(len(self.neurons)):
            # set some random numbers for the weights of the neuron
            feedWeights = weights[idx]
            #print(weights)
            #print(feedWeights)
            # set a random number for the bias
            feedBias = bias[idx]
            self.neurons[idx] = neuron(idx, self.activationsOfPreviousLayer, feedWeights, feedBias)

    def changeNeuronWeightAndActivation(self, neuronIndex, weightIndex_, deltaWeight_):
        self.neurons[neuronIndex].changeWeightAndActivation(weightIndex_, deltaWeight_)


# end of code for layers


def fit(epochs):
    global correctAnswer
    if epochs < 1:
        quit(code="epoch is less than 1 (not logical to repeat 0 or negative times)")
    # training data
    global importedData
    global trainingData
    global trainingDataSelection
    importedData = []
    importedData = savedImportedData
    importedData = jsonToDictionary(importedData)
    seperateAnswers(importedData)
    trainingData = importedData
    trainingData = sigmoidInput(trainingData)
    trainingDataSelection = r.randrange(0, len(trainingData))
    setupInputNeurons(trainingData[trainingDataSelection])
    # set model
        
    hiddenLayer1 = layer(5, inputNeurons)

    # get the previous layer's activations to feed forward
    hiddenLayer1Activations = hiddenLayer1.getAllActivations()
    hiddenLayer2 = layer(10, hiddenLayer1Activations)

    # get the previous layer's activations to feed forward
    hiddenLayer2Activations = hiddenLayer2.getAllActivations()
    outputLayer = layer(6, hiddenLayer2Activations)

    correctAnswer = str(answers[trainingDataSelection])
    prediction = str(outputLayer.getPrediction()[0])

    print(f"Correct Answer: {correctAnswer}")
    print(f"{outputLayer.printPrediction()}")
    print(f"Cost: {outputLayer.getCost()}") 
    
    #########################################################

    # train hidden layer 1 weights

    printOut = False
    for _ in range(epochs):
        neuronCount = 0
        weightCount = 0

        changeWeightAmount = 0.1
        changeWeight = 0

        # loop amount of neurons in layer
        for i in range(hiddenLayer1.amountOfNeurons):
            weightCount = 0
            # loop amount of neurons in previous layer (loop for the weights)
            for i in range(len(inputNeurons)):
                changeWeight = 0
                # loop for altering the weight by a little
                for i in range(2):
                    if i == 0:
                        changeWeight = float(changeWeightAmount)
                    else:
                        changeWeight = -2 * float(changeWeightAmount)
                    # feed values in to get the cost
                    hiddenLayer1.feedForwardAgain(5, inputNeurons, hiddenLayer1.getAllWeights(), hiddenLayer1.getAllBiases())
                    hiddenLayer1.changeNeuronWeightAndActivation(neuronCount, weightCount, changeWeight)

                    hiddenLayer1Activations = hiddenLayer1.getAllActivations()
                    hiddenLayer2.feedForwardAgain(10, hiddenLayer1Activations, hiddenLayer2.getAllWeights(), hiddenLayer2.getAllBiases())

                    hiddenLayer2Activations = hiddenLayer2.getAllActivations()
                    outputLayer.feedForwardAgain(6, hiddenLayer2Activations, outputLayer.getAllWeights(), outputLayer.getAllBiases())

                    if i == 0:
                        initialCost = outputLayer.getCost()
                    else:
                        finalCost = outputLayer.getCost()

                    if printOut:
                        print(f"Correct Answer: {correctAnswer}")
                        print(f"{outputLayer.printPrediction()}")
                        print(f"Cost: {outputLayer.getCost()}")

                changeWeight = changeWeightAmount
                hiddenLayer1.changeNeuronWeightAndActivation(neuronCount, weightCount, changeWeight)
                if initialCost < finalCost:
                    hiddenLayer1.changeNeuronWeightAndActivation(neuronCount, weightCount, changeWeight)
                else:
                    changeWeight = -1 * changeWeightAmount
                    hiddenLayer1.changeNeuronWeightAndActivation(neuronCount, weightCount, changeWeight)
                weightCount += 1
            neuronCount += 1

        #########################################################
        # train hidden layer 2 weights

        neuronCount = 0
        weightCount = 0

        changeWeightAmount = 0.1
        changeWeight = 0

        # loop amount of neurons in layer
        for i in range(hiddenLayer2.amountOfNeurons):
            weightCount = 0
            # loop amount of neurons in previous layer (loop for the weights)
            for i in range(hiddenLayer1.amountOfNeurons):
                changeWeight = 0
                # loop for altering the weight by a little
                for i in range(2):
                    if i == 0:
                        changeWeight = float(changeWeightAmount)
                    else:
                        changeWeight = -2 * float(changeWeightAmount)
                    # feed values in to get the cost
                    hiddenLayer1.feedForwardAgain(5, inputNeurons, hiddenLayer1.getAllWeights(), hiddenLayer1.getAllBiases())

                    hiddenLayer1Activations = hiddenLayer1.getAllActivations()
                    hiddenLayer2.feedForwardAgain(10, hiddenLayer1Activations, hiddenLayer2.getAllWeights(), hiddenLayer2.getAllBiases())
                    hiddenLayer2.changeNeuronWeightAndActivation(neuronCount, weightCount, changeWeight)

                    hiddenLayer2Activations = hiddenLayer2.getAllActivations()
                    outputLayer.feedForwardAgain(6, hiddenLayer2Activations, outputLayer.getAllWeights(), outputLayer.getAllBiases())

                    if i == 0:
                        initialCost = outputLayer.getCost()
                    else:
                        finalCost = outputLayer.getCost()

                    if printOut:
                        print(f"Correct Answer: {correctAnswer}")
                        print(f"{outputLayer.printPrediction()}")
                        print(f"Cost: {outputLayer.getCost()}")

                changeWeight = changeWeightAmount
                hiddenLayer2.changeNeuronWeightAndActivation(neuronCount, weightCount, changeWeight)
                if initialCost < finalCost:
                    hiddenLayer2.changeNeuronWeightAndActivation(neuronCount, weightCount, changeWeight)
                else:
                    changeWeight = -1 * changeWeightAmount
                    hiddenLayer2.changeNeuronWeightAndActivation(neuronCount, weightCount, changeWeight)
                weightCount += 1
            neuronCount += 1

        #########################################################

        # train ouput layer weights

        neuronCount = 0
        weightCount = 0

        changeWeightAmount = 0.1
        changeWeight = 0

        # loop amount of neurons in layer
        for i in range(outputLayer.amountOfNeurons):
            weightCount = 0
            # loop amount of neurons in previous layer (loop for the weights)
            for i in range(hiddenLayer2.amountOfNeurons):
                changeWeight = 0
                # loop for altering the weight by a little
                for i in range(2):
                    if i == 0:
                        changeWeight = float(changeWeightAmount)
                    else:
                        changeWeight = -2 * float(changeWeightAmount)
                    # feed values in to get the cost
                    hiddenLayer1.feedForwardAgain(5, inputNeurons, hiddenLayer1.getAllWeights(), hiddenLayer1.getAllBiases())

                    hiddenLayer1Activations = hiddenLayer1.getAllActivations()
                    hiddenLayer2.feedForwardAgain(10, hiddenLayer1Activations, hiddenLayer2.getAllWeights(), hiddenLayer2.getAllBiases())

                    hiddenLayer2Activations = hiddenLayer2.getAllActivations()
                    outputLayer.feedForwardAgain(6, hiddenLayer2Activations, outputLayer.getAllWeights(), outputLayer.getAllBiases())
                    outputLayer.changeNeuronWeightAndActivation(neuronCount, weightCount, changeWeight)

                    if i == 0:
                        initialCost = outputLayer.getCost()
                    else:
                        finalCost = outputLayer.getCost()

                    if printOut:
                        print(f"Correct Answer: {correctAnswer}")
                        print(f"{outputLayer.printPrediction()}")
                        print(f"Cost: {outputLayer.getCost()}")

                changeWeight = changeWeightAmount
                outputLayer.changeNeuronWeightAndActivation(neuronCount, weightCount, changeWeight)
                if initialCost < finalCost:
                    outputLayer.changeNeuronWeightAndActivation(neuronCount, weightCount, changeWeight)
                else:
                    changeWeight = -1 * changeWeightAmount
                    outputLayer.changeNeuronWeightAndActivation(neuronCount, weightCount, changeWeight)
                weightCount += 1
            neuronCount += 1
    
    

    print(f"Correct Answer: {correctAnswer}")
    print(f"{outputLayer.printPrediction()}")
    print(f"Cost: {outputLayer.getCost()}")

fit(20)
