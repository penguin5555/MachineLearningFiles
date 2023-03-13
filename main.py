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

# imports                         $ &
import pandas as pd
import re
import random as r
import os
import numpy as np
import matplotlib.pyplot as plt
import time as t
global answerIdx
answerIdx = 100

# clear the terminal screen for ease on eyes
os.system("cls")
os.system("cls")

# initialize data
data = pd.read_csv("five.csv")
trainData = []
testData = []

# make csv into an array
temp = []
for rowNumber in range(len(data)):
    row = []
    for item in range(len(data.columns)):
        row.append(data.iloc[rowNumber, item])
    temp.append(row)
data = list(temp)

# shuffle data and put it into train and test data
for i in range(20):
    r.shuffle(data)
for i in range(len(data)):
    if r.randint(0, 1) == 0:
        testData.append(data[i])
    else:
        trainData.append(data[i])
# make the answers into numbers
idx = 0
for i in trainData:
    idx2 = 0
    for ii in i:
        if ii == 'Five':
            trainData[idx][idx2] = 1
        elif ii == 'NotFive':
            trainData[idx][idx2] = 0
        idx2 += 1
    idx += 1
idx = 0
for i in testData:
    idx2 = 0
    for ii in i:
        if ii == 'Five':
            testData[idx][idx2] = 1
        elif ii == 'NotFive':
            testData[idx][idx2] = 0
        idx2 += 1
    idx += 1
savedTrainData = list(trainData)
savedTestData = list(testData)


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


def sigmoid(val):
    # ReLu
    # if val > 0:
    #     pass
    # else:
    #     val = val * 0.1
    # sigmoid for getting rid of some errors
    if val > 0:
        return 1/(1 + np.exp(-val))
    else:
        return np.exp(val)/(1 + np.exp(val))
    # regular sigmoid
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


def separateAnswers(list):
    global answerIdx
    global inputNeurons
    answers = []
    for sublist in list:
        for i in range(len(sublist)):
            if i == answerIdx:
                answers.append(sublist.pop(answerIdx))
    return answers


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

# setup data                    $ &


# testing data
testData = savedTestData
testAnswers = separateAnswers(testData)
testData = sigmoidInput(testData)
testDataSelection = r.randrange(0, len(testData))


# setup the input neurons with a random selection of the imported data


# code below moved down
# setupInputNeurons(trainingData[trainingDataSelection])

# end of data config

# start of layers


class neuron:
    def __init__(self, id, activationsOfPreviousLayer, fedInWeights, fedInBias):
        self.activationsOfPreviousLayer = activationsOfPreviousLayer
        self.fedInWeights = fedInWeights
        self.id = id
        self.innerWeights = fedInWeights
        self.bias = fedInBias

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

    def changeBiasAndActivation(self, deltaBias):
        self.bias = self.bias + deltaBias

        self.weightedSum = 0
        idx = 0
        # calculate the weighted sum -> w1a1 + w2a2 + ... + wn an
        for item in self.innerWeights:
            self.weightedSum += (item * self.activationsOfPreviousLayer[idx])
            idx += 1
        # calculate the activation from 0 to 1 -> sigmoid(weightedSum + bias)
        self.activation = sigmoid(self.weightedSum + self.bias)


class layer:
    def __init__(self, amountOfNeurons, activationsOfPreviousLayer, passInWeights=[], passInBiases=[]):
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
                feedWeights.append(r.uniform(-1.0, 1.0))
            # set a random number for the bias
            feedBias = r.randrange(-5.0, 5.0)
            self.neurons[idx] = neuron(
                idx, self.activationsOfPreviousLayer, feedWeights, feedBias)

        if passInWeights != []:
            if passInBiases != []:
                for idx in range(len(self.neurons)):
                    # feed in some weights that are given already
                    feedWeights = []
                    for _ in self.activationsOfPreviousLayer:
                        feedWeights.append(float(passInWeights[idx]))
                    # set a random number for the bias
                    feedBias = float(passInBiases[idx])
                    self.neurons[idx] = neuron(
                        idx, self.activationsOfPreviousLayer, feedWeights, feedBias)

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
        return f"Prediction: With {max(activations)}% certainty it is {activations.index(max(activations))}"

    def getCost(self, rightAnswer=False):
        global correctAnswer
        # only use for output layer
        actualOutput = []
        predictedOutput = []
        predictedOutput = self.getAllActivations()

        for i in range(1, len(predictedOutput)+1):
            if rightAnswer:
                if float(i-1) == float(correctAnswer):
                    actualOutput.append(1.0)
                else:
                    actualOutput.append(0.0)
            else:
                if float(i-1) == float(rightAnswer):
                    actualOutput.append(1.0)
                else:
                    actualOutput.append(0.0)
        # print(predictedOutput, actualOutput, correctAnswer)

        def meanSquaredError(p, y):
            # use this only (best cost for this usage)
            sum = 0
            for idx in range(len(p)):
                sum += np.square(y[idx] - p[idx])
            return sum

        # multiplying by 0.5 (dividing by 2) so that when doing the
        # derivative, it will cancel it out
        return 0.5*meanSquaredError(predictedOutput, actualOutput)

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
            # print(weights)
            # print(feedWeights)
            # set a random number for the bias
            feedBias = bias[idx]
            self.neurons[idx] = neuron(
                idx, self.activationsOfPreviousLayer, feedWeights, feedBias)

    def changeNeuronWeightAndActivation(self, neuronIndex, weightIndex_, deltaWeight_):
        self.neurons[neuronIndex].changeWeightAndActivation(
            weightIndex_, deltaWeight_)

    def changeNeuronBiasAndActivation(self, neuronIndex, deltaBias_):
        self.neurons[neuronIndex].changeBiasAndActivation(deltaBias_)


# backpropogate             $ &
def backpropogate(epochs, learnRate, inputNeurons, hiddenLayer1, hiddenLayer2, outputLayer, printOut, numberOfNeuronsHL1, numberOfNeuronsHL2, numberOfNeuronsOL):
    global correctAnswer
    global answers
    global trainingData
    global testData
    global testAnswers
    printProgressBar(0, 1)

    def avgCost(hiddenLayer1_, hiddenLayer2_, outputLayer_):
        global correctAnswer
        global answers
        global trainData
        totCost = 0
        for i in range(0, len(trainData)):
            inputNeurons_ = trainData[i]
            correctAnswer = answers[i]
            hiddenLayer1_.feedForwardAgain(
                numberOfNeuronsHL1, inputNeurons_, hiddenLayer1_.getAllWeights(), hiddenLayer1_.getAllBiases())

            hiddenLayer1Activations_ = hiddenLayer1.getAllActivations()
            hiddenLayer2_.feedForwardAgain(
                numberOfNeuronsHL2, hiddenLayer1Activations_, hiddenLayer2_.getAllWeights(), hiddenLayer2_.getAllBiases())

            hiddenLayer2Activations_ = hiddenLayer2.getAllActivations()
            outputLayer_.feedForwardAgain(
                numberOfNeuronsOL, hiddenLayer2Activations_, outputLayer_.getAllWeights(), outputLayer_.getAllBiases())
            totCost += float(outputLayer_.getCost(correctAnswer))
        return totCost/len(trainData)

    def feedForwardGetCost(hiddenLayer1_, hiddenLayer2_, outputLayer_, testing=False):
        hiddenLayer1_.feedForwardAgain(
            numberOfNeuronsHL1, inputNeurons, hiddenLayer1_.getAllWeights(), hiddenLayer1_.getAllBiases())

        hiddenLayer1Activations_ = hiddenLayer1.getAllActivations()
        hiddenLayer2_.feedForwardAgain(numberOfNeuronsHL2, hiddenLayer1Activations_,
                                       hiddenLayer2_.getAllWeights(), hiddenLayer2_.getAllBiases())

        hiddenLayer2Activations_ = hiddenLayer2.getAllActivations()
        outputLayer_.feedForwardAgain(numberOfNeuronsOL, hiddenLayer2Activations_,
                                      outputLayer_.getAllWeights(), outputLayer_.getAllBiases())

        if testing == False:
            return outputLayer_.getCost()
        else:
            return outputLayer_

    for epochCount in range(0, epochs):
        baselineCost = avgCost(hiddenLayer1, hiddenLayer2, outputLayer)
        neuronCount = 0
        weightCount = 0

        changeWeightAmount = learnRate
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
                    hiddenLayer1.feedForwardAgain(
                        numberOfNeuronsHL1, inputNeurons, hiddenLayer1.getAllWeights(), hiddenLayer1.getAllBiases())
                    hiddenLayer1.changeNeuronWeightAndActivation(
                        neuronCount, weightCount, changeWeight)

                    hiddenLayer1Activations = hiddenLayer1.getAllActivations()
                    hiddenLayer2.feedForwardAgain(
                        numberOfNeuronsHL2, hiddenLayer1Activations, hiddenLayer2.getAllWeights(), hiddenLayer2.getAllBiases())

                    hiddenLayer2Activations = hiddenLayer2.getAllActivations()
                    outputLayer.feedForwardAgain(
                        numberOfNeuronsOL, hiddenLayer2Activations, outputLayer.getAllWeights(), outputLayer.getAllBiases())

                    if i == 0:
                        initialCost = avgCost(
                            hiddenLayer1, hiddenLayer2, outputLayer)
                    else:
                        finalCost = avgCost(
                            hiddenLayer1, hiddenLayer2, outputLayer)

                    if printOut:
                        print(f"Correct Answer: {correctAnswer}")
                        print(f"{outputLayer.printPrediction()}")
                        print(f"Cost: {outputLayer.getCost()}")

                changeWeight = changeWeightAmount
                hiddenLayer1.changeNeuronWeightAndActivation(
                    neuronCount, weightCount, changeWeight)
                if initialCost < finalCost:
                    hiddenLayer1.changeNeuronWeightAndActivation(
                        neuronCount, weightCount, changeWeight)
                else:
                    changeWeight = -1 * changeWeightAmount
                    hiddenLayer1.changeNeuronWeightAndActivation(
                        neuronCount, weightCount, changeWeight)
                weightCount += 1
            neuronCount += 1

        #########################################################
        # train hidden layer 2 weights

        neuronCount = 0
        weightCount = 0

        changeWeightAmount = learnRate
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
                    hiddenLayer1.feedForwardAgain(
                        numberOfNeuronsHL1, inputNeurons, hiddenLayer1.getAllWeights(), hiddenLayer1.getAllBiases())

                    hiddenLayer1Activations = hiddenLayer1.getAllActivations()
                    hiddenLayer2.feedForwardAgain(
                        numberOfNeuronsHL2, hiddenLayer1Activations, hiddenLayer2.getAllWeights(), hiddenLayer2.getAllBiases())
                    hiddenLayer2.changeNeuronWeightAndActivation(
                        neuronCount, weightCount, changeWeight)

                    hiddenLayer2Activations = hiddenLayer2.getAllActivations()
                    outputLayer.feedForwardAgain(
                        numberOfNeuronsOL, hiddenLayer2Activations, outputLayer.getAllWeights(), outputLayer.getAllBiases())

                    if i == 0:
                        initialCost = avgCost(
                            hiddenLayer1, hiddenLayer2, outputLayer)
                    else:
                        finalCost = avgCost(
                            hiddenLayer1, hiddenLayer2, outputLayer)

                    if printOut:
                        print(f"Correct Answer: {correctAnswer}")
                        print(f"{outputLayer.printPrediction()}")
                        print(f"Cost: {outputLayer.getCost()}")

                changeWeight = changeWeightAmount
                hiddenLayer2.changeNeuronWeightAndActivation(
                    neuronCount, weightCount, changeWeight)
                if initialCost < finalCost:
                    hiddenLayer2.changeNeuronWeightAndActivation(
                        neuronCount, weightCount, changeWeight)
                else:
                    changeWeight = -1 * changeWeightAmount
                    hiddenLayer2.changeNeuronWeightAndActivation(
                        neuronCount, weightCount, changeWeight)
                weightCount += 1
            neuronCount += 1

        #########################################################

        # train output layer weights

        neuronCount = 0
        weightCount = 0

        changeWeightAmount = learnRate
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
                    hiddenLayer1.feedForwardAgain(
                        numberOfNeuronsHL1, inputNeurons, hiddenLayer1.getAllWeights(), hiddenLayer1.getAllBiases())

                    hiddenLayer1Activations = hiddenLayer1.getAllActivations()
                    hiddenLayer2.feedForwardAgain(
                        numberOfNeuronsHL2, hiddenLayer1Activations, hiddenLayer2.getAllWeights(), hiddenLayer2.getAllBiases())

                    hiddenLayer2Activations = hiddenLayer2.getAllActivations()
                    outputLayer.feedForwardAgain(
                        numberOfNeuronsOL, hiddenLayer2Activations, outputLayer.getAllWeights(), outputLayer.getAllBiases())
                    outputLayer.changeNeuronWeightAndActivation(
                        neuronCount, weightCount, changeWeight)

                    if i == 0:
                        initialCost = avgCost(
                            hiddenLayer1, hiddenLayer2, outputLayer)
                    else:
                        finalCost = avgCost(
                            hiddenLayer1, hiddenLayer2, outputLayer)

                    if printOut:
                        print(f"Correct Answer: {correctAnswer}")
                        print(f"{outputLayer.printPrediction()}")
                        print(f"Cost: {outputLayer.getCost()}")

                changeWeight = changeWeightAmount
                outputLayer.changeNeuronWeightAndActivation(
                    neuronCount, weightCount, changeWeight)
                if initialCost < finalCost:
                    outputLayer.changeNeuronWeightAndActivation(
                        neuronCount, weightCount, changeWeight)
                else:
                    changeWeight = -1 * changeWeightAmount
                    outputLayer.changeNeuronWeightAndActivation(
                        neuronCount, weightCount, changeWeight)
                weightCount += 1
            neuronCount += 1
        #######################################################
        # training for hidden layer 1 biases
        neuronCount = 0

        changeBiasAmount = learnRate
        changeBias = 0

        # loop amount of neurons in layer
        for i in range(hiddenLayer1.amountOfNeurons):
            # loop amount of neurons in previous layer (loop for the biases)

            changeBias = 0
            # loop for altering the bias by a little
            for i in range(2):
                if i == 0:
                    changeBias = float(changeBiasAmount)
                else:
                    changeBias = -2 * float(changeBiasAmount)
                # feed values in to get the cost
                hiddenLayer1.feedForwardAgain(
                    numberOfNeuronsHL1, inputNeurons, hiddenLayer1.getAllWeights(), hiddenLayer1.getAllBiases())
                hiddenLayer1.changeNeuronBiasAndActivation(
                    neuronCount, changeBias)

                hiddenLayer1Activations = hiddenLayer1.getAllActivations()
                hiddenLayer2.feedForwardAgain(
                    numberOfNeuronsHL2, hiddenLayer1Activations, hiddenLayer2.getAllWeights(), hiddenLayer2.getAllBiases())

                hiddenLayer2Activations = hiddenLayer2.getAllActivations()
                outputLayer.feedForwardAgain(
                    numberOfNeuronsOL, hiddenLayer2Activations, outputLayer.getAllWeights(), outputLayer.getAllBiases())

                if i == 0:
                    initialCost = avgCost(
                        hiddenLayer1, hiddenLayer2, outputLayer)
                else:
                    finalCost = avgCost(
                        hiddenLayer1, hiddenLayer2, outputLayer)

                if printOut:
                    print(f"Correct Answer: {correctAnswer}")
                    print(f"{outputLayer.printPrediction()}")
                    print(f"Cost: {outputLayer.getCost()}")

            changeBias = changeBiasAmount
            hiddenLayer1.changeNeuronBiasAndActivation(
                neuronCount, changeBias)
            if initialCost < finalCost:
                hiddenLayer1.changeNeuronBiasAndActivation(
                    neuronCount, changeBias)
            else:
                changeBias = -1 * changeBiasAmount
                hiddenLayer1.changeNeuronBiasAndActivation(
                    neuronCount, changeBias)
            neuronCount += 1

        # training for hidden layer 2 biases
        neuronCount = 0

        changeBiasAmount = learnRate
        changeBias = 0

        # loop amount of neurons in layer
        for i in range(hiddenLayer2.amountOfNeurons):
            # loop amount of neurons in previous layer (loop for the biases)

            changeBias = 0
            # loop for altering the bias by a little
            for i in range(2):
                if i == 0:
                    changeBias = float(changeBiasAmount)
                else:
                    changeBias = -2 * float(changeBiasAmount)
                # feed values in to get the cost
                hiddenLayer1.feedForwardAgain(
                    numberOfNeuronsHL1, inputNeurons, hiddenLayer1.getAllWeights(), hiddenLayer1.getAllBiases())

                hiddenLayer1Activations = hiddenLayer1.getAllActivations()
                hiddenLayer2.feedForwardAgain(
                    numberOfNeuronsHL2, hiddenLayer1Activations, hiddenLayer2.getAllWeights(), hiddenLayer2.getAllBiases())
                hiddenLayer2.changeNeuronBiasAndActivation(
                    neuronCount, changeBias)

                hiddenLayer2Activations = hiddenLayer2.getAllActivations()
                outputLayer.feedForwardAgain(
                    numberOfNeuronsOL, hiddenLayer2Activations, outputLayer.getAllWeights(), outputLayer.getAllBiases())

                if i == 0:
                    initialCost = avgCost(
                        hiddenLayer1, hiddenLayer2, outputLayer)
                else:
                    finalCost = avgCost(
                        hiddenLayer1, hiddenLayer2, outputLayer)

                if printOut:
                    print(f"Correct Answer: {correctAnswer}")
                    print(f"{outputLayer.printPrediction()}")
                    print(f"Cost: {outputLayer.getCost()}")

            changeBias = changeBiasAmount
            hiddenLayer2.changeNeuronBiasAndActivation(
                neuronCount, changeBias)
            if initialCost < finalCost:
                hiddenLayer2.changeNeuronBiasAndActivation(
                    neuronCount, changeBias)
            else:
                changeBias = -1 * changeBiasAmount
                hiddenLayer2.changeNeuronBiasAndActivation(
                    neuronCount, changeBias)
            neuronCount += 1

        # training for hidden layer 3 biases
        neuronCount = 0

        changeBiasAmount = learnRate
        changeBias = 0

        # loop amount of neurons in layer
        for i in range(outputLayer.amountOfNeurons):
            # loop amount of neurons in previous layer (loop for the biases)

            changeBias = 0
            # loop for altering the bias by a little
            for i in range(2):
                if i == 0:
                    changeBias = float(changeBiasAmount)
                else:
                    changeBias = -2 * float(changeBiasAmount)
                # feed values in to get the cost
                hiddenLayer1.feedForwardAgain(
                    numberOfNeuronsHL1, inputNeurons, hiddenLayer1.getAllWeights(), hiddenLayer1.getAllBiases())

                hiddenLayer1Activations = hiddenLayer1.getAllActivations()
                hiddenLayer2.feedForwardAgain(
                    numberOfNeuronsHL2, hiddenLayer1Activations, hiddenLayer2.getAllWeights(), hiddenLayer2.getAllBiases())

                hiddenLayer2Activations = hiddenLayer2.getAllActivations()
                outputLayer.feedForwardAgain(
                    numberOfNeuronsOL, hiddenLayer2Activations, outputLayer.getAllWeights(), outputLayer.getAllBiases())
                outputLayer.changeNeuronBiasAndActivation(
                    neuronCount, changeBias)

                if i == 0:
                    initialCost = avgCost(
                        hiddenLayer1, hiddenLayer2, outputLayer)
                else:
                    finalCost = avgCost(
                        hiddenLayer1, hiddenLayer2, outputLayer)

                if printOut:
                    print(f"Correct Answer: {correctAnswer}")
                    print(f"{outputLayer.printPrediction()}")
                    print(f"Cost: {outputLayer.getCost()}")

            changeBias = changeBiasAmount
            outputLayer.changeNeuronBiasAndActivation(
                neuronCount, changeBias)
            if initialCost < finalCost:
                outputLayer.changeNeuronBiasAndActivation(
                    neuronCount, changeBias)
            else:
                changeBias = -1 * changeBiasAmount
                outputLayer.changeNeuronBiasAndActivation(
                    neuronCount, changeBias)
            neuronCount += 1
        afterCost = avgCost(hiddenLayer1, hiddenLayer2, outputLayer)
        # only plot baselineCost if you want to see the change in cost
        # plotYAxis.append(baselineCost)
        plotYAxis.append(afterCost)
        accuracy = f"{(1-afterCost)*100}%"
        printProgressBar(epochCount, epochs, printEnd=accuracy)

    print(f"Correct Answer: {correctAnswer}")
    print(f"{outputLayer.printPrediction()}")
    print(f"Cost: {outputLayer.getCost()}")

    ##
    printProgressBar(1, 1)
    w = open("weights.txt", "r+")
    w.truncate(0)

    weightsHL1 = hiddenLayer1.getAllWeights()
    weightsHL2 = hiddenLayer2.getAllWeights()
    weightsOL = outputLayer.getAllWeights()

    w.write(str(weightsHL1))
    w.write("\n")
    w.write(str(weightsHL2))
    w.write("\n")
    w.write(str(weightsOL))
    w.close()

    ##
    b = open("biases.txt", "r+")
    b.truncate(0)

    biasesHL1 = hiddenLayer1.getAllBiases()
    biasesHL2 = hiddenLayer2.getAllBiases()
    biasesOL = outputLayer.getAllBiases()

    b.write(str(biasesHL1))
    b.write("\n")
    b.write(str(biasesHL2))
    b.write("\n")
    b.write(str(biasesOL))
    b.close()

    os.system("cls")
    print("Train Data below")
    for i in range(0, len(trainData)):
        pick = i
        inputNeurons = trainData[pick]
        correctAnswer = answers[pick]
        hiddenLayer1.feedForwardAgain(
            numberOfNeuronsHL1, inputNeurons, hiddenLayer1.getAllWeights(), hiddenLayer1.getAllBiases())

        hiddenLayer1Activations = hiddenLayer1.getAllActivations()
        hiddenLayer2.feedForwardAgain(numberOfNeuronsHL2, hiddenLayer1Activations,
                                      hiddenLayer2.getAllWeights(), hiddenLayer2.getAllBiases())

        hiddenLayer2Activations = hiddenLayer2.getAllActivations()
        outputLayer.feedForwardAgain(numberOfNeuronsOL, hiddenLayer2Activations,
                                     outputLayer.getAllWeights(), outputLayer.getAllBiases())

        print(f"{outputLayer.printPrediction()}, correct: {correctAnswer}")
    print("Test Data below")
    for i in range(0, len(testData)):
        pick = i
        inputNeurons = testData[pick]
        correctAnswer = testAnswers[pick]
        hiddenLayer1.feedForwardAgain(
            numberOfNeuronsHL1, inputNeurons, hiddenLayer1.getAllWeights(), hiddenLayer1.getAllBiases())

        hiddenLayer1Activations = hiddenLayer1.getAllActivations()
        hiddenLayer2.feedForwardAgain(numberOfNeuronsHL2, hiddenLayer1Activations,
                                      hiddenLayer2.getAllWeights(), hiddenLayer2.getAllBiases())

        hiddenLayer2Activations = hiddenLayer2.getAllActivations()
        outputLayer.feedForwardAgain(numberOfNeuronsOL, hiddenLayer2Activations,
                                     outputLayer.getAllWeights(), outputLayer.getAllBiases())

        print(f"{outputLayer.printPrediction()}, correct: {correctAnswer}")
    print(accuracy)


# code for fit function(training)


# fit (train)                 $ &
def fit(epochs, learnRate, printOut, reUse):
    '''
        usage - fit(epochs, learnRate, printOut, reUse)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        epochs - epochs are how many times you train on a specific sample
        learRate - learn rate should be between 0.01 and 1
        printOut - printOut specifies if it should print out its training results as it goes(make it True to display and False to not display)
        reUse - reUse specifies whether to import the old weights and biases
    '''
    global plotYAxis
    global correctAnswer
    if epochs < 1:
        quit(code="epoch is less than 1 (not logical to repeat 0 or negative times)")
    # training data
    global trainData
    global testData
    global trainingDataSelection
    global numberOfNeuronsHL1
    global numberOfNeuronsHL2
    global inputNeurons
    global hiddenLayer1
    global hiddenLayer2
    global outputLayer
    global answers
    answers = separateAnswers(trainData)
    plotYAxis = []
    trainData = []
    trainData = savedTrainData
    separateAnswers(trainData)
    outputCount = set(answers)
    numberOfNeuronsOL = len(outputCount)
    trainingData = trainData
    trainingData = sigmoidInput(trainingData)
    trainingDataSelection = r.randrange(0, len(trainData))
    testingDataSelection = r.randrange(0, len(testData))
    setupInputNeurons(trainingData[trainingDataSelection])
    # initialize model
    w = open("weights.txt", "r+")
    read = w.readlines()
    gottenWeights = []
    for i in range(len(read)):
        gottenWeights.append(re.findall("-?\d+.\d+", str(read[i])))
    w.close()

    b = open("biases.txt", "r+")
    read = b.readlines()
    gottenBiases = []
    for i in range(len(read)):
        gottenBiases.append(re.findall("-?\d+.\d+", str(read[i])))
    b.close()

    if reUse:
        hiddenLayer1 = layer(numberOfNeuronsHL1, inputNeurons,
                             gottenWeights[0], gottenBiases[0])

        # get the previous layer's activations to feed forward
        hiddenLayer1Activations = hiddenLayer1.getAllActivations()
        hiddenLayer2 = layer(
            numberOfNeuronsHL2, hiddenLayer1Activations, gottenWeights[1], gottenBiases[1])

        # get the previous layer's activations to feed forward
        hiddenLayer2Activations = hiddenLayer2.getAllActivations()
        outputLayer = layer(
            numberOfNeuronsOL, hiddenLayer2Activations, gottenWeights[2], gottenBiases[2])

        correctAnswer = str(answers[trainingDataSelection])
    else:
        hiddenLayer1 = layer(numberOfNeuronsHL1, inputNeurons)

        # get the previous layer's activations to feed forward
        hiddenLayer1Activations = hiddenLayer1.getAllActivations()
        hiddenLayer2 = layer(numberOfNeuronsHL2, hiddenLayer1Activations)

        # get the previous layer's activations to feed forward
        hiddenLayer2Activations = hiddenLayer2.getAllActivations()
        outputLayer = layer(numberOfNeuronsOL, hiddenLayer2Activations)

        correctAnswer = str(answers[trainingDataSelection])

    ########################################################################

    backpropogate(epochs, learnRate, inputNeurons, hiddenLayer1, hiddenLayer2, outputLayer,
                  printOut, numberOfNeuronsHL1, numberOfNeuronsHL2, numberOfNeuronsOL)

 # fit(epochs, learnRate, printOut, reUse)
 # epochs are how many times you train on a certain sample
 # learn rate should be between 0.01 and 1
 # printOut specifies if it should print out its training results as it goes(make it True to display and False to not display)
 # reUse specifies whether to import the old weights and biases
numberOfNeuronsHL1 = 5
numberOfNeuronsHL2 = 10
startTime = t.time()
# implement                    $ &
fit(10, 0.1, False, False)
endTime = t.time()
elapsedTime = endTime - startTime


def time_convert(sec):
    mins = sec // 60
    sec = sec % 60
    hours = mins // 60
    mins = mins % 60
    print("Time Lapsed = {0}:{1}:{2}".format(int(hours), int(mins), sec))


time_convert(elapsedTime)

# plots things
plt.xlabel("Epoch")
plt.ylabel("Cost")
plt.fill_between([x for x in range(0, len(plotYAxis))], plotYAxis)
plt.plot(plotYAxis)
plt.show()

###########################################


# # code for training on one sample
# def trainSample(epochs, learnRate, inputNeurons, hiddenLayer1, hiddenLayer2, outputLayer, printOut, numberOfNeuronsHL1, numberOfNeuronsHL2, numberOfNeuronsOL):
#     for epochCount in range(epochs):
#         neuronCount = 0
#         weightCount = 0

#         changeWeightAmount = learnRate
#         changeWeight = 0

#         # loop amount of neurons in layer
#         for i in range(hiddenLayer1.amountOfNeurons):
#             weightCount = 0
#             # loop amount of neurons in previous layer (loop for the weights)
#             for i in range(len(inputNeurons)):
#                 changeWeight = 0
#                 # loop for altering the weight by a little
#                 for i in range(2):
#                     if i == 0:
#                         changeWeight = float(changeWeightAmount)
#                     else:
#                         changeWeight = -2 * float(changeWeightAmount)
#                     # feed values in to get the cost
#                     hiddenLayer1.feedForwardAgain(
#                         numberOfNeuronsHL1, inputNeurons, hiddenLayer1.getAllWeights(), hiddenLayer1.getAllBiases())
#                     hiddenLayer1.changeNeuronWeightAndActivation(
#                         neuronCount, weightCount, changeWeight)

#                     hiddenLayer1Activations = hiddenLayer1.getAllActivations()
#                     hiddenLayer2.feedForwardAgain(
#                         numberOfNeuronsHL2, hiddenLayer1Activations, hiddenLayer2.getAllWeights(), hiddenLayer2.getAllBiases())

#                     hiddenLayer2Activations = hiddenLayer2.getAllActivations()
#                     outputLayer.feedForwardAgain(
#                         numberOfNeuronsOL, hiddenLayer2Activations, outputLayer.getAllWeights(), outputLayer.getAllBiases())

#                     if i == 0:
#                         initialCost = outputLayer.getCost()
#                     else:
#                         finalCost = outputLayer.getCost()

#                     if printOut:
#                         print(f"Correct Answer: {correctAnswer}")
#                         print(f"{outputLayer.printPrediction()}")
#                         print(f"Cost: {outputLayer.getCost()}")

#                 changeWeight = changeWeightAmount
#                 hiddenLayer1.changeNeuronWeightAndActivation(
#                     neuronCount, weightCount, changeWeight)
#                 if initialCost < finalCost:
#                     hiddenLayer1.changeNeuronWeightAndActivation(
#                         neuronCount, weightCount, changeWeight)
#                 else:
#                     changeWeight = -1 * changeWeightAmount
#                     hiddenLayer1.changeNeuronWeightAndActivation(
#                         neuronCount, weightCount, changeWeight)
#                 weightCount += 1
#             neuronCount += 1

#         #########################################################
#         # train hidden layer 2 weights

#         neuronCount = 0
#         weightCount = 0

#         changeWeightAmount = learnRate
#         changeWeight = 0

#         # loop amount of neurons in layer
#         for i in range(hiddenLayer2.amountOfNeurons):
#             weightCount = 0
#             # loop amount of neurons in previous layer (loop for the weights)
#             for i in range(hiddenLayer1.amountOfNeurons):
#                 changeWeight = 0
#                 # loop for altering the weight by a little
#                 for i in range(2):
#                     if i == 0:
#                         changeWeight = float(changeWeightAmount)
#                     else:
#                         changeWeight = -2 * float(changeWeightAmount)
#                     # feed values in to get the cost
#                     hiddenLayer1.feedForwardAgain(
#                         numberOfNeuronsHL1, inputNeurons, hiddenLayer1.getAllWeights(), hiddenLayer1.getAllBiases())

#                     hiddenLayer1Activations = hiddenLayer1.getAllActivations()
#                     hiddenLayer2.feedForwardAgain(
#                         numberOfNeuronsHL2, hiddenLayer1Activations, hiddenLayer2.getAllWeights(), hiddenLayer2.getAllBiases())
#                     hiddenLayer2.changeNeuronWeightAndActivation(
#                         neuronCount, weightCount, changeWeight)

#                     hiddenLayer2Activations = hiddenLayer2.getAllActivations()
#                     outputLayer.feedForwardAgain(
#                         numberOfNeuronsOL, hiddenLayer2Activations, outputLayer.getAllWeights(), outputLayer.getAllBiases())

#                     if i == 0:
#                         initialCost = outputLayer.getCost()
#                     else:
#                         finalCost = outputLayer.getCost()

#                     if printOut:
#                         print(f"Correct Answer: {correctAnswer}")
#                         print(f"{outputLayer.printPrediction()}")
#                         print(f"Cost: {outputLayer.getCost()}")

#                 changeWeight = changeWeightAmount
#                 hiddenLayer2.changeNeuronWeightAndActivation(
#                     neuronCount, weightCount, changeWeight)
#                 if initialCost < finalCost:
#                     hiddenLayer2.changeNeuronWeightAndActivation(
#                         neuronCount, weightCount, changeWeight)
#                 else:
#                     changeWeight = -1 * changeWeightAmount
#                     hiddenLayer2.changeNeuronWeightAndActivation(
#                         neuronCount, weightCount, changeWeight)
#                 weightCount += 1
#             neuronCount += 1

#         #########################################################

#         # train output layer weights

#         neuronCount = 0
#         weightCount = 0

#         changeWeightAmount = learnRate
#         changeWeight = 0

#         # loop amount of neurons in layer
#         for i in range(outputLayer.amountOfNeurons):
#             weightCount = 0
#             # loop amount of neurons in previous layer (loop for the weights)
#             for i in range(hiddenLayer2.amountOfNeurons):
#                 changeWeight = 0
#                 # loop for altering the weight by a little
#                 for i in range(2):
#                     if i == 0:
#                         changeWeight = float(changeWeightAmount)
#                     else:
#                         changeWeight = -2 * float(changeWeightAmount)
#                     # feed values in to get the cost
#                     hiddenLayer1.feedForwardAgain(
#                         numberOfNeuronsHL1, inputNeurons, hiddenLayer1.getAllWeights(), hiddenLayer1.getAllBiases())

#                     hiddenLayer1Activations = hiddenLayer1.getAllActivations()
#                     hiddenLayer2.feedForwardAgain(
#                         numberOfNeuronsHL2, hiddenLayer1Activations, hiddenLayer2.getAllWeights(), hiddenLayer2.getAllBiases())

#                     hiddenLayer2Activations = hiddenLayer2.getAllActivations()
#                     outputLayer.feedForwardAgain(
#                         numberOfNeuronsOL, hiddenLayer2Activations, outputLayer.getAllWeights(), outputLayer.getAllBiases())
#                     outputLayer.changeNeuronWeightAndActivation(
#                         neuronCount, weightCount, changeWeight)

#                     if i == 0:
#                         initialCost = outputLayer.getCost()
#                     else:
#                         finalCost = outputLayer.getCost()

#                     if printOut:
#                         print(f"Correct Answer: {correctAnswer}")
#                         print(f"{outputLayer.printPrediction()}")
#                         print(f"Cost: {outputLayer.getCost()}")

#                 changeWeight = changeWeightAmount
#                 outputLayer.changeNeuronWeightAndActivation(
#                     neuronCount, weightCount, changeWeight)
#                 if initialCost < finalCost:
#                     outputLayer.changeNeuronWeightAndActivation(
#                         neuronCount, weightCount, changeWeight)
#                 else:
#                     changeWeight = -1 * changeWeightAmount
#                     outputLayer.changeNeuronWeightAndActivation(
#                         neuronCount, weightCount, changeWeight)
#                 weightCount += 1
#             neuronCount += 1
#         plotYAxis.append(outputLayer.getCost())
#         #######################################################
#         # training for hidden layer 1 biases
#         neuronCount = 0

#         changeBiasAmount = learnRate
#         changeBias = 0

#         # loop amount of neurons in layer
#         for i in range(hiddenLayer1.amountOfNeurons):
#             # loop amount of neurons in previous layer (loop for the biases)

#             changeBias = 0
#             # loop for altering the bias by a little
#             for i in range(2):
#                 if i == 0:
#                     changeBias = float(changeBiasAmount)
#                 else:
#                     changeBias = -2 * float(changeBiasAmount)
#                 # feed values in to get the cost
#                 hiddenLayer1.feedForwardAgain(
#                     numberOfNeuronsHL1, inputNeurons, hiddenLayer1.getAllWeights(), hiddenLayer1.getAllBiases())
#                 hiddenLayer1.changeNeuronBiasAndActivation(
#                     neuronCount, changeBias)

#                 hiddenLayer1Activations = hiddenLayer1.getAllActivations()
#                 hiddenLayer2.feedForwardAgain(
#                     numberOfNeuronsHL2, hiddenLayer1Activations, hiddenLayer2.getAllWeights(), hiddenLayer2.getAllBiases())

#                 hiddenLayer2Activations = hiddenLayer2.getAllActivations()
#                 outputLayer.feedForwardAgain(
#                     numberOfNeuronsOL, hiddenLayer2Activations, outputLayer.getAllWeights(), outputLayer.getAllBiases())

#                 if i == 0:
#                     initialCost = outputLayer.getCost()
#                 else:
#                     finalCost = outputLayer.getCost()

#                 if printOut:
#                     print(f"Correct Answer: {correctAnswer}")
#                     print(f"{outputLayer.printPrediction()}")
#                     print(f"Cost: {outputLayer.getCost()}")

#             changeBias = changeBiasAmount
#             hiddenLayer1.changeNeuronBiasAndActivation(
#                 neuronCount, changeBias)
#             if initialCost < finalCost:
#                 hiddenLayer1.changeNeuronBiasAndActivation(
#                     neuronCount, changeBias)
#             else:
#                 changeBias = -1 * changeBiasAmount
#                 hiddenLayer1.changeNeuronBiasAndActivation(
#                     neuronCount, changeBias)
#             neuronCount += 1

#         # training for hidden layer 2 biases
#         neuronCount = 0

#         changeBiasAmount = learnRate
#         changeBias = 0

#         # loop amount of neurons in layer
#         for i in range(hiddenLayer2.amountOfNeurons):
#             # loop amount of neurons in previous layer (loop for the biases)

#             changeBias = 0
#             # loop for altering the bias by a little
#             for i in range(2):
#                 if i == 0:
#                     changeBias = float(changeBiasAmount)
#                 else:
#                     changeBias = -2 * float(changeBiasAmount)
#                 # feed values in to get the cost
#                 hiddenLayer1.feedForwardAgain(
#                     numberOfNeuronsHL1, inputNeurons, hiddenLayer1.getAllWeights(), hiddenLayer1.getAllBiases())

#                 hiddenLayer1Activations = hiddenLayer1.getAllActivations()
#                 hiddenLayer2.feedForwardAgain(
#                     numberOfNeuronsHL2, hiddenLayer1Activations, hiddenLayer2.getAllWeights(), hiddenLayer2.getAllBiases())
#                 hiddenLayer2.changeNeuronBiasAndActivation(
#                     neuronCount, changeBias)

#                 hiddenLayer2Activations = hiddenLayer2.getAllActivations()
#                 outputLayer.feedForwardAgain(
#                     numberOfNeuronsOL, hiddenLayer2Activations, outputLayer.getAllWeights(), outputLayer.getAllBiases())

#                 if i == 0:
#                     initialCost = outputLayer.getCost()
#                 else:
#                     finalCost = outputLayer.getCost()

#                 if printOut:
#                     print(f"Correct Answer: {correctAnswer}")
#                     print(f"{outputLayer.printPrediction()}")
#                     print(f"Cost: {outputLayer.getCost()}")

#             changeBias = changeBiasAmount
#             hiddenLayer2.changeNeuronBiasAndActivation(
#                 neuronCount, changeBias)
#             if initialCost < finalCost:
#                 hiddenLayer2.changeNeuronBiasAndActivation(
#                     neuronCount, changeBias)
#             else:
#                 changeBias = -1 * changeBiasAmount
#                 hiddenLayer2.changeNeuronBiasAndActivation(
#                     neuronCount, changeBias)
#             neuronCount += 1

#         # training for hidden layer 3 biases
#         neuronCount = 0

#         changeBiasAmount = learnRate
#         changeBias = 0

#         # loop amount of neurons in layer
#         for i in range(outputLayer.amountOfNeurons):
#             # loop amount of neurons in previous layer (loop for the biases)

#             changeBias = 0
#             # loop for altering the bias by a little
#             for i in range(2):
#                 if i == 0:
#                     changeBias = float(changeBiasAmount)
#                 else:
#                     changeBias = -2 * float(changeBiasAmount)
#                 # feed values in to get the cost
#                 hiddenLayer1.feedForwardAgain(
#                     numberOfNeuronsHL1, inputNeurons, hiddenLayer1.getAllWeights(), hiddenLayer1.getAllBiases())

#                 hiddenLayer1Activations = hiddenLayer1.getAllActivations()
#                 hiddenLayer2.feedForwardAgain(
#                     numberOfNeuronsHL2, hiddenLayer1Activations, hiddenLayer2.getAllWeights(), hiddenLayer2.getAllBiases())

#                 hiddenLayer2Activations = hiddenLayer2.getAllActivations()
#                 outputLayer.feedForwardAgain(
#                     numberOfNeuronsOL, hiddenLayer2Activations, outputLayer.getAllWeights(), outputLayer.getAllBiases())
#                 outputLayer.changeNeuronBiasAndActivation(
#                     neuronCount, changeBias)

#                 if i == 0:
#                     initialCost = outputLayer.getCost()
#                 else:
#                     finalCost = outputLayer.getCost()

#                 if printOut:
#                     print(f"Correct Answer: {correctAnswer}")
#                     print(f"{outputLayer.printPrediction()}")
#                     print(f"Cost: {outputLayer.getCost()}")

#             changeBias = changeBiasAmount
#             outputLayer.changeNeuronBiasAndActivation(
#                 neuronCount, changeBias)
#             if initialCost < finalCost:
#                 outputLayer.changeNeuronBiasAndActivation(
#                     neuronCount, changeBias)
#             else:
#                 changeBias = -1 * changeBiasAmount
#                 outputLayer.changeNeuronBiasAndActivation(
#                     neuronCount, changeBias)
#             neuronCount += 1
#     print(f"Correct Answer: {correctAnswer}")
#     print(f"{outputLayer.printPrediction()}")
#     print(f"Cost: {outputLayer.getCost()}")

#     ##

#     w = open("weights.txt", "r+")
#     w.truncate(0)

#     weightsHL1 = hiddenLayer1.getAllWeights()
#     weightsHL2 = hiddenLayer2.getAllWeights()
#     weightsOL = outputLayer.getAllWeights()

#     w.write(str(weightsHL1))
#     w.write("\n")
#     w.write(str(weightsHL2))
#     w.write("\n")
#     w.write(str(weightsOL))
#     w.close()

#     ##
#     b = open("biases.txt", "r+")
#     b.truncate(0)

#     biasesHL1 = hiddenLayer1.getAllBiases()
#     biasesHL2 = hiddenLayer2.getAllBiases()
#     biasesOL = outputLayer.getAllBiases()

#     b.write(str(biasesHL1))
#     b.write("\n")
#     b.write(str(biasesHL2))
#     b.write("\n")
#     b.write(str(biasesOL))
#     b.close()
