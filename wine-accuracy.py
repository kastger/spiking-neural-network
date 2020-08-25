"""
FILES IN PACKAGE:
snn-stdp.py
get-networks-accuracy.py
read-pkl-spikes.py
iris-train.txt
iris-test.txt
README.md

Files generated after training:
results/wineInput.pkl
results/wineOutput.pkl

Files generated after testing:
results/inputWineTest.pkl
results/outputWineTest.pkl

Author: Geraldas Kastauna
Email: GK468@live.mdx.ac.uk OR geraldaskastauna@gmail.com

About this program:
This program reads the spikes of the output file (outputWineTest.pkl) that was generated 
after testing in snn-wines.py program. Converts those spikes to answers and calculates
the accuracy of the network by comparing system generated answers to correct answer
in wine-test.txt file. Prints total correct answers and accuracy percentage to the console
"""
# === Dependencies ===
import numpy as np
import _pickle as pickle
import csv

# === File names ===
TESTING_RESULTS_FILE = 'results/outputWineTest.pkl'
TESTING_FILE = 'wine-test.txt'

# === Time variables ===
# FIRING_PERIOD should be the same as in snn-wines.py used during training
FIRING_PERIOD = 25.0 #ms
START_TIME = 10.0 #ms

# === Functions ===
"""
Functions that reads spike files that was saved after iris testing (or training)
by snn-stdp.py program and returns spike sequence of each neuron saved in an array.
It also ignores all the spikes before testStartTime argument (users input).
Parameters: file name (string), 
            testStartTime (int or float)
Returns: an array of spike sequence for each neuron in a 2D array (float values)
"""
def read_wine_result_spikes_file(fileName, testStartTime):
    timeResults = [[],[],[]]

    fileHandler = open(fileName, 'rb') # Opens a file for reading only in binary format
    neoObject = pickle.load(fileHandler, encoding = 'utf8') 
    segments = neoObject.segments
    segment = segments[0]
    spikeSequence = segment.spiketrains
    totalNeurons = len(spikeSequence)
    
    # Loop through all neurons
    for neuron in range(0, totalNeurons):
        spikeSequenceOfEachNeuron = spikeSequence[neuron]
        numberOfSpikesForEachNeuron = len(spikeSequenceOfEachNeuron)

        # Loop through all spikes in a neuron
        for spike in range(0, numberOfSpikesForEachNeuron):
            # Get one spike at a time and convert for Quanitity type to float
            spikeTime = str(spikeSequenceOfEachNeuron[spike]).split()
            spikeTime = float(spikeTime[0])

            # Ignore training spikes (before testStartTime)
            if (spikeTime >= testStartTime):
                # Save results of each neuron
                timeResults[neuron] = timeResults[neuron] + [spikeTime]

    return timeResults

"""
Function that gets the correct answers from wines data file
Parameters: fileName (string)
Returns: array full of integer values (1 to 3)
"""
def get_correct_wine_answers(fileName):
    answers = []
    
    # Open file using CSV file reader
    dataFileHandle = list(csv.reader(open(fileName)))
    totalWines = len(dataFileHandle)

    # Loop through test wine data and save the answers into an array
    for wine in range(0, totalWines):
        answer = int(dataFileHandle[wine][0])
        answers = answers + [answer]
    return answers

"""
Function that calculates firing rates in a neuron taking testStartTime
(for what time the spikes should be ignored) and exampleTime () 
variables into consideration and saves systems answers into an array. 
It also ignores everything after wineNumber becomes higher than 89
(because thats how many irises are in the test data file).
Parameters: outputNeuronNumber (0-2 for iris data because of 3 different classes)
            totalNumberOfSpikesInANeuron (self-explanatory)
            spikeSequence (spikes sequence of all neurons in a 2D array)
            testStartTime (time for which the spikes will be ignored starting 0.0 ms)
            exampleTime ()
Returns: answer (array containing amount of firing spikes,)
"""
def calculate_firing_rate(outputNeuronNumber, totalNumberOfSpikesInANeuron, 
                                            spikeSequence, testStartTime, exampleTime):
    totalTestWines = len(get_correct_wine_answers(TESTING_FILE))

    # Create an array full of zeros with the size of the testing data file
    answer = np.zeros(totalTestWines)

    # Loop through first neuron spike sequence
    for spikeTime in range(0, totalNumberOfSpikesInANeuron):
        actualSpikeTime = spikeSequence[outputNeuronNumber][spikeTime]
        baseTime = actualSpikeTime - testStartTime
        wineNumber = baseTime / exampleTime
        wineNumber = int(wineNumber)
        if (wineNumber >= totalTestWines):
            print('Ignoring firing in class ', outputNeuronNumber, 
                            "actual spike time: ", actualSpikeTime)
        else:
            answer[wineNumber] = answer[wineNumber] + 1
    
    return answer

"""
Function that saves the firing rates of each output neuron and returns 
all of that information in an array.
Parameters: spikeSequence (spikes sequence of all neurons in a 2D array)
            testStartTime (time for which the spikes will be ignored starting 0.0 ms)
            exampleTime ()
Returns: firingRates (array containing firing rates of each output neuron)
"""
def save_firing_rate_of_each_neuron(spikeSequence, testStartTime, exampleTime):
    firingRates = [[], [], []]
    totalNumberOfNeurons = len(spikeSequence)

    # Loop through all output neurons
    for neuron in range(0, totalNumberOfNeurons):
        totalSpikesInANeuron = len(spikeSequence[neuron])
        firingRates[neuron] = calculate_firing_rate(neuron, totalSpikesInANeuron, 
                                                              spikeSequence, testStartTime, 
                                                              exampleTime)

    return firingRates

"""
Function that checks each output layer neuron firing rate and compares them in order to
determine the answer. For example: If first neuron firing rate is higher than second and
third neuron rate then the classification of the iris class is 1 (iris-setosa).
All other options return 0 meaning that the network did not know how to categorise them.
Parameters: spikePerExample (2D array of spikes of each neuron generated by
            convertTestSpikeTimesToPerExample function)
Returns: answers (array of integers)
"""
def convert_rates_to_answers(firingRates):
    answers = []
    totalNumberOfAnswers = len(firingRates[0])

    # Loop through all answers (0 to 89)
    for eachAnswer in range (0, totalNumberOfAnswers):
        answer = 0

        # Get spikes of each neuron
        firstOutputLayerNeuronSpike = firingRates[0][eachAnswer]
        secondOutputLayerNeuronSpike = firingRates[1][eachAnswer]
        thirdOutputLayerNeuronSpike = firingRates[2][eachAnswer]

        # Determine the answer
        if (firstOutputLayerNeuronSpike > secondOutputLayerNeuronSpike) and (firstOutputLayerNeuronSpike > thirdOutputLayerNeuronSpike): #1 > 2&3
            answer = 1
        elif (secondOutputLayerNeuronSpike > firstOutputLayerNeuronSpike) and (secondOutputLayerNeuronSpike > thirdOutputLayerNeuronSpike): #2 > 1&3
            answer = 2
        elif (thirdOutputLayerNeuronSpike > firstOutputLayerNeuronSpike) and (thirdOutputLayerNeuronSpike > secondOutputLayerNeuronSpike): #3 > 1&2
            answer = 3
        else: answer = 0
        
        # Save the answers into an array
        answers = answers + [answer]

    return answers

"""
Function that counts the amount of correct system answers and 
accuracy percentage and prints it to the console.
Parameters: systemAnswers (array of integers),
            correctAnswers (array of integers)
"""
def get_networks_accuracy(systemAnswers, correctAnswers):
    countCorrectGuesses = 0
    totalAnswers = len(correctAnswers)

    # Loop through all answers
    for answer in range(0, totalAnswers):
        if(systemAnswers[answer] == correctAnswers[answer]):
            countCorrectGuesses = countCorrectGuesses + 1
    # Calculate the accuracy percentage
    accuracyPercentage = countCorrectGuesses / totalAnswers * 100

    # Print to console
    print("\nCorrect system answers:", countCorrectGuesses, "/", totalAnswers)
    print("Accuracy percentage:", accuracyPercentage, "%")

# === Main ===
correctAnswers = get_correct_wine_answers(TESTING_FILE)
print(correctAnswers)
spikeSequence = read_wine_result_spikes_file(TESTING_RESULTS_FILE, START_TIME)
firingRates = save_firing_rate_of_each_neuron(spikeSequence, START_TIME, FIRING_PERIOD)
systemAnswers = convert_rates_to_answers(firingRates)
print(systemAnswers)

# === Print the correct answers and accuracy percentage ===
get_networks_accuracy(systemAnswers, correctAnswers)

# === End of program ===