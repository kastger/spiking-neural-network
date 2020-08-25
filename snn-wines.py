"""
FILES IN PACKAGE:
snn.py
wine.data
wine.names
wine-test.txt
wine-train.txt
wine-accuracy.py

Author: Geraldas Kastauna
Email: GK468@live.mdx.ac.uk OR geraldaskastauna@gmail.com

About this program:
This program reads the wine training data of 89 wines and
creates a spiking neural network that uses spike-timing-dependent plasticity (STDP)
synapses for training. It generates a spike sequence for presynaptic and
postsynaptic layers which are used during network training period. After training
the connections between neurons are saved and used for testing on a static synapse 
model that has connections with fixed weights and delay. The network has to classify
wines based on all 13 features

THIS IS A FUTHER STUDY WITH snn-stdp.py WHICH WAS MADE FOR IRIS CLASSIFICATION
"""

# === Dependencies ===
import csv
import pyNN.nest as sim

# === Parameters for spiking neural network ===
# === Text files ===
TRAINING_FILE = "wine-train.txt"
TESTING_FILE = "wine-test.txt"

# === Simulation time and epochs ===
TRAINING_TIME = 8000.0 #ms
TESTING_TIME = 4000.0 #ms
FIRING_PERIOD = 25.0 #ms

TRAINING_EPOCHS = 20

TRAINING_START_TIME = 10.0 #ms (error when 0)
TESTING_START_TIME = 10.0 #ms

# === Number of neurons for each layer ===
INPUT_LAYER_NEURONS = 338
OUTPUT_LAYER_NEURONS = 3

NEURONS_PER_FEATURE = 26

numberOfWineClasses = 3
synapses = 0 # Used to save weights during training

# === Variables for network initialization ===
TIME_STEP = 0.1 #ms
MIN_DELAY = 0.1 #ms
MAX_DELAY = 0.1 #ms
DELAY = 0.1     #ms
LEARNING_OFFSET = 1.0

# === Firing weight for connections ===
FIRING_WEIGHT = 0.1

# === Functions ===
"""
Function that reads the wine data from the file and converts to integer values.
Parameters: string (file name)
Returns an array
"""
def read_file(fileName):
    data = []
    # Open file using CSV file reader
    dataFileHandle = list(csv.reader(open(fileName)))
    totalWines = len(dataFileHandle)

    # Loop through all wines
    for lineNumber in range (0, totalWines):
        wine_class = int(dataFileHandle[lineNumber][0])

        alcohol = float(dataFileHandle[lineNumber][1])
        alcohol = int(alcohol * 100)

        malic_acid = float(dataFileHandle[lineNumber][2])
        malic_acid = int(malic_acid * 100)

        ash = float(dataFileHandle[lineNumber][3])
        ash = int(ash * 100)

        alcalinity_of_ash = float(dataFileHandle[lineNumber][4])
        alcalinity_of_ash = int(alcalinity_of_ash * 100)

        magnesium = int(dataFileHandle[lineNumber][5])

        total_phenols = float(dataFileHandle[lineNumber][6])
        total_phenols = int(total_phenols * 100)

        flavanoids = float(dataFileHandle[lineNumber][7])
        flavanoids = int(flavanoids * 100)

        nonflavanoid_phenols = float(dataFileHandle[lineNumber][8])
        nonflavanoid_phenols = int(nonflavanoid_phenols * 100)

        proanthocyanins = float(dataFileHandle[lineNumber][9])
        proanthocyanins = int(proanthocyanins * 100)

        color_intensity = float(dataFileHandle[lineNumber][10])
        color_intensity = int(color_intensity * 100)

        hue = float(dataFileHandle[lineNumber][11])
        hue = int(hue * 100)

        wines_od = float(dataFileHandle[lineNumber][12])
        wines_od = int(wines_od * 100)

        proline = int(dataFileHandle[lineNumber][13])

        # Add each line to data array
        dataLine = [wine_class, alcohol, malic_acid, ash, alcalinity_of_ash, 
        magnesium, total_phenols, flavanoids, nonflavanoid_phenols, 
        proanthocyanins, color_intensity, hue, wines_od, proline]

        data = data + [dataLine]

    return data

"""
Function that initializes the PyNN simulated network
Parameters: timestep (float),
           min_delay (float),
           max_delay (float),
           debug (integer)
"""
def initialize_network(time_step, min_delay, max_delay):
    sim.setup(timestep = time_step, min_delay = min_delay, max_delay = max_delay)

"""
Function that creates a spike sequence for data file.
Parameters: data (array), 
            startTime (float), 
            learningOffSet (integer)
Returns: Generated nest simulator input and output spikes
             (array of 2 elements - [0] input [1] output)
"""
def create_spike_sequence(data, startTime, learningOffSet):
    numberOfDataElements = len(data)
    epochTime = FIRING_PERIOD * numberOfDataElements

    # Empty arrays to store spike sequences
    inputSpikeTimesSequence = []
    outputSpikeTimesSequence = []

    for element in range (0, numberOfDataElements):
        # Empty arrays to store spike times
        inputSpikeTimes = []
        outputSpikeTimes = []

        for epoch in range (0, TRAINING_EPOCHS): 
            # Calculate the timing of spikes for input layer and store them into inputSpikeTimes
            inputSpikeTime = startTime + (element * FIRING_PERIOD) + (epoch * epochTime)
            inputSpikeTimes = inputSpikeTimes + [inputSpikeTime]

            # Calculate the timing of spikes for output layer and store them into outputSpikeTimes
            outputSpikeTime = inputSpikeTime + learningOffSet
            outputSpikeTimes = outputSpikeTimes + [outputSpikeTime]

        # Create spike times sequence array for input and output layers
        inputSpikeTimesSequence = inputSpikeTimesSequence + [inputSpikeTimes]
        outputSpikeTimesSequence = outputSpikeTimesSequence + [outputSpikeTimes]

    # Generate spikes for input and output with previously stored spike times
    generatedInputSpikes = sim.Population(numberOfDataElements, sim.SpikeSourceArray,
                {'spike_times': inputSpikeTimesSequence}, label="input spikes sequence")

    generatedOutputSpikes = sim.Population(numberOfDataElements, sim.SpikeSourceArray,
                {'spike_times': outputSpikeTimesSequence}, label="output spikes sequence")

    return [generatedInputSpikes, generatedOutputSpikes]

"""
FOR NETWORK TESTING
Function that creates a spike sequence for testing.
Parameters: data (array), 
            startTime (float)
Returns: Generated PyNN.nest spikes
"""
def create_test_spike_sequence(data, startTime):
    numberDataItems = len(data)
    testSpikeTimesSequence = []

    # Loop through all data elements (89 for wine test data)
    for element in range (0, numberDataItems):
        # Calculate the test timing of spikes for each element and store them into testSpikeTimesSequnece
        testSpikeTimes = [[startTime + (element * FIRING_PERIOD)]]
        testSpikeTimesSequence = testSpikeTimesSequence + testSpikeTimes

    # Generate spikes for testing with previously stored spike times
    generatedTestSpikes = sim.Population(numberDataItems, sim.SpikeSourceArray,
                {'spike_times': testSpikeTimesSequence}, label="test spike sequence")

    return generatedTestSpikes

"""
Function that calculates which neurons should be simulated for each wine data 
feature and saves the connections in an array for later use.
Parameters: dataItem (array which has whole wine line with all 13 features and a class)
            irisNumberOnTheList (integer that represents wine number on the list)
Returns: an array of neuron connections
"""
def generate_feature_connections(dataItem, wineNumberOnTheList):
    inputConnector = []
    total_features = len(dataItem)

    # Loop through all wine features
    for dataFeature in range (0, total_features):
        dataFeatureValue = dataItem[dataFeature]

        # Loop through neurons in range that depends on feature numerical value
        for neuron in range((dataFeatureValue - numberOfWineClasses), (dataFeatureValue + numberOfWineClasses + 1)):
            if ((neuron >= 0) and (neuron < NEURONS_PER_FEATURE)):
                toNeuron = neuron + (dataFeature * NEURONS_PER_FEATURE)
                inputConnector = inputConnector + [(wineNumberOnTheList, toNeuron, FIRING_WEIGHT, TIME_STEP)]

    return inputConnector

"""
Function that makes a container of connections for layers using spike sequence 
(presynaptic layer of neurons and postsynaptic layer of neurons).
Parameters: PyNN.nest created spike sequence
            data (array)
            PyNN.nest input layer of neurons
            PyNN.nest output layer of neurons
"""
def build_network_connections(spikeSequence, data, inputLayer, outputLayer):
    # Get generated spike sequence for each layer
    inputSpikeSequence = spikeSequence[0]
    outputSpikeSequence = spikeSequence[1]

    # Total number of data items
    totalNumberOfDataItems = len(data)

    # Loop through all data items
    for eachWine in range(0, totalNumberOfDataItems):
        dataItem = data[eachWine]

        # === Presynaptic layer ==
        # Gets presynaptic connections for each data item
        inputConnector = generate_feature_connections(dataItem, eachWine)
        inputFromListConnector = sim.FromListConnector(inputConnector)
        sim.Projection(inputSpikeSequence, inputLayer, inputFromListConnector, 
                   receptor_type='excitatory')

        # === Postsynaptic layer ===
        outputClass = dataItem[0] - 1 # -1 so the classes are from 0, 1 and 2.
        outputConnector = [(eachWine, outputClass, FIRING_WEIGHT, TIME_STEP)] 
        outputFromListConnector = sim.FromListConnector(outputConnector)
        sim.Projection(outputSpikeSequence, outputLayer, outputFromListConnector, 
                   receptor_type='excitatory')

"""
FOR NETWORK TESTING
Function that makes a container of connections for input layer using 
generated test spike sequence.
Parameters: PyNN.nest created spike sequence
            data (array)
            PyNN.nest input layer of neurons
"""
def build_testing_connections(testSpikeSources, data, testInputLayer):
    numberDataItems = len(data)

    # Loop through all data elements (89 for wine test data)
    for element in range (0, numberDataItems):
        dataItem = data[element]

        # Get connections for each data item
        testConnector = generate_feature_connections(dataItem, element)
        testFromListConnector = sim.FromListConnector(testConnector)
        sim.Projection(testSpikeSources, testInputLayer, testFromListConnector, receptor_type='excitatory')

"""
Function that connects presynaptic layer to postsynaptic layer using STDP synapses.
Parameters: presynaptic pyNN.nest population of neurons, 
            postsynaptic pyNN.nest population of neurons
"""
def connect_layers(firstLayer, secondLayer):
    # === Parameters for STDP mechanism ===

    # === Timing dependence ===

    # Time constant of the positive part of the STDP curve, in milliseconds.
    tau_plus = 20.0
    # Time constant of the negative part of the STDP curve, in milliseconds.
    tau_minus = 20.0
    
    # Amplitude of the positive part of the STDP curve.
    A_plus = 0.006
    # Amplitude of the negative part of the STDP curve.
    A_minus = 0.0055

    # === Weight dependence ===

    # Minimum synaptic weight
    w_min = 0.0
    # Maximum synaptic weight
    w_max = 1.0
    # Default weight
    w_default = 0.0

    # Synapses to use later for testing
    global synapses
    
    # The amplitude of the weight change is independent of the current weight. 
    # If the new weight would be less than w_min it is set to w_min. If it would be greater than w_max it is set to w_max.
    stdp_synapse = sim.STDPMechanism(
        timing_dependence = sim.SpikePairRule(tau_plus = tau_plus, tau_minus = tau_minus,
                                              A_plus = A_plus, A_minus = A_minus),
        weight_dependence = sim.AdditiveWeightDependence(w_min = w_min, w_max = w_max),
        dendritic_delay_fraction = 1.0,
        weight = w_default,
        delay  = DELAY)

    # Save synapses onto a global variable
    synapses = sim.Projection(inputLayer, outputLayer, sim.AllToAllConnector(), synapse_type = stdp_synapse)

"""
FOR NETWROK TESTING
Function that connects presynaptic layer to postsynaptic using weights that
were generated using STDP synapses during training.
Parameters: presynaptic pyNN.nest population of neurons, 
            postsynaptic pyNN.nest population of neurons,
            synapses (as a list)
"""
def connect_testing_layers(inputLayer, outputLayer, synapses):
    connector = []

    for synapseOffset in range (0, len(synapses)):
        fromNeuron = synapses[synapseOffset][0]
        toNeuron = synapses[synapseOffset][1]
        weight = synapses[synapseOffset][2]
        connector = connector + [(fromNeuron, toNeuron, weight, TIME_STEP)]

    testFromListConnector = sim.FromListConnector(connector)

    sim.Projection(inputLayer, outputLayer, testFromListConnector, receptor_type = 'excitatory')

"""
Function that lets user create a layer with defined number of neurons
Neuron type: 'IF_cond_exp': Leaky integrate and fire model with fixed 
threshold and exponentially-decaying post-synaptic conductance.
Parameters: numberOfNeurons (integer), 
            label (string that represents the name of the layer)
Returns: pyNN.nest population of neurons (layer)
"""
def create_layer_of_neurons(numberOfNeurons, label):
    layer = sim.Population(numberOfNeurons, sim.IF_cond_exp, cellparams = {}, label = label)
    return layer

"""
Function that records the spikes from the layer
Parameters: pyNN.nest population of neurons (layer)
"""
def record_spikes(layer):
    layer.record({'spikes', 'v'})

"""
Function that saves the results of each layer spikes into a file
Parameters: pyNN.nest population of neurons (layer), file name (string)
"""
def save_results(layer, file):
    layer.write_data(file + '.pkl', 'spikes')

# === Main ===

# === Read the files for training and testing data ===
wineTrainingData = read_file(TRAINING_FILE)
wineTestingData = read_file(TESTING_FILE)


# === Initialize the network for training ===
initialize_network(TIME_STEP, MIN_DELAY, MAX_DELAY)

spikeSequence = create_spike_sequence(wineTrainingData, TRAINING_START_TIME, LEARNING_OFFSET)
inputLayer = create_layer_of_neurons(INPUT_LAYER_NEURONS, 'input layer')
outputLayer = create_layer_of_neurons(OUTPUT_LAYER_NEURONS, 'output layer')

build_network_connections(spikeSequence, wineTrainingData, inputLayer, outputLayer)
connect_layers(inputLayer, outputLayer)

record_spikes(inputLayer)
record_spikes(outputLayer)

sim.run(TRAINING_TIME)

save_results(inputLayer, 'results/wineInput')
save_results(outputLayer, 'results/wineOutput')

# Save the weights after training
synapseWeights = synapses.get(["weight"], format="list")

# Can print it to console to check for inactive neurons
# print(synapseWeights)

sim.reset()

# === Training is done, weights are saved and network is reset ===
# === Test the network ===
initialize_network(TIME_STEP, MIN_DELAY, MAX_DELAY)

testSpikeSequence = create_test_spike_sequence(wineTestingData, TESTING_START_TIME)
testInputLayer = create_layer_of_neurons(INPUT_LAYER_NEURONS, 'test input layer')
testOutputLayer = create_layer_of_neurons(OUTPUT_LAYER_NEURONS, 'test output layer')

build_testing_connections(testSpikeSequence, wineTestingData, testInputLayer)

# Connects layers using weights that were generated during training using STDP synapse
connect_testing_layers(testInputLayer, testOutputLayer, synapseWeights)

record_spikes(testInputLayer)
record_spikes(testOutputLayer)

sim.run(TESTING_TIME)

save_results(testInputLayer, 'results/inputWineTest')
save_results(testOutputLayer, 'results/outputWineTest')

sim.reset()
# === End of program ===