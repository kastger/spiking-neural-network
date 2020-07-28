"""
FILES IN PACKAGE:
snn-stdp.py
get-networks-accuracy.py
read-pkl-spikes.py
iris-train.txt
iris-test.txt
README.md

Files generated after training:
results/input.pkl
results/output.pkl

Files generated after testing:
results/inputTest.pkl
results/outputTest.pkl

Author: Geraldas Kastauna
Supervisor: Professor Dr. Chris Huyck (https://www.cwa.mdx.ac.uk/chris/chrisroot.html)
Email: GK468@live.mdx.ac.uk OR geraldaskastauna@gmail.com

Middlesex University, London Campus
Undergradute Student of Computer Science BSc

Thesis: 'Learing with Spiking Neural Network'

About this program:
This program reads the iris training data (which was randomized) of 75 irises and
creates a spiking neural network that uses spike-timing-dependent plasticity (STDP)
synapses for training. It generates a spike sequence for presynaptic and
postsynaptic layers which are used during network training period. After training
the connections between neurons are saved and used for testing on a static synapse 
model that has connections with fixed weights and delay.
"""

# === Dependencies ===
import csv
import pyNN.nest as sim

# === Parameters for spiking neural network ===
# === Text files ===
TRAINING_FILE = "iris-train.txt"
TESTING_FILE = "iris-test.txt"

# === Simulation time and epochs ===
TRAINING_TIME = 8000.0 #ms
TESTING_TIME = 4000.0 #ms
FIRING_PERIOD = 27.0 #ms

TRAINING_EPOCHS = 5

TRAINING_START_TIME = 10.0 #ms (error when 0)
TESTING_START_TIME = 10.0 #ms

# === Number of neurons for each layer ===
INPUT_LAYER_NEURONS = 104
OUTPUT_LAYER_NEURONS = 3

NEURONS_PER_FEATURE = 26

numberOfIrisClasses = 3 # iris-setosa, iris-versicolour, iris-virginica
synapses = 0 # Used to save weights during training

# === Variables for network initialization ===
TIME_STEP = 0.1 #ms
MIN_DELAY = 0.1 #ms
MAX_DELAY = 0.1 #ms
LEARNING_OFFSET = 1.0

# === Firing weight for connections ===
FIRING_WEIGHT = 0.1

# === Functions ===
"""
Function that reads the data from the file and converts each iris feature
to an integer value between 0 and 100.
Parameters: string (file name)
Returns an array
"""
def read_file(fileName):
    data = []
    # Open file using CSV file reader
    dataFileHandle = list(csv.reader(open(fileName)))
    totalIrises = len(dataFileHandle)

    # Loop through all irises
    for lineNumber in range (0, totalIrises):
        # Sepal length feature
        sepal_length = float(dataFileHandle[lineNumber][0])
        sepal_length = int(sepal_length * 10)

        # Sepal width feature
        sepal_width = float(dataFileHandle[lineNumber][1])
        sepal_width = int(sepal_width * 10)

        # Petal length feature
        petal_length = float(dataFileHandle[lineNumber][2])
        petal_length = int(petal_length * 10)

        # Petal width feature
        petal_width = float(dataFileHandle[lineNumber][3])
        petal_width = int(petal_width * 10)

        # Class
        iris_class = int(dataFileHandle[lineNumber][4])

        # Add each line to data array
        dataLine = [sepal_length, sepal_width, petal_length, petal_width, iris_class]
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

    # Loop through all data elements (75 for iris test data)
    for element in range (0, numberDataItems):
        # Calculate the test timing of spikes for each element and store them into testSpikeTimesSequnece
        testSpikeTimes = [[startTime + (element * FIRING_PERIOD)]]
        testSpikeTimesSequence = testSpikeTimesSequence + testSpikeTimes

    # Generate spikes for testing with previously stored spike times
    generatedTestSpikes = sim.Population(numberDataItems, sim.SpikeSourceArray,
                {'spike_times': testSpikeTimesSequence}, label="test spike sequence")

    return generatedTestSpikes

"""
Function that calculates which neurons should be simulated for each iris data 
feature and saves the connections in an array for later use.
Parameters: dataItem (array which has whole iris line with all 4 features and a class)
            irisNumberOnTheList (integer that represents iris number on the list)
Returns: an array of neuron connections
"""
def generate_feature_connections(dataItem, irisNumberOnTheList):
    inputConnector = []

    # Loop through all iris features
    for dataFeature in range (0,4):
        dataFeatureValue = dataItem[dataFeature]

        # Loop through neurons in range that depends on feature numerical value
        for neuron in range((dataFeatureValue - numberOfIrisClasses), (dataFeatureValue + numberOfIrisClasses + 1)):
            if ((neuron >= 0) and (neuron < NEURONS_PER_FEATURE)):
                toNeuron = neuron + (dataFeature * NEURONS_PER_FEATURE)
                inputConnector = inputConnector + [(irisNumberOnTheList, toNeuron, FIRING_WEIGHT, TIME_STEP)]

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
    for eachIris in range(0, totalNumberOfDataItems):
        dataItem = data[eachIris]

        # === Presynaptic layer ==
        # Gets presynaptic connections for each data item
        inputConnector = generate_feature_connections(dataItem, eachIris)
        inputFromListConnector = sim.FromListConnector(inputConnector)
        sim.Projection(inputSpikeSequence, inputLayer, inputFromListConnector, 
                   receptor_type='excitatory')

        # === Postsynaptic layer ===
        outputClass = dataItem[4] - 1 # -1 so the classes are from 0, 1 and 2.
        outputConnector = [(eachIris, outputClass, FIRING_WEIGHT, TIME_STEP)] 
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

    # Loop through all data elements (75 for iris test data)
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
    w_max = 0.1
    # Default weight
    w_default = 0.0

    # === Delay ===
    delay = 0.1

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
        delay  = delay)

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
irisTrainingData = read_file(TRAINING_FILE)
irisTestingData = read_file(TESTING_FILE)

# === Initialize the network for training ===
initialize_network(TIME_STEP, MIN_DELAY, MAX_DELAY)

spikeSequence = create_spike_sequence(irisTrainingData, TRAINING_START_TIME, LEARNING_OFFSET)
inputLayer = create_layer_of_neurons(INPUT_LAYER_NEURONS, 'input layer')
outputLayer = create_layer_of_neurons(OUTPUT_LAYER_NEURONS, 'output layer')

build_network_connections(spikeSequence, irisTrainingData, inputLayer, outputLayer)
connect_layers(inputLayer, outputLayer)

record_spikes(inputLayer)
record_spikes(outputLayer)

sim.run(TRAINING_TIME)

save_results(inputLayer, 'results/input')
save_results(outputLayer, 'results/output')

# Save the weights after training
synapseWeights = synapses.get(["weight"], format="list")

# Can print it to console to check for inactive neurons
#print(synapseWeights)

sim.reset()
# === Training is done, weights are saved and network is reset ===
# === Test the network ===
initialize_network(TIME_STEP, MIN_DELAY, MAX_DELAY)

testSpikeSequence = create_test_spike_sequence(irisTestingData, TESTING_START_TIME)
testInputLayer = create_layer_of_neurons(INPUT_LAYER_NEURONS, 'test input layer')
testOutputLayer = create_layer_of_neurons(OUTPUT_LAYER_NEURONS, 'test output layer')

build_testing_connections(testSpikeSequence, irisTestingData, testInputLayer)

# Connects layers using weights that were generated during training using STDP synapse
connect_testing_layers(testInputLayer, testOutputLayer, synapseWeights)

record_spikes(testInputLayer)
record_spikes(testOutputLayer)

sim.run(TESTING_TIME)

save_results(testInputLayer, 'results/inputTest')
save_results(testOutputLayer, 'results/outputTest')

sim.reset()
# === End of program ===