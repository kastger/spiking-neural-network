# Dependencies
import _pickle as pickle

# Training spikes
TRAINING_INPUT_SPIKES_FILE = 'results/input.pkl'
TRAINING_OUTPUT_SPIKES_FILE = 'results/output.pkl'

# Testing spikes
TESTING_INPUT_SPIKES_FILE = 'results/inputTest.pkl'
TESTING_OUTPUT_SPIKES_FILE = 'results/outputTest.pkl'

"""
Functions that prints spike sequences from .pkl file to console
Arguemnts: file name (string)
"""
def print_pkl_spikes(fileName):
        fileHandler = open(fileName, 'rb') # Opens a file for reading only in binary format
        neoObject = pickle.load(fileHandler, encoding = 'utf8') 
        segments = neoObject.segments
        segment = segments[0]
        spikeSequence = segment.spiketrains
        neurons = len(spikeSequence)

        # Loop through active neurons
        for neuronNum in range (0,neurons):
            if (len(spikeSequence[neuronNum])>0):
                spikes = spikeSequence[neuronNum]
                for spike in range (0,len(spikes)):
                    print (neuronNum, spikes[spike])
        fileHandler.close()

# === Main =====================================================================
# Delete or add # (hastag) infront of the line to print neuron and spike times to console

# === Training files ===
#print_pkl_spikes(TRAINING_INPUT_SPIKES_FILE)
#print_pkl_spikes(TRAINING_OUTPUT_SPIKES_FILE)
    
# === Testing files ===
#print_pkl_spikes(TESTING_INPUT_SPIKES_FILE)
print_pkl_spikes(TESTING_OUTPUT_SPIKES_FILE)