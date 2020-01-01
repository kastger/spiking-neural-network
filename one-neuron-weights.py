from sklearn.svm import LinearSVC
from scipy.special import erf

import pylab
import nest

neuron = nest.Create("iaf_psc_alpha")
neuron2 = nest.Create("iaf_psc_alpha")

nest.GetStatus(neuron)

#Constant background current
nest.GetStatus(neuron, "I_e")

#Values of the reset potential and threshold of the neuron
nest.GetStatus(neuron, ["V_reset", "V_th"])

#Set background current to 376 to cause neuron to spike periodically
nest.SetStatus(neuron, {"I_e" : 0.0})

#Neuron receives 2 Poisson spike trains, one excitatory and the other inhibitory
noise_ex = nest.Create("poisson_generator")
noise_in = nest.Create("poisson_generator")

#Set rates (Hz)
nest.SetStatus(noise_ex, {"rate": 80000.0})
nest.SetStatus(noise_in, {"rate": 15000.0})

#Set the weights
#Postsynaptic current
syn_dict_ex = {"weight": 1.2}

#Presynaptic current
syn_dict_in = {"weight": -2.0}

#Pass the synaptic weights in a dictionary
nest.Connect(noise_ex, neuron, syn_spec=syn_dict_ex)
nest.Connect(noise_in, neuron, syn_spec=syn_dict_in)

#Device thats used to record the membrane voltage of neuron over time
multimeter = nest.Create("multimeter")
nest.SetStatus(multimeter, {"withtime" : True, "record_from" : ["V_m"]})

#Create spikedetector that records the spiking events produced by a neuron
#withgid indicates spike detector to record the id of the neuron
spikedetector = nest.Create("spike_detector",
                params = {"withgid" : True, "withtime" : True})

#Connect individual nodes to form a small network
nest.Connect(multimeter, neuron)
nest.Connect(neuron, spikedetector)

#Simulate for a set period of time (ms)
nest.Simulate(1000.0)

#After simulation is finished, obtain the data recorder by the multimeter
dmm = nest.GetStatus(multimeter)[0]
Vms = dmm["events"]["V_m"]
ts = dmm["events"]["times"]

#Open a figure and produce the plot
pylab.figure(1)
pylab.plot(ts, Vms)

#Extracts the dictionary elements with the key "events" which was used earlier
dSD = nest.GetStatus(spikedetector,keys="events")[0]
evs = dSD["senders"]
ts = dSD["times"]
pylab.figure(2)
pylab.plot(ts, evs, ".")
pylab.show()
