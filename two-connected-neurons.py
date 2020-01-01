import pylab
import nest

#Create integrate and fire neuron
neuron1 = nest.Create("iaf_psc_alpha")
nest.SetStatus(neuron1, {"I_e" : 376.0})

#Create second iaf neuron
neuron2 = nest.Create("iaf_psc_alpha")

#Create multimeter
multimeter = nest.Create("multimeter")
nest.SetStatus(multimeter, {"withtime" : True, "record_from" : ["V_m"]})

#Connect neuron1 to neuron2 and record the membrane potentail from neuron2
nest.Connect(neuron1, neuron2, syn_spec = {"weight" : 20.0, "delay" : 1.0})
nest.Connect(multimeter, neuron2)

#Simulate for a set period of time (ms)
nest.Simulate(1000.0)

#After simulation is finished, obtain the data recorder by the multimeter
dmm = nest.GetStatus(multimeter)[0]
Vms = dmm["events"]["V_m"]
ts = dmm["events"]["times"]

#Open a figure and produce the plot
pylab.figure(1)
pylab.plot(ts, Vms)
pylab.show()
