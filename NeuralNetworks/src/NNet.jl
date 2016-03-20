"""
Basic Implementation of Neural Networks for Julia.

* Neural Network Programming with Java, Packt Publishing, Jan 2016.
"""

module NNet

include("layer.jl")
include("neuralNet.jl")
include("neuralNetTest.jl")
include("training.jl")
include("backpropagation.jl")
include("levenbergMarquardt.jl")
include("kohonen.jl")

export Neuron, Layer, InputLayer, OutputLayer, HiddenLayer, initNeuron, printLayer,

       NeuralNet, initNet, printNet, trainNet, printTrainedNetResult, netValidation,

       testPerceptron, testAdaline, testLMA, testKohonen,

       Neuron, Layer, InputLayer, OutputLayer, HiddenLayer, initNeuron, printLayer
end
