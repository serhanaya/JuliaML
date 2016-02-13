"""
Basic Implementation of Neural Networks for Julia. 

* Neural Network Programming with Java, Packt Publishing, Jan 2016.
"""

module Nnet

include("layer.jl")
include("neuralNet.jl")
include("neuralNetTest.jl")
include("train/backpropagation.jl")
include("train/training.jl")
include("som/kohonen.jl")

end
