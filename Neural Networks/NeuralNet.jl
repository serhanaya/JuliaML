# This is the beginning of basic Neural Network implementation. (as of 04-01-2016).

typealias Vector{T} Array{T,1}

abstract Layer
abstract LayerIn <: Layer
abstract LayerOut <: Layer

type Neuron
    listOfWeightIn::Vector{Float64}
    listOfWeightOut::Vector{Float64}
end

type InputLayer <: LayerIn
    listOfNeurons::Vector{Neuron}
    numberOfNeuronsInLayer::Int64
end

type OutputLayer <: LayerOut
    listOfNeurons::Vector{Neuron}
    numberOfNeuronsInLayer::Int64
end

function initNeuron()
    r = rand()
    return r
end

# Initializes the input layer with pseudo random numbers.
function initLayer(iLayer::InputLayer)
    listOfWeightInTemp = Vector{Float64}(0)
    listOfNeuronsUpd = Vector{Neuron}(0)
    for i=1:iLayer.numberOfNeuronsInLayer
        neuron = Neuron(Vector{Float64}(0), Vector{Float64}(0))
        push!( listOfWeightInTemp, initNeuron() );
        neuron.listOfWeightIn = listOfWeightInTemp;
        push!( listOfNeuronsUpd, neuron )

        listOfWeightInTemp = Vector{Float64}(0)
    end

    iLayer.listOfNeurons = listOfNeuronsUpd

    return iLayer
end

# Initializes the output layer with pseudo random numbers.
function initLayer(oLayer::OutputLayer)
    listOfWeightOutTemp = Vector{Float64}(0)
    listOfNeuronsUpd = Vector{Neuron}(0)
    for i=1:oLayer.numberOfNeuronsInLayer
        neuron = Neuron(Vector{Float64}(0), Vector{Float64}(0))
        push!( listOfWeightOutTemp, initNeuron() );
        neuron.listOfWeightOut = listOfWeightOutTemp;
        push!( listOfNeuronsUpd, neuron )

        listOfWeightOutTemp = Vector{Float64}(0)
    end

    oLayer.listOfNeurons = listOfNeuronsUpd

    return oLayer
end
