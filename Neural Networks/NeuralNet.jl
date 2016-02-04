# This is the beginning of basic Neural Network implementation. (as of 04-01-2016).

typealias Vector{T} Array{T,1}

abstract Layer
abstract LayerIn <: Layer
abstract LayerOut <: Layer
abstract LayerHidden <: Layer

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

type HiddenLayer <: LayerHidden
    listOfNeurons::Vector{Neuron}
    numberOfHiddenLayers::Int64
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


function printLayer(iLayer::InputLayer)
    println("### INPUT LAYER ###")
    n = 1
    for neuron::Neuron in iLayer.listOfNeurons
        println("Neuron #" + n + ":")
        println("Input Weights:")
        println(neuron.listOfWeightIn)
        n += 1
    end
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


function printLayer(oLayer::OutputLayer)
    println("### OUTPUT LAYER ###")
    n = 1
    for neuron::Neuron in oLayer.listOfNeurons
        println("Neuron #" + n + ":")
        println("Output Weights:")
        println(neuron.listOfWeightOut)
        n += 1
    end
end


# Initializes the hidden layer with pseudo random numbers.
function initLayer(hLayer::HiddenLayer, iLayer::InputLayer,
                   oLayer::OutputLayer, listOfHiddenLayer::HiddenLayer)
    listOfWeightInTemp = Vector{Float64}(0)
    listOfWeightOutTemp = Vector{Float64}(0)
    listOfNeuronsUpdate = Vector{Float64}(0)

    numberOfNeuronsInLayer = length(listOfHiddenLayer)

    for i=1:hLayer.numberOfHiddenLayers
        for j=1:hLayer.numberOfNeuronsInLayer
            neuron = Neuron(Vector{Float64}(0), Vector{Float64}(0))

            if (i == 0)
                if hLayer.numberOfHiddenLayers > 1
                limitOut = listOfHiddenLayer[i+1].numberOfNeuronsInLayer
                else
                limitOut = listOfHiddenLayer[i].numberOfNeuronsInLayer
                end
            elseif (i == hLayer.numberOfHiddenLayers - 1)
                limitIn = listOfHiddenLayer[i-1]
                limitOut = oLayer.numberOfNeuronsInLayer
            else
                limitIn = listOfHiddenLayer[i-1].numberOfNeuronsInLayer
                limitOut = listOfHiddenLayer[i+1].numberOfNeuronsInLayer
            end

            for k=1:limitIn
                push!( listOfWeightInTemp, initNeuron() )
            end

            for k=1:limitOut
                push!( listOfWeightOutTemp, initNeuron() )
            end

            iLayer.listOfWeightIn = listOfWeightInTemp
            oLayer.listOfWeightOut = listOfWeightOutTemp

            push!( listOfNeuronsUpdate, neuron )

            listOfWeightInTemp = Vector{Float64}(0)
            listOfWeightOutTemp = Vector{Float64}(0)

        end

        listOfHiddenLayer[i].listOfNeurons  = listOfNeuronsUpdate
        listOfNeuronsUpdate = Vector{Float64}(0)

    end

    return listOfHiddenLayer

end


function printLayer(listOfHiddenLayer::HiddenLayer)
    if size(listOfHiddenLayer) > 0
        println("### HIDDEN LAYER ###")
        h = 1
        for hiddenLayer::HiddenLayer in listOfHiddenLayer
            println("Hidden Layer #" + h)
            n = 1
            for neuron in hiddenLayer.listOfNeurons
                println("Neuron #" + n);
                println("Input Weights:");
                println(neuron.listOfWeightIn)
                println("Output Weights:");
                println(neuron.listOfWeightOut)
                n += 1
            end
            h += 1
        end
    end
end