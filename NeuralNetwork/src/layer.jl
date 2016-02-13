# A neural network consist of several layers which are basically groups of neurons.

# A vector is a one-dimensional array.
typealias Vector{T} Array{T,1}
typealias Matrix{T} Array{T,2}

# Layer hierarchy.
abstract Layer

# Define a neuron.
type Neuron
    listOfWeightIn::Vector{Float64}
    listOfWeightOut::Vector{Float64}
    outputValue::Float64
    error::Float64
    sensibility::Float64
    function Neuron(;listOfWeightIn=Vector{Float64}(0),
        listOfWeightOut=Vector{Float64}(0), outputValue=0.0, error=0.0,
        sensibility=0.0)
        new(listOfWeightIn, listOfWeightOut, outputValue, error, sensibility)
    end
end

# Define InputLayer.
type InputLayer <: Layer
    listOfNeurons::Vector{Neuron}
    numberOfNeuronsInLayer::Int
    function InputLayer(;listOfNeurons=Vector{Neuron}(0),
        numberOfNeuronsInLayer=0)
        new(listOfNeurons, numberOfNeuronsInLayer)
    end
end

# Define OutputLayer.
type OutputLayer <: Layer
    listOfNeurons::Vector{Neuron}
    numberOfNeuronsInLayer::Int
    function OutputLayer(;listOfNeurons=Vector{Neuron}(0),
        numberOfNeuronsInLayer=0)
        new(listOfNeurons, numberOfNeuronsInLayer)
    end
end

# Define HiddenLayer.
type HiddenLayer <: Layer
    listOfNeurons::Vector{Neuron}
    numberOfNeuronsInLayer::Int
    function HiddenLayer(;listOfNeurons=Vector{Neuron}(0),
        numberOfNeuronsInLayer=0)
        new(listOfNeurons, numberOfNeuronsInLayer)
    end
end

# Generate a pseudo random number.
function initNeuron()
    r = rand()
    return r
end

# Initializes the input layer with pseudo random numbers.
function initLayer(inputLayer::InputLayer)
    listOfWeightIn = Vector{Float64}(0)
    listOfNeurons = Vector{Neuron}(0)
    for i=1:inputLayer.numberOfNeuronsInLayer
        neuron = Neuron()
        push!( listOfWeightIn, initNeuron() );
        neuron.listOfWeightIn = listOfWeightIn;
        push!( listOfNeurons, neuron )

        listOfWeightIn = Vector{Float64}(0)
    end

    inputLayer.listOfNeurons = listOfNeurons

    return inputLayer
end

# Print InputLayer properties.
function printLayer(inputLayer::InputLayer)
    println("### INPUT LAYER ###")
    n = 1
    for neuron::Neuron in inputLayer.listOfNeurons
        println("Neuron # ", n, ":")
        println("Input Weights:")
        println(neuron.listOfWeightIn)
        n += 1
    end
end


# Initializes the output layer with pseudo random numbers.
function initLayer(outputLayer::OutputLayer)
    listOfWeightOut = Vector{Float64}(0)
    listOfNeurons = Vector{Neuron}(0)
    for i=1:outputLayer.numberOfNeuronsInLayer
        neuron = Neuron()
        push!( listOfWeightOut, initNeuron() );
        neuron.listOfWeightOut = listOfWeightOut;
        push!( listOfNeurons, neuron )

        listOfWeightOut = Vector{Float64}(0)
    end

    outputLayer.listOfNeurons = listOfNeurons

    return outputLayer
end

# Print OutputLayer properties.
function printLayer(outputLayer::OutputLayer)
    println("### OUTPUT LAYER ###")
    n = 1
    for neuron::Neuron in outputLayer.listOfNeurons
        println("Neuron #", n, " :")
        println("Output Weights:")
        println(neuron.listOfWeightOut)
        n += 1
    end
end

# Initializes the hidden layer with pseudo random numbers.
function initLayer(hiddenLayer::HiddenLayer, inputLayer::InputLayer,
                   outputLayer::OutputLayer, listOfHiddenLayer::Vector{HiddenLayer})
    listOfWeightIn = Vector{Float64}(0)
    listOfWeightOut = Vector{Float64}(0)
    listOfNeurons = Vector{Float64}(0)

    numberOfHiddenLayers = size(listOfHiddenLayer)[1]

    for hdn_i=1:numberOfHiddenLayers
        for neuron_i=1:hiddenLayer.numberOfNeuronsInLayer
            neuron = Neuron()

            limitIn = 0
            limitOut = 0

            if (hdn_i == 1)  # First
                limitIn = inputLayer.numberOfNeuronsInLayer
                if numberOfHiddenLayers > 1
                    limitOut = listOfHiddenLayer[hdn_i+1].numberOfNeuronsInLayer
                elseif numberOfHiddenLayers == 1
                    limitOut = outputLayer.numberOfNeuronsInLayer
                end
            elseif (hdn_i == numberOfHiddenLayers)  # Last
                limitIn = listOfHiddenLayer[hdn_i-1].numberOfNeuronsInLayer
                limitOut = outputLayer.numberOfNeuronsInLayer
            else  # Middle
                limitIn = listOfHiddenLayer[hdn_i-1].numberOfNeuronsInLayer
                limitOut = listOfHiddenLayer[hdn_i+1].numberOfNeuronsInLayer
            end

            limitIn = limitIn - 1  # bias not connected
            limitOut = limitOut - 1  # bias not connected

            if neuron_i >= 1  # bias has no input
                for k=1:(limitIn + 1)
                push!( listOfWeightIn, initNeuron() )
                # push!( listOfWeightIn, initNeuron(k, neuron_i, 1) )
                end
            end

            for k=1:(limitOut + 1)
                push!( listOfWeightOut, initNeuron() )
                # push!( listOfWeightOut, initNeuron(k, neuron_i, 2) )
            end

            neuron.listOfWeightIn = listOfWeightIn
            neuron.listOfWeightOut = listOfWeightOut
            push!( listOfNeurons, neuron )

            listOfWeightIn = Vector{Float64}(0)
            listOfWeightOut = Vector{Float64}(0)

        end

        listOfHiddenLayer[i].listOfNeurons  = listOfNeurons
        listOfNeurons = Vector{Float64}(0)

    end

    return listOfHiddenLayer

end

# Print HiddenLayer properties.
function printLayer(listOfHiddenLayer::Vector{HiddenLayer})
    if size(listOfHiddenLayer)[1] > 0
        println("### HIDDEN LAYER ###")
        h = 1
        for hiddenLayer::HiddenLayer in listOfHiddenLayer
            println("Hidden Layer # ", h)
            n = 1
            for neuron::Neuron in hiddenLayer.listOfNeurons
                println("Neuron # ", n);
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


