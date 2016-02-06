# import learn.Adaline
# import learn.Perceptron
# import learn.Training.ActivationFncENUM
# import learn.Training.TrainingTypesENUM;

# Parametric type definition of a Vector as an one-dimensional 
# array, and a matrix as a two dimensional array. 
typealias Vector{T} Array{T,1}
typealias Matrix{T} Array{T,2}

@enum TrainingTypesENUM PERCEPTRON ADALINE
@enum ActivationFncENUM STEP LINEAR SIGLOG HYPERTAN

type NeuralNet
    iLayer::InputLayer
    hLayer::HiddenLayer
    listOfHiddenLayer::Vector{HiddenLayer}
    oLayer::OutputLayer
    numberOfHiddenLayers::Int64

    trainSet::Matrix{Float64}
    validationSet::Matrix{Float64}
    realOutputSet::Vector{Float64}
    realMatrixOutputSet::Matrix{Float64}

    maxEpochs::Int64
    learningRate::Float64
    targetError::Float64
    trainingError::Float64
    errorMean::Float64
    listOfMSE::Vector{Float64}


    activationFncType::ActivationFncENUM
    activationFncTypeOutputLayer::ActivationFncENUM
    trainType::TrainingTypesENUM
#    function NeuralNet(...) =
#        new(listOfMSE = Vector{Float64}(0)...)
end

function initNet(numberOfInputNeurons::Int64, numberOfHiddenLayers::Int64,
    numberOfNeuronsInHiddenLayer::Int64, numberOfOutputNeurons::Int64)

    inputLayer = InputLayer(listOfNeurons = Vector{Neuron}(0),
                        numberOfNeuronsInLayer = numberOfInputNeurons + 1)
    listOfHiddenLayer = Vector{HiddenLayer}(0)

    for i = 1:numberOfHiddenLayers
        hiddenLayer = HiddenLayer()
        hiddenLayer.numberOfNeuronsInLayer = numberOfNeuronsInHiddenLayer
        push!( listOfHiddenLayer, hiddenLayer )
    end

    outputLayer.numberOfNeuronsInLayer = numberOfOutputNeurons
    inputLayer = initLayer(inputLayer) # import Layer.initLayer method.

    if numberOfHiddenLayers > 0
        listOfHiddenLayer = initLayer(hiddenLayer,  inputLayer, outputLayer,
                                      listOfHiddenLayer) # import Layer.initLayer method.
    end

    ouputLayer = initLayer(ouputLayer) # import Layer.initLayer method.

    newNet = NeuralNet(inputLayer, hiddenLayer, listOfHiddenLayer, 
                       numberOfHiddenLayers, outputLayer)  # Create Constructor for this situation!!
    return newNet
end

function printNet(n::NeuralNet)
    printLayer(n.iLayer)  # import Layer.printLayer method!
    println()
    if isdefined(n.hLayer) != false # n.HiddenLayer == null ?
        printLayer(n.listOfHiddenLayer)
        println()
    end
    printLayer(n.oLayer)
end
 
function trainNet(n::NeuralNet)
    if n.trainType == PERCEPTRON          
        return Perceptron.train(n)  # Import Perceptron.
    elseif n.trainType == ADALINE
        return Adaline.train(n)  # Import Adaline.
    else
        throw(ArgumentError(n.trainType+" does not exist in TrainingTypesENUM"))
    end
end


function printTrainedResult(n::NeuralNet)
    if n.trainType == PERCEPTRON 
        Perceptron.printTrainedResult(n)  # Import Perceptron. 
        break
    elseif n.trainType == ADALINE
        Adaline.printTrainedResult(n) 
        break
    else
        throw(ArgumentError(n.trainType+" does not exist in TrainingTypesENUM"))
    end
end

