# import learn.Adaline
# import learn.Perceptron
# import learn.Training.ActivationFncENUM
# import learn.Training.TrainingTypesENUM;

typealias Vector{T} Array{T,1}
typealias Matrix{T} Array{T,2}

# ActivationFncENUM

# TrainingTypesENUM


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

#    activationFnc::ActivationFncENUM
#    activationFncOutputLayer::ActivationFncENUM
#    trainType::TrainingTypesENUM
#    function NeuralNet(...) =
#        new(listOfMSE = Vector{Float64}(0)...)
end

function initNet(numberOfInputNeurons::Int64, numberOfHiddenLayers::Int64,
    numberOfNeuronsInHiddenLayer::Int64, numberOfOutputNeurons::Int64)

    iLayer = InputLayer(listOfNeurons = Vector{Neuron}(0),
                        numberOfNeuronsInLayer = numberOfInputNeurons + 1)
    listOfHiddenLayer = Vector{HiddenLayer}(0)

    for i = 1:numberOfHiddenLayers
        hLayer::HiddenLayer
        hLayer.numberOfNeuronsInLayer = numberOfNeuronsInHiddenLayer
        push!( listOfHiddenLayer, hLayer )
    end

    oLayer.numberOfNeuronsInLayer = numberOfOutputNeurons
    iLayer = initLayer(iLayer) # import Layer.initLayer method.

    if numberOfHiddenLayers > 0
        listOfHiddenLayer = initLayer(hLayer,  iLayer, oLayer,
                                      listOfHiddenLayer) # import Layer.initLayer method.
    end

    oLayer = initLayer(oLayer)

    newNet = NeuralNet(iLayer, hLayer, listOfHiddenLayer, 
                       numberOfHiddenLayers, oLayer)  # Create Constructor for this situation!!
end

function printNet(n::NeuralNet)
    printLayer(n.iLayer)  # import Layer.printLayer method!
    println()
    if (n.hLayer != nothing) # Check if null == nothing in Julia
        printLayer(n.listOfHiddenLayer)
        println()
    end
    printLayer(n.oLayer)
end

function trainNet(n::NeuralNet)
    if n.trainType == "PERCEPTRON"          
        return Perceptron.train(n)  # Import Perceptron.
    elseif n.trainType == "ADALINE"
        return Adaline.train(n)  # Import Adaline.
    else
        throw(Error) # Throwing exceptions!
    end
end


function printTrainedResult(n::NeuralNet)
    if n.trainType == "PERCEPTRON"  
        Perceptron.printTrainedResult(n)  # Import Perceptron. 
        break
    elseif n.trainType == "ADALINE"
        Adaline.printTrainedResult(n) 
        break
    else
        throw(IllegalArgument) # Throwing exceptions!
    end
end
