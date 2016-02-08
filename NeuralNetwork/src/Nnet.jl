module Nnet

export NeuralNet

include("layer.jl")
include("nnetTest.jl")

@enum TrainingTypesENUM PERCEPTRON ADALINE
@enum ActivationFncENUM STEP LINEAR SIGLOG HYPERTAN

type NeuralNet

    inputLayer::InputLayer
    hiddenLayer::HiddenLayer
    listOfHiddenLayer::Vector{HiddenLayer}  
    numberOfHiddenLayers::Int64
    outputLayer::OutputLayer

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

    function NeuralNet(inputLayer, hiddenLayer, listOfHiddenLayer, numberOfHiddenLayers, 
            outputLayer; trainSet=Matrix{Float64}(0,0), validationSet=Matrix{Float64}(0,0),
            realOutputSet=Vector{Float64}(0), realMatrixOutputSet=Matrix{Float64}(0,0), 
            maxEpochs=0, learningRate=0, targetError=0, trainingError=0, errorMean=0,
            listOfMSE=Vector{Float64}(0), activationFncType=STEP, 
            activationFncTypeOutputLayer=STEP, trainType=PERCEPTRON)

        new(inputLayer, hiddenLayer, listOfHiddenLayer, numberOfHiddenLayers, 
            outputLayer, trainSet, validationSet, realOutputSet, realMatrixOutputSet,
            maxEpochs, learningRate, targetError, trainingError, errorMean, listOfMSE,
            activationFncType, activationFncTypeOutputLayer, trainType)
    end
end

function initNet(numberOfInputNeurons::Int64, numberOfHiddenLayers::Int64,
    numberOfNeuronsInHiddenLayer::Int64, numberOfOutputNeurons::Int64)

    inputLayer = InputLayer(listOfNeurons = Vector{Neuron}(0),
                        numberOfNeuronsInLayer = numberOfInputNeurons + 1)  # Bias
    
    hiddenLayer = HiddenLayer() # If there is no hidden layer.
    listOfHiddenLayer = Vector{HiddenLayer}(0)

    for i = 1:numberOfHiddenLayers
        hiddenLayer = HiddenLayer(numberOfNeuronsInLayer=numberOfNeuronsInHiddenLayer)
        push!( listOfHiddenLayer, hiddenLayer )
    end

    outputLayer = OutputLayer(numberOfNeuronsInLayer=numberOfOutputNeurons)
    inputLayer = initLayer(inputLayer) 

    if numberOfHiddenLayers > 0
        listOfHiddenLayer = initLayer(hiddenLayer,  inputLayer, outputLayer,
                                      listOfHiddenLayer)
    end

    outputLayer = initLayer(outputLayer) 

    newNet = NeuralNet(inputLayer, hiddenLayer, listOfHiddenLayer,
                       numberOfHiddenLayers, outputLayer)  # Create Constructor for this situation!!
    return newNet
end

function printNet(n::NeuralNet)
    printLayer(n.inputLayer)  # import Layer.printLayer
    println()
     # if n.HiddenLayer != null
    printLayer(n.listOfHiddenLayer)
    println()
    printLayer(n.outputLayer)
end

typealias Vector{T} Array{T,1}


function train(n::NeuralNet)
    inputWeightIn = Vector{Float64}(0)
    rows = size(n.trainSet)[1]
    cols = size(n.trainSet)[2]
    epochs = 0
    error = 0.0
    mse = 0.0

    while epochs < n.maxEpochs
        estimatedOutput = 0.0
        realOutput = 0.0

        for i = 1:rows
            netValue = 0.0

            for j = 1:cols
                inputWeightIn = n.inputLayer.listOfNeurons[j].listOfWeightIn
                inputWeight = inputWeightIn[1]
                netValue = netValue + inputWeight * n.trainSet[i,j]
            end

            estimatedOutput = activationFnc(n.activationFncType, netValue)
            realOutput = n.realOutputSet[i]
            error = realOutput - estimatedOutput

            if abs(error) > n.targetError
                # fix weights
                inputLayerTemp =
                    InputLayer(listOfNeurons = teachNeuronsOfLayer(cols, i,
                        n, netValue))  # Fix constructor of InputLayer.
                n.inputLayer = inputLayerTemp
            end
        end

        mse = (realOutput - estimatedOutput)^2
        push!(n.listOfMSE, mse)
        epochs += 1
    end

    n.trainingError = error

    return n
end


function teachNeuronsOfLayer(numberOfInputNeurons::Int64, line::Int64,
        n::NeuralNet, netValue::Float64)
    listOfNeuronsUpdate = Vector{Neuron}(0)
    inputWeightsInNew = Vector{Float64}(0)
    inputWeightsInOld = Vector{Float64}(0)
    error = 0.0

    for j = 1:numberOfInputNeurons
        inputWeightsInOld = n.inputLayer.listOfNeurons[j].listOfWeightIn
        inputWeightOld = inputWeightsInOld[1]

        push!( inputWeightsInNew, calcNewWeight(n.trainType,
            inputWeightOld, n, error, n.trainSet[line,j], netValue) )

        neuron = Neuron()  # Constructor needed.
        neuron.listOfWeightIn = inputWeightsInNew
        push!( listOfNeuronsUpdate, neuron )
        inputWeightsInNew = Vector{Float64}(0)
    end

    return listOfNeuronsUpdate
end


function calcNewWeight(trainType::TrainingTypesENUM, inputWeightOld::Float64,
    n::NeuralNet, error::Float64, trainSample::Float64, netValue::Float64)

    if trainType == TrainingTypesENUM(PERCEPTRON)
        return inputWeightOld + n.learningRate * error * trainSample
    elseif TrainingTypesENUM(ADALINE)
        return inputWeightOld * n.learningRate * error * trainSample *
             derivativeActivationFnc(n.activationFncType, netValue)
    else
        throw(ArgumentError(trainType
                    + " does not exist in TrainingTypesENUM"))
    end
end

function activationFnc(fnc::ActivationFncENUM, value::Float64)

    if fnc == ActivationFncENUM(STEP)
        if value >= 0
            return 1.0
        else
            return 0.0
        end
        return value

    elseif fnc == ActivationFncENUM(LINEAR)
        return value

    elseif fnc == ActivationFncENUM(SIGLOG)
        return 1.0 / (1.0 + exp(-value))

    elseif fnc == ActivationFncENUM(HYPERTAN)   
        return tanh(value)

    else
        throw(ArgumentError(fnc
                    + " does not exist in ActivationFncENUM"))

    end
end

function derivativeActivationFnc(fnc::ActivationFncENUM, value::Float64)

    if fnc == ActivationFncENUM(LINEAR)
        return 1.0

    elseif fnc == ActivationFncENUM(SIGLOG)
        return value * (1.0 - value);

    elseif fnc == ActivationFncENUM(HYPERTAN)
        return 1.0 / cosh(value)^2

    else
        throw(ArgumentError(fnc
                    + " does not exist in ActivationFncENUM"))

    end
end

function printTrainedNetResult(trainedNet::NeuralNet)
    rows = size(trainedNet.trainSet)[1]
    cols = size(trainedNet.trainSet)[2]

    inputWeightIn = Vector{Float64}(0)

    for i = 1:rows
        netValue = 0.0
        for j = 1:cols
            inputWeightIn = trainedNet.inputLayer.listOfNeurons[j].listOfWeightIn
            inputWeight = inputWeightIn[1]
            netValue = netValue + inputWeight * trainedNet.trainSet[i,j]
            print(trainedNet.trainSet[i,j], "\t")  
        end

        estimatedOutput = activationFnc(trainedNet.activationFncType, netValue)
        
        print(" NET OUTPUT: ", estimatedOutput, "\t");  
        print(" REAL OUTPUT: ", trainedNet.realOutputSet[i], "\t")
        error = estimatedOutput - trainedNet.realOutputSet[i]
        print(" ERROR: ", error, "\n")

    end
end

end # module