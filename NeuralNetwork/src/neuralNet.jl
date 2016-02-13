@enum TrainingTypesENUM PERCEPTRON ADALINE BACKPROPAGATION LEVENBERG_MARQUARDT KOHONEN
@enum ActivationFncENUM STEP LINEAR SIGLOG HYPERTAN

type NeuralNet

    inputLayer::InputLayer
    hiddenLayer::HiddenLayer
    listOfHiddenLayer::Vector{HiddenLayer}
    numberOfHiddenLayers::Int
    outputLayer::OutputLayer

    trainSet::Matrix{Float64}
    validationSet::Matrix{Float64}
    realOutputSet::Vector{Float64}
    realMatrixOutputSet::Matrix{Float64}

    maxEpochs::Int
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


function initNet(numberOfInputNeurons::Int, numberOfHiddenLayers::Int,
    numberOfNeuronsInHiddenLayer::Int, numberOfOutputNeurons::Int)

    inputLayer = InputLayer(numberOfNeuronsInLayer = numberOfInputNeurons + 1)

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
                       numberOfHiddenLayers, outputLayer)
    return newNet
end


function printNet(n::NeuralNet)
    printLayer(n.inputLayer)
    println()
    printLayer(n.listOfHiddenLayer)
    println()
    printLayer(n.outputLayer)
end


function trainNet(n::NeuralNet)
    if n.trainType == PERCEPTRON
        return training.train!(n)
    elseif n.trainType == ADALINE
        return training.train!(n)
    elseif n.trainType == BACKPROPAGATION
        return backpropagation.train!(n)
    elseif n.trainType == KOHONEN
        return kohonen.train!(n)
    else
        throw(Error(n.trainType, " does not exist in TrainingTypesENUM"))
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


function netValidation(n::NeuralNet)
    if n.trainType == KOHONEN
        netValidation( n )
        break
    else
        throw(Error(n.trainType, " does not exists in TrainingTypesENUM"))
    end
end
