typealias Vector{T} Array{T,1}

type Training
    epochs::Int64
    error::Float64
    mse::Float64
    function Training(;epochs=0, error=0, mse=0)
        new(epochs, error, mse)
    end
end

@enum TrainingTypesENUM PERCEPTRON ADALINE
@enum ActivationFncENUM STEP LINEAR SIGLOG HYPERTAN

function train(n::NeuralNet, t::Training)
    inputWeightIn = Vector{Float64}(0)
    rows = size(n.trainSet)[1]
    cols = size(n.trainSet)[2]

    while t.epochs < n.maxEpochs
        estimatedOutput = 0.0
        realOutput = 0.0

        for i = 1:rows
            netValue = 0.0

            for j = 1:cols
                inputWeightIn = n.iLayer.listOfNeurons[j].listOfWeightIn
                inputWeight = inputWeightIn[1]
                netValue = netValue + inputWeight * n.trainSet[i,j]
            end

            estimatedOutput = activationFnc(n, netValue)
            realOutput = n.realOutputSet[i] 
            t.error = realOutput - estimatedOutput
            
            if abs(t.error) > n.targetError
                # fix weights
                iLayerTemp = 
                    InputLayer(listOfNeurons = teachNeuronsOfLayer(cols, i,
                        n, netValue))  # Fix constructor of InputLayer.
                n.iLayer = iLayerTemp
            end
        end

        t.mse = (realOutput - estimatedOutput)^2
        push!(n.listOfMSE, t.mse)
        t.epochs += 1
    end

    n.trainingError = t.error

    return n
end


function teachNeuronsOfLayer(numberOfInputNeurons::Int64, line::Int64,
        n::NeuralNet, t::Training, netValue::Float64)
    listOfNeuronsUpdate = Vector{Neuron}(0)
    inputWeightsInNew = Vector{Float64}(0)
    inputWeightsInOld = Vector{Float64}(0)

    for j = 1:numberOfInputNeurons
        inputWeightsInOld = n.iLayer.listOfNeurons[j].listOfWeightIn
        inputWeightOld = inputWeightsInOld[1]

        push!( inputWeightsInNew, calcNewWeight(n.trainType, 
            inputWeightOld, n, t.error, n.trainSet[line][j], netValue) )

        neuron = Neuron()  # Constructor needed.
        n.listOfWeightIn = inputWeightsInNew
        push!( listOfNeuronsUpdate, neuron )
        inputWeightsInNew = Vector{Float64}(0)
    end

    return listOfNeuronsUpdate
end


function calcNewWeight(trainType::TrainingTypesENUM, inputWeightOld::Float64,
    n::NeuralNet, error::Float64, trainSample::Float64, netValue::Float64)
    
    if trainType == PERCEPTRON
        return inputWeightOld + n.learningRate * error * trainSample
    elseif trainType == ADALINE
        return inputWeightOld * n.learningRate * error * trainSample *
             derivativeActivationFnc(n.activationFncType, netValue)
    else
        throw(ArgumentError(trainType
                    + " does not exist in TrainingTypesENUM"))
    end
end

function activationFnc(fnc::ActivationFncENUM, value::Float64)
    
    if fnc == STEP
        if value >= 0
            return 1.0
        else
            return 0.0
        end
    
    elseif fnc == LINEAR
        return value
    
    elseif fnc == SIGLOG
        return 1.0 / (1.0 + exp(-value))

    elseif fnc == HYPERTAN
        return tanh(value)

    else
        throw(ArgumentError(fnc
                    + " does not exist in ActivationFncENUM"))
    
    end
end

function derivativeActivationFnc(fnc::ActivationFncENUM, value::Float64)
    
    if fnc == LINEAR
        return 1.0
    
    elseif fnc == SIGLOG
        return value * (1.0 - value);

    elseif fnc == HYPERTAN
        return 1.0 / cosh(value)^2

    else
        throw(ArgumentError(fnc
                    + " does not exist in ActivationFncENUM"))
    
    end
end    

function printTrainedResult(trainedNet::NeuralNet, t::Training)
    rows = size(trainedNet.trainSet)[1]
    cols = size(trainedNet.trainSet)[2]

    inputWeightIn = Vector{Float64}(0)

    for i = 1:rows
        netValue = 0.0
        for j = 1:cols
            inputWeightIn = trainedNet.iLayer.listOfNeurons[j].listOfWeightIn
            inputWeight = inputWeightIn[1]
            netValue = netValue + inputWeight * trainedNet.trainSet[i,j]
            print(trainedNet.trainSet[i,j] + "\t")  # May cause a problem
        end

        estimatedOutput = activationFnc(trainedNet.trainSet[i,j])
        print(trainedNet.trainSet[i,j] + "\t")  # Change with @printf
        print(" REAL OUTPUT: "
                    + trainedNet.realOutputSet[i] + "\t")
        error = estimatedOutput - trainedNet.realOutputSet[i]
        print(" ERROR: " + error + "\n")

    end
end