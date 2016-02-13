function train!(n::NeuralNet)
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
                        n, netValue, error))
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


function teachNeuronsOfLayer(numberOfInputNeurons::Int, line::Int,
        n::NeuralNet, netValue::Float64, error::Float64)
    listOfNeurons = Vector{Neuron}(0)
    inputWeightsInNew = Vector{Float64}(0)
    inputWeightsInOld = Vector{Float64}(0)

    for j = 1:numberOfInputNeurons
        inputWeightsInOld = n.inputLayer.listOfNeurons[j].listOfWeightIn
        inputWeightOld = inputWeightsInOld[1]

        push!( inputWeightsInNew, calcNewWeight(n.trainType,
            inputWeightOld, n, error, n.trainSet[line,j], netValue) )

        neuron = Neuron()
        neuron.listOfWeightIn = inputWeightsInNew
        push!( listOfNeurons, neuron )
        inputWeightsInNew = Vector{Float64}(0)
    end

    return listOfNeurons
end


function calcNewWeight(trainType::TrainingTypesENUM, inputWeightOld::Float64,
    n::NeuralNet, error::Float64, trainSample::Float64, netValue::Float64)

    if trainType == TrainingTypesENUM(PERCEPTRON)
        return ( inputWeightOld + n.learningRate * error * trainSample )
    elseif trainType == TrainingTypesENUM(ADALINE)
        return ( inputWeightOld + n.learningRate * error * trainSample *
            derivativeActivationFnc(n.activationFncType, netValue) )
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

    elseif fnc == ActivationFncENUM(LINEAR)
        return value

    elseif fnc == ActivationFncENUM(SIGLOG)
        return ( 1.0 / (1.0 + exp(-value)) )

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


