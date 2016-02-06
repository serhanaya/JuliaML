typealias Vector{T} Array{T,1}

type Training
    epochs::Int64
    error::Float64
    mse::Float64
    function Training(;epochs=0, error=0, mse=0)
        new(epochs, error, mse)
    end
end

# TrainingTypesENUM

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
            
            if abs(t.error > n.targetError
                # fix weights
                inpLayer = InputLayer()
                inpLayer.listOfNeurons = teachNeuronsOfLayer(cols, i,
                    n, netValue)
                n.InputLayer = inpLayer
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





