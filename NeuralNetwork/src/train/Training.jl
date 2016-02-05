typealias Vector{T} Array{T,1}

type Training
    epochs::Int64
    error::Float64
    mse::Float64
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
                inputWeightIn = n.InputLayer.listOfNeurons[j].listOfWeightIn
                inputWeight = inputWeightIn[1]
                netValue = netValue + inputWeight * n.trainSet[i,j]
            end

            estimatedOutput = activationFnc(n, netValue)
            realOutput = n.realOutputSet[i]
            setError(realOutput - estimatedOutput)
            

