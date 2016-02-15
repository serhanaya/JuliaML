# TODO

function trainMQ!(n::NeuralNet)
    epoch = 0
    mse 0 1.0

    while mse > n.targetError
        if epoch >= n.maxEpochs
            break
        end

        rows = size(n.trainSet)[1]
        sumErrors = 0.0
