# rows @line21
# super.activate

function train(n::neuralNet)
    epoch = 0
    mse = 1.0

    while mse < n.targetError
        if epoch >= n.targetError
            break
        end

        rows = size(n.trainSet)[1]
        sumErrors = 0.0

        for rows_i = 1:rows
            n = forward(n, rows_i)
            n = backpropagation(n, rows_i)
            sumErrors = sumErrors + n.errorMean
        end

        mse = sumErrors / rows
        println(mse)
        epoch += 1
    end
    println("Number of epochs: ", epoch)
    return n
end

function forward(n::neuralNet, row::Int)
    listOfHiddenLayer = n.listOfHiddenLayer

    estimatedOutput = 0.0
    realOutput = 0.0
    sumError = 0.0

    if size(listOfHiddenLayer)[1] > 0
        hiddenLayer_i = 0

        for hiddenLayer::HiddenLayer in listOfHiddenLayer
            numberOfNeuronsInLayer = hiddenLayer.numberOfNeuronsInLayer

            for neuron::Neuron in hiddenLayer.listOfNeurons

                netValueOut = 0.0

                if size(neuron.listOfWeightIn)[1] > 0  # exclude bias
                    netValue = 0.0

                    for layer_j = 1:numberOfNeuronsInLayer  # exclude bias
                        hiddenWeightIn = neuron.listOfWeightIn[layer_j]
                        netValue = netValue + hiddenWeightIn * n.trainSet[row, layer_j]
                    end

                    # output hidden layer (1)
                    netValueOut = activationFnc(n.activationFncType, netValue)
                    neuron.outputValue = netValueOut

                else
                    neuron.outputValue = 1.0
                end

            end

            # output hidden layer (2)

            for outLayer_i = 1:n.outputLayer.numberOfNeuronsInLayer
                netValue = 0.0
                netValueOut = 0.0

                for neuron::Neuron in hiddenLayer.listOfNeurons
                    hiddenWeightOut = neuron.listOfWeightOut[outLayer_i]
                    netValue = netValue + hiddenWeightOut * neuron.outputValue
                end

                netValueOut = 
