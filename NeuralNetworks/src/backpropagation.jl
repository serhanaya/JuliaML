function trainBP!(n::NeuralNet)
    epoch = 0
    mse = 1.0

    while mse < n.targetError
        if epoch >= n.maxEpochs
            break
        end

        rows = size(n.trainSet)[1]
        sumErrors = 0.0

        for rows_i = 1:rows
            n = forward!(n, rows_i)
            n = backpropagation!(n, rows_i)
            sumErrors = sumErrors + n.errorMean
        end

        mse = sumErrors / rows
        println(mse)
        epoch += 1
    end
    println("Number of epochs: ", epoch)
    return n
end

function forward!(n::NeuralNet, row::Int)
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

                netValueOut = activationFnc(n.activationFncOutputLayer, netValue)

                n.outputLayer.listOfNeurons[outLayer_i].outputValue = netValueOut

                # error
                estimatedOutput = netValueOut
                realOutput = n.realMatrixOutputSet[row, outLayer_i]
                error = realOutput - estimatedOutput
                n.outputLayer.listOfNeurons[outLayer_i].error
                sumError = sumError + error^2

            end

            # error mean
            errorMean = sumError / n.outputLayer.numberOfNeuronsInLayer
            n.errorMean = errorMean

            n.listOfHiddenLayer[hiddenLayer_i].listOfNeurons = hiddenLayer.listOfNeurons

            hiddenLayer_i += 1

        end

    end

    return n

end

function backpropagation!(n::NeuralNet, row::Int)
    outputLayer = n.outputLayer.listOfNeurons

    hiddenLayer = n.listOfHiddenLayer[1].listOfNeurons

    error = 0.0
    netValue = 0.0
    sensibility = 0.0

    # sensibility output layer
    for neuron::Neuron in outputLayer
        error = neuron.error
        netValue = neuron.outputValue
        sensibility = derivativeActivationFnc(n.activationFncOutputLayer, netValue * error)

        neuron.sensibility = sensibility

    end

    # sensibility hidden layer
    for neuron::Neuron in hiddenLayer
        sensibility = 0.0

        if (size(neuron.listOfWeightIn)[1] > 0) # exclude bias
            listOfWeightsOut = neuron.listOfWeightOut

            tempSensibility = 0.0

            weight_i = 0
            for weight in listOfWeightsOut
                tempSensibility += weight * outputLayer[weight_i].sensibility
                weight_i += 1
            end

            sensibility = derivativeActivationFnc(n.activationFncType, neuron.outputValue) * tempSensibility

            neuron.sensibility = sensibility

        end

    end

    # fix weights (teach) [output layer to hidden layer]
    for outLayer_i = 1:outputLayer.numberOfNeuronsInLayer

        for neuron::Neuron in hiddenLayer
            netWeight = neuron.listOfWeightOut[outLayer_i] +
                ( n.learningRate * outputLayer[outLayer_i].sensibility * neuron.outputValue )

            neuron.listOfWeightOut[outLayer_i] = newWeight  # Kontrol et!

        end

    end

    # fix weights (teach) [hidden layer to input layer]
    for neuron::Neuron in hiddenLayer
        hiddenLayerInputWeights = neuron.listOfWeightIn

        if size(hiddenLayerInputWeights)[1] > 0  # exclude bias

            hidden_i = 0
            netWeight = 0.0

            for i = 1:n.inputLayer.numberOfNeuronsInLayer
                newWeight = hiddenLayerInputWeights[hidden_i] +
                    ( n.learningRate * neuron.sensibility *
                        neuron.sensibility * n.trainSet[row, i] )

                neuron.listOfWeightIn[hidden_i] = newWeight

                hidden_i += 1

            end

        end

    end

    n.listOfHiddenLayer[1].listOfNeurons = hiddenLayer

    return n

end
