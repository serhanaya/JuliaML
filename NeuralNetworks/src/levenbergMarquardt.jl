# Levenberg-Marquardt method.

function trainMQ!(n::NeuralNet)
	damping = 0.1
    epoch = 0
    mse = 1.0
    while mse > n.targetError
        if epoch >= n.maxEpochs
            break
        end
        rows = size(n.trainSet)[1]
        sumErrors = 0.0
        for rows_i = 1:rows
        	n = forward!(n, rows_i)
        	buildJacobianMatrix(n, rows_i)
			sumErrors += n.errorMean()
        end
        mse = sumErrors / rows
        n = updateWeights!(n)
        println(mse)
        epoch += 1
    end
    println("Number of epochs", epoch)
    return n
end

function buildJacobianMatrix(n::NeuralNet, row::Int)
	outputLayer = n.outputLayer.listOfNeurons
	hiddenLayer = n.listOfHiddenLayer[1].listOfNeurons
	nb = backpropagation!(n, row)
	numberOfInputs = n.getInputLayer.numberOfNeuronsInLayer
	numberOfHiddenNeurons = n.hiddenLayer.numberOfNeuronsInLayer
	numberOfOutputs = n.outputLayer.numberOfNeuronsInLayer
	if isdefined(:jacobian) == false
		nrows = size(n.trainSet)[1]
		ncols = (numberOfInputs)*(numberOfHiddenNeurons-1)+
			(numberOfHiddenNeurons)*(numberOfOutputs)
		jacobian = Array(Float64, nrows, ncols)
	end
	i = 1
	# Hidden Layer
	for (neuron::Neuron in hiddenLayer)
		hiddenLayerInputWeights = neuron.listOfWeightIn
		if  size(hiddenLayerInputWeights)[1] > 0  # exclude bias
			for j = 1:n.inputLayer.numberOfNeuronsInLayer
				jacobian[row, ((i-1)*(numberOfInputs))+(j)] =
				     (neuron.sensibility * n.trainSet[row,j])/n.errorMean  # TODO: check
			end
		else
			# jacobian[row, i*numberOfInputs] = 1.0
		end
		# Bias will have no effect
		i += 1
	end
	if isdefined(error) == false
		error = Array(Float64, size(n.trainSet)[1], 1)
	end
	i = 1
	# output layer
	for output::Neuron in outputLayer
		j = 1
		for neuron::Neuron in hiddenLayer
			jacobian[row, (numberOfInputs)*(numberOfHiddenNeurons-1)+
						(i*(numberOfHiddenNeurons))+j] =
                output.sensibility * neuron.outputValue/n.errorMean
		end
		j += 1
	end
	# Bias will have no effect
	# jacobian[row, (numberOfInputs)*(numberOfHiddenNeurons-1)+
	#			(i*(numberOfHiddenNeurons))+j] = 1.0
	i += 1
	error[row, 1] = n.errorMean
end

function updateWeights!(n::NeuralNet)
	# delta = inv(J`J + damping I) * J` error
	term1 = transpose(jacobian) * jacobian + identity(size(jacobian)[2]) * damping
	term2 = transpose(jacobian) * error
	delta = inv(term1) * term2

	outputLayer = n.outputLayer.listOfNeurons
	hiddenLayer = n.listOfHiddenLayer[1].listOfNeurons

	numberOfInputs = n.inputLayer.numberOfNeuronsInLayer
	numberOfHiddenNeurons = n.hiddenLayer.numberOfNeuronsInLayer
	numberOfOutputs = n.outputLayer.numberOfNeuronsInLayer

	i = 1
	for hidden::Neuron in hiddenLayer
		hiddenLayerInputWeights = hidden.listOfWeightIn
		if size(hiddenLayerInputWeights)[1] > 0  # exclude bias
			newWeight = 0.0
			for j = 1:n.inputLayer.numberOfNeuronsInLayer
				newWeight = hiddenLayerInputWeights[i] + delta[i * numberOfInputs + j, 1]
				push!(hidden.listOfWeightIn[i], newWeight)
			end
			i += 1
		end
	end

	i += 1
	for output::Neuron in outputLayer
		j += 1
		newWeight = 0.0
		for neuron::Neuron in hiddenLayer
			newWeight = neuron.listOfWeightOut[i] + delta[numberOfInputs * (numberOfHiddenNeurons-1) + (i * numberOfHiddenNeurons) + j, 1]
			push!(neuron.listOfWeightOut[i], newWeight)
			j += 1
		end
		i += 1
	end

	return n
end
