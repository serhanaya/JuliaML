# TODO
type LevenbergMarquardt
	jacobian::Matrix{Float64}
	damping::Float64
	error::Float64

	function LevenbergMarquardt(jacobian=Matrix{Float64}(0), damping=0.0, error=0.0)
		new(jacobian, damping, error)
	end

end


function trainMQ!(n::NeuralNet)
    epoch = 0
    mse = 1.0

    while mse > n.targetError
        if epoch >= n.maxEpochs
            break
        end

        rows = size(n.trainSet)[1]
        sumErrors = 0.0

        for row_i = 1:rows

        	n = forward(n, rows_i)

        	buildJacobianMatrix(n, rows_i)

        	sumErrors += n.errorMean()

        end

        mse = sumErrors / rows

        n = updateWeights(n)

        println(mse)

        epoch += 1

    end

    println("Number of epochs", epoch)

    return n

end


function buildJacobianMatrix(n::NeuralNet, row::Int)
	outputLayer = n.outputLayer.listOfNeurons
	hiddenLayer = n.listOfHiddenLayer[1].listOfNeurons

	nb = backpropagation(n, row)

	numberOfInputs = n.getInputLayer.numberOfNeuronsInLayer
	numberOfHiddenNeurons = n.hiddenLayer.numberOfNeuronsInLayer
	numberOfOututs = n.outputLayer.numberOfNeuronsInLayer

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
		error = Array(size(n.trainSet)[1], 1)
	end

	i = 1
	# output layer
	for output::Neuron in outputLayer
		j = 1
		for neuron::Neuron in hiddenLayer
			jacobian[]
