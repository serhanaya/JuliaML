function train!(n::NeuralNet)
	rows = size(n.trainSet)[1]
	n = initNet(n)
	trainData = n.trainSet

	for epoch = 1:n.maxEpochs

		# println("### EPOCH: ", epoch)

		for row_i = 1:rows
			listOfDistances = calcEuclideanDistance(n, trainData, row_i)
			 
			winnerNeuron, winnerNeuronIndex = findmin(listOfDistances)  # TODO: check
			
			n = fixWinnerWeights(n, winnerNeuron, row_i)
		end

    end
	
	return n

end


function initNet(n::NeuralNet)

	listOfWeightOut = Vector{Float64}(0)

	for int = 1:n.inputLayer.numberOfNeuronsInLayer
		push!(listOfWeightOut, 0.0)
	end

	n.inputLayer.listOfNeurons[1].listOfWeightOut = listOfWeightOut

	return n

end


function calcEuclideanDistance(n::NeuralNet, data::Matrix{Float64}, row::Int)

	weight_i = 0

	for cluster_i = 1:n.outputLayer.numberOfNeuronsInLayer

		distance = 0.0

		for input_j = 1:n.inputLayer.numberOfNeuronsInLayer

			weight = n.inputLayer.listOfNeurons[1].listOfWeightOut[weight_i]

			distance += (data[row, input_j] - weight)^2

			weight += 1

		end

		push!(listOfDistances, distance)

	end

	return listOfDistances

end


function fixWinnerWeights(n::NeuralNet, winnerNeuron::Int, trainSetRow::Int)

	start = winnerNeuron * n.inputLayer.numberOfNeuronsInLayer

	if start < 0
		start = 0
	end

	last = start + n.inputLayer.numberOfNeuronsInLayer

	listOfOldWeights = n.inputLayer.listOfNeurons[1].listOfWeightOut[start:last]  # TODO: check
	
	listOfWeights = n.inputLayer.listOfNeurons[1].listOfWeightOut

	col_i = 0

	for j = start:last

		trainSetValue = n.trainSet[trainSetRow, col_i]
		newWeight = listOfOldWeights[col_i] + n.learningRate * 
			( trainSetValue - listOfOldWeights[col_i] )

		# println("newWeight: " + newWeight)
		listOfWeights[j] = newWeight
		col_i += 1

	end

	n.inputLayer.listOfNeurons[1].listOfWeightOut = listOfWeights

	return n

end


function netValidation(n::NeuralNet)
	rows = size(n.validationSet)[1]

	validationData = n.validationSet

	for row_i = 1:rows
		listOfDistances = calcEuclideanDistance(n, validationData, row_i)

		winnerNeuron, winnerNeuronIndex = findmin(listOfDistances)  # TODO: check
		println("### VALIDATION RESULT ###")

		if winnerNeuron == 0
			println("CLUSTER 1")
			break;
		elseif winnerNeuron == 1
			println("CLUSTER 2")
			break;
		else
			throw(Error("Error! Without neural clustering..."))
		end
	end
end

