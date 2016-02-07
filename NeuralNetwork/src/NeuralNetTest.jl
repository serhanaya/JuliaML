# import NeuralNet

function testPerceptron()
    testNet = NeuralNet.initNet(2, 0, 0, 1)

    println("---------PERCEPTRON INIT NET---------")

    NeuralNet.printNet(testNet)  # testNet.printNet(testNet)

    # First column has bias
    trainSet = [ 1.0 0.0 0.0 ;
    		     1.0 0.0 1.0 ;
                 1.0 1.0 0.0 ;
                 1.0 1.0 1.0 ]
    realOutputSet = [0.0, 0.0, 0.0, 1.0]
    maxEpochs = 10
    targetError = 0.002
    learningRate = 1.0
    trainType = NeuralNet.TrainingTypesENUM(PERCEPTRON)
    activationFncType = NeuralNet.ActivationFncENUM(STEP)

    n = NeuralNet(trainSet = trainSet, realOutputSet = realOutputSet,
        maxEpochs = maxEpochs, targetError = targetError, learningRate = learningRate,
        trainType = trainType, activationFncType = activationFncType)

    trainedNET = NeuralNet.train(n)

    println()
    println("---------PERCEPTRON TRAINED NET---------")
    NeuralNet.printNet(trainedNet)

    println()
    println("---------PERCEPTRON PRINT RESULT---------")
    NeuralNet.printTrainedNetResult(trainedNet)
end

function testAdaline()
    testNet = NeuralNet.initNet(3, 0, 0, 1)

    println("---------ADALINE INIT NET---------")

    NeuralNet.printNet(testNet)

    # First column has bias
    trainSet = [ 1.0 0.98 0.94 0.95 ;
                 1.0 0.60 0.60 0.85 ; 1.0 0.35 0.15 0.15 ;
                 1.0 0.25 0.30 0.98 ; 1.0 0.75 0.85 0.91 ;
                 1.0 0.43 0.57 0.87 ; 1.0 0.05 0.06 0.01 ]
    realOutputSet = [0.80, 0.59, 0.23, 0.45, 0.74, 0.63, 0.10]
    maxEpochs = 10
    targetError = 0.0001
    learningRate = 0.5
    trainType = NeuralNet.TrainingTypesENUM(ADALINE)
    activationFncType = NeuralNet.ActivationFncENUM(LINEAR)

    n = NeuralNet(trainSet = trainSet, realOutputSet = realOutputSet,
        maxEpochs = maxEpochs, targetError = targetError, learningRate = learningRate,
        trainType = trainType, activationFncType = activationFncType)

    trainedNET = NeuralNet.train(n)

    println()
    println("---------ADALINE TRAINED NET---------")

    NeuralNet.printNet(trainedNet)

    println()
    println("---------ADALINE PRINT RESULT---------")
    NeuralNet.printTrainedNetResult(trainedNet)

    println()
    println("---------ADALINE MSE BY EPOCH---------")
    println(trainedNet.listOfMSE)
end
