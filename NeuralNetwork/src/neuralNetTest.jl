function testPerceptron()

    testNet = initNet(2, 0, 0, 1)

    println("---------PERCEPTRON INIT NET---------")

    printNet(testNet)  # testNet.printNet(testNet)

    # First column has bias
    testNet.trainSet = [ 1.0 0.0 0.0 ;
    		             1.0 0.0 1.0 ;
                         1.0 1.0 0.0 ;
                         1.0 1.0 1.0 ]
    testNet.realOutputSet = [0.0, 0.0, 0.0, 1.0]
    testNet.maxEpochs = 10
    testNet.targetError = 0.002
    testNet.learningRate = 1.0
    testNet.trainType = TrainingTypesENUM(PERCEPTRON)
    testNet.activationFncType = ActivationFncENUM(STEP)

    trainedNet = train!(testNet)

    println()
    println("---------PERCEPTRON TRAINED NET---------")
    printNet(trainedNet)

    println()
    println("---------PERCEPTRON PRINT RESULT---------")
    printTrainedNetResult(trainedNet)

end

function testAdaline()

    testNet = initNet(3, 0, 0, 1)

    println("---------ADALINE INIT NET---------")

    printNet(testNet)

    # First column has bias
    testNet.trainSet = [ 1.0 0.98 0.94 0.95 ;
                         1.0 0.60 0.60 0.85 ; 1.0 0.35 0.15 0.15 ;
                         1.0 0.25 0.30 0.98 ; 1.0 0.75 0.85 0.91 ;
                         1.0 0.43 0.57 0.87 ; 1.0 0.05 0.06 0.01 ]
    testNet.realOutputSet = [0.80, 0.59, 0.23, 0.45, 0.74, 0.63, 0.10]
    testNet.maxEpochs = 20
    testNet.targetError = 0.0001
    testNet.learningRate = 0.5
    testNet.trainType = TrainingTypesENUM(ADALINE)
    testNet.activationFncType = ActivationFncENUM(LINEAR)

    trainedNet = train!(testNet)

    println()
    println("---------ADALINE TRAINED NET---------")

    printNet(trainedNet)

    println()
    println("---------ADALINE PRINT RESULT---------")
    printTrainedNetResult(trainedNet)

    println()
    println("---------ADALINE MSE BY EPOCH---------")
    println(trainedNet.listOfMSE)

end
