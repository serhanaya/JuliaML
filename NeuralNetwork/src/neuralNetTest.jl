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

    trainedNet = trainNet(testNet)  # TODO: check

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

    trainedNet = trainNet(testNet)  # TODO: check

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

function testBackpropagation()
    testNet = initNet(2, 1, 3, 2)

    println("---------BACKPROPAGATION INIT NET---------")

    printNet(testNet)

    # first column has BIAS
    testNet.trainSet = [ 1.0 1.0 0.73 ; 1.0 1.0 0.81 ; 1.0 1.0 0.86 ;
                         1.0 1.0 0.95 ; 1.0 0.0 0.45 ; 1.0 1.0 0.70 ;
                         1.0 0.0 0.51 ; 1.0 1.0 0.89 ; 1.0 1.0 0.79 ;
                         1.0 0.0 0.54 ]
    testNet.realMatrixOutputSet = [ 1.0 0.0; 1.0 0.0; 1.0 0.0;
                                    1.0 0.0; 1.0 0.0; 0.0 1.0;
                                    0.0 1.0; 0.0 1.0; 0.0 1.0;
                                    0.0 1.0 ]

    testNet.maxEpochs = 1000
    testNet.targetError = 0.002
    testNet.learningRate = 0.1
    testNet.trainType = TrainingTypesENUM(BACKPROPAGATION)
    testNet.activationFncType = ActivationFncENUM(SIGLOG)
    testNet.activationFncTypeOutputLayer = ActivationFncENUM(LINEAR)

    trainedNet = trainNet(testNet)  # todo: check

    println()
    println("---------BACKPROPAGATION TRAINED NET---------")

    printNet(trainedNet)

end


function testLMA()
    testNet = initNet(2, 1, 3, 2)

    println("---------LEVENBERG-MARQUARDT NET---------")

    printNet(testNet)

     # first column has BIAS
    testNet.trainSet = [ 1.0 1.0 0.73 ; 1.0 1.0 0.81 ; 1.0 1.0 0.86 ;
                         1.0 1.0 0.95 ; 1.0 0.0 0.45 ; 1.0 1.0 0.70 ;
                         1.0 0.0 0.51 ; 1.0 1.0 0.89 ; 1.0 1.0 0.79 ;
                         1.0 0.0 0.54 ]
    testNet.realMatrixOutputSet = [ 1.0 0.0; 1.0 0.0; 1.0 0.0;
                                    1.0 0.0; 1.0 0.0; 0.0 1.0;
                                    0.0 1.0; 0.0 1.0; 0.0 1.0;
                                    0.0 1.0 ]

    testNet.maxEpochs = 1000
    testNet.targetError = 0.002
    testNet.learningRate = 0.1
    testNet.trainType = TrainingTypesENUM(LEVENBERG_MARQUARDT)
    testNet.activationFncType = ActivationFncENUM(SIGLOG)
    testNet.activationFncTypeOutputLayer = ActivationFncENUM(LINEAR)

    trainedNet = trainNet(testNet)  # TODO: check.

    println()
    println("---------BACKPROPAGATION TRAINED NET---------")

    printNet(trainedNet)

end


function testKohonen()
    # 2 inputs because of "bias"
    testNet = initNet(2, 0, 0, 2)

    testNet.trainSet = [ 1.0 -1.0  1.0; -1.0 -1.0 -1.0; -1.0 -1.0  1.0
                         1.0  1.0 -1.0; -1.0  1.0  1.0;  1.0 -1.0 -1.0 ]

    testNet.validationSet = [ -1.0  1.0 -1.0;
                               1.0  1.0  1.0 ]

    testNet.maxEpochs = 10
    testNet.learningRate = 0.1
    testNet.trainType = TrainingTypesENUM(KOHONEN)

    trainedNet = trainNet(testNet)  # todo: check

    println()
    println("---------KOHONEN VALIDATION NET---------")

    netValidation(trainedNet)

end
