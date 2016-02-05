typealias Vector{T} Array{T,1}
typealias Matrix{T} Array{T,2}

# ActivationFncENUM

# TrainingTypesENUM


type NeuralNet
    iLayer::InputLayer
    hLayer::HiddenLayer
    listOfHiddenLayer::Vector{HiddenLayer}
    oLayer::OutputLayer
    numberOfHiddenLayers::Int64

    trainSet::Matrix{Float64}
    validationSet::Matrix{Float64}
    realOutputSet::Vector{Float64}
    realMatrixOutputSet::Matrix{Float64}

    maxEpochs::Int64
    learningRate::Float64
    targetError::Float64
    trainingError::Float64
    errorMean::Float64
    listOfMSE::Vector{Float64}

#    activationFnc::ActivationFncENUM
#    activationFncOutputLayer::ActivationFncENUM
#    trainType::TrainingTypesENUM
#    function NeuralNet(...) =
#        new(listOfMSE = Vector{Float64}(0)...)
end

function initNet(numberOfInputNeurons::Int64, numberOfHiddenLayers::Int64,
    numberOfNeuronsInHiddenLayer::Int64, numberOfOutputNeurons::Int64)

    iLayer = InputLayer(numberOfNeuronsInLayer = numberOfInputNeurons)
