push!(LOAD_PATH, pwd())
println(LOAD_PATH)

import Nnet

Nnet.testPerceptron()