#################################################
##  Genetic algorithm

# 0. Input parameters
# 1. Initialize population
# 2. Fitness Evaluation
# 3. Reproduction
# 4. Crossover
# 5. Mutation
#################################################
## Input parameters for Genetic algorithm
#


problem = 'min';                    # used with roulette wheel.
                                    # 'min': roulette, 'max': tournament
numberOfVariables = 1               # number of variables
numberOfGenerations = 10000         # maximum number of generations
populationSize = 10                 # population size
variableBound = [40 90]             # variable bound
numberOfBits = Array{Int,1}()       # number of bits (related to sensitivity)
push!(numberOfBits, 15)
crossoverProbability = 0.9          # crossover probability
multiCrossover = 1                  # use multi-crossover
mutationProbability = 0.02          # mutation probability
tourniFlag = 0                      # use roulette wheel
epsilon = 1e-7                      # function tolerance
stallFlag = 0                       # stall generations flag
scalingFlag = 0                     # scaling flag
stallGenerations = 500              # stall generations for termination
numberOfConstaints = 0              # for constraint handling
#
#################################################
# import necessary libraries.
using Distributions
#
#################################################
# Initialization of strings
stringLength = 0
for i = 1:numberOfVariables
    stringLength += numberOfBits[1]
end

# All individuals in the population constitues
# populationSize * stringLength pop matrix.
r = rand(0:1, populationSize, stringLength)

## Main loop
# A generation: (begin) fitness -> reproduction (roulette or
#                   tournament ) -> crossover & mutation -> (end)

for generation = 1: numberOfGenerations
    # Decoded value of r (with left bit as MSB)
    de
