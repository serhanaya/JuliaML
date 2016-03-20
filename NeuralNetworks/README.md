A glimpse on the use of NNet:

```julia

push!(LOAD_PATH, pwd())

import NNet

NNet.testPerceptron()

NNet.testAdaline()

NNet.testLMA()

NNet.testBackpropagation()

NNet.testKohonen()
```

Basic implementation is ready and tested - Perceptron and Adaline.

Advanced techniques are implementing and testing.

Please see the post at [my blog](http://serhanaya.github.io) for implementation
details and a use-case example.
