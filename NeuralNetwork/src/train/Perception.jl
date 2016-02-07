# To be implemented into NeuralNet

type Perception
  """
  eta : Learning rate
  n_iter : Number of epochs (passes over the training set)
  """
  eta::Float64
  n_iter::Int64
  w_
  errors_
end


function fit(p::Perception, X, y)
  """Fit training data.

  Parameters
  ---------
  X : {array-like}, shape = [n_samples, n_features]
      Training vectors, where n_samples
      is the number of samples and
      n_features is the number of features.
  y : array-like. shape = [n_samples]
      Target values.

  Returns
  -------
  p : object
  """
  p.w_ = zeros(1 + size(X)[1])  # initialize the weights to a
                                # zero vector
  p.errors_ = []

  for _ in range(1, p.n_iter)
    errors = 0
    for (xi, target) in zip(X, y)
      update = p.eta * (target - p.predict(xi))
      p.w_[2:end] += update * xi
      p.w_[1] += update
      errors += convert(Int64, update != 0.0)
    end
    append!(p.errors_, errors)
  end
  return p
end

function net_input(p::Perception, X)
  """Calculate net input"""
  return dot(X, p.w_[2:end]) + p.w_[1]
end

function predict(p::Perception, X)
  """Return class label after unit step"""
  pred = zeros(size(net_input(p,X)))
  k, l = size(net_input(p,X))
  for i = 1:k, j=1:j
    net_input(p,X)[i,j] >= 0.0 ? pred[i,j] = 1 : pred[i,j] = -1
  end
  return pred
end
