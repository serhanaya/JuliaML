type AdalineGD
  """ADAptive LInear NEuron classifier.

  Parameters
  ------------
  eta : float
      Learning rate (between 0.0 and 1.0)
  n_iter : int
      Passes over the training dataset.

  Attributes
  -----------
  w_ : 1d-array
      Weights after fitting.
  errors_ : list
      Number of misclassifications in every epoch.

  """
  eta::Float64
  n_iter::Int64
  w_
  errors_
end

function fit(p::AdalineGD, X, y)
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
  p.w_ = zeros(1 + size(X)[1])
  p.cost_ = []

  for i in range(1, p.n_iter):
    output = p.net_input(X)
    errors = (y - output)
    p.w_[2:] += p.eta * dot(transpose(X), errors)
