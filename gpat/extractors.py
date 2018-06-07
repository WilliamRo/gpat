import numpy as np


def _naive_top_k(probs, k):
  assert isinstance(probs, np.ndarray) and len(probs.shape) == 2
  prob_sum = np.sum(probs, axis=0, keepdims=True)
  assert len(prob_sum.shape) == 2 and prob_sum.shape[0] == 1
  return np.flip(np.argsort(prob_sum, axis=1), axis=1)[:k]

def naive_top_k(k):
  return lambda one_hot: _naive_top_k(one_hot, k)


if __name__ == '__main__':
  probs = np.arange(12).reshape((4, 3))
  a = _naive_top_k(probs, 2)
  b = np.concatenate([a, a])
  c = 1
