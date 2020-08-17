import tensorflow as tf

def evaluate(xs, ys, model):
  xs = tf.stack(xs)
  ys = tf.stack(ys)
  res = model.evaluate(xs, ys)
  return res

  