import {
  Operand,
  Model,
  OperandDescriptor,
  NeuralNetworkContextBase
} from './index';

const boa = require('@pipcook/boa');
const tf = boa.import('tensorflow');
const np = boa.import('numpy');
type long = number;

const tftypes: Record<string, any> = {
  'float16': tf.float16,
  'float32': tf.float32,
  'int32': tf.int32,
  'uint32': tf.uint32,
  'tensor-float32': tf.float32,
  'tensor-float16': tf.float16,
  'tensor-int32': tf.int32
};

export class NeuralNetworkContext implements NeuralNetworkContextBase {
  input(desc: OperandDescriptor): Operand {
    return tf.Variable(boa.kwargs({
      initial_value: np.zeros(desc.dimensions),
      dtype: tftypes[desc.type],
      shape: desc.dimensions
    }));
  }
  constant(desc: OperandDescriptor, value: number[]): Operand {
    return tf.constant(value, boa.kwargs({
      dtype: tftypes[desc.type],
      shape: desc.dimensions
    }));
  }
  async createModel(outputs: Operand[]): Promise<Model> {
    const graph = tf.Graph();
    graph.add_to_collection('output', outputs[0]);
    return graph;
  }
  add(a: Operand, b: Operand): Operand {
    return tf.math.add(a, b);
  }
  matmul(a: Operand, b: Operand): Operand {
    return tf.math.matmul(a, b);
  }
  mul(a: Operand, b: Operand): Operand {
    return tf.math.multiply(a, b);
  }
}
