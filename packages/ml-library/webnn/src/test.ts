import { ml, OperandDescriptor } from './index';

// Use tensors in 4 dimensions.
const TENSOR_DIMS = [ 2, 2 ];

// Create OperandDescriptor object.
const float32TensorType: OperandDescriptor = {
  type: 'tensor-float32',
  dimensions: TENSOR_DIMS
};

(async function() {
  const nn = ml.getNeuralNetworkContext();
  const tensor0 = nn.constant(float32TensorType, [ 10, 22, 33, 5 ]);
  const tensor1 = nn.input(float32TensorType);
  const tensor2 = nn.input(float32TensorType);

  const i0 = nn.add(tensor0, tensor1);
  const output = nn.mul(i0, tensor2);
  const model = await nn.createModel([ output ]);
  console.log(model);
})();
