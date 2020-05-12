// See https://webmachinelearning.github.io/webnn/#api for living standard.

type long = number;
type double = number;

export type Operand = object;
export type OperandLayout = 'nchw' | 'nhwc';
export type OperandType =
  'float32' |
  'float16' |
  'int32' |
  'uint32' |
  'tensor-float32' |
  'tensor-float16' |
  'tensor-int32' |
  'tensor-quant8-asymm';

export type PowerPreference =
  // Let the user agent decide the most suitable behavior. This is the default value.
  "default" |
  // Prioritizes execution speed over other considerations e.g. power consumption
  "high-performance" |
  // Prioritizes power consumption over other consideraions e.g. execution speed
  "low-power";

export interface CompilationOptions {
  // Compilation preference as related to power consumption level
  powerPreference: PowerPreference;
}

export interface OperandDescriptor {
  // The operand type.
  type: OperandType;

  // The dimensions field is only required for tensor operands.
  // The negative value means an unknown dimension.
  dimensions: long[];

  // The following two fields are only required for quantized operand.
  // scale: an non-negative floating point value
  // zeroPoint: an integer, in range [0, 255]
  // The real value is (value - zeroPoint) * scale
  scale?: double;
  zeroPoint?: long;
}

export declare class Compilation {
  createExecution(): Promise<Execution>;
}

export declare class Execution {
  setInput(index: long, data: ArrayBufferView): void;
  setOutput(index: long, data: ArrayBufferView): void;
  startCompute(): Promise<void>;
}

export declare class Model {
  createCompilation(options: CompilationOptions): Promise<Compilation>;
}

export interface NeuralNetworkContextBase {
  // Create an Operand object that represents a model input.
  input(desc: OperandDescriptor): Operand;

  // Create an Operand object that represents a model constant.
  constant(desc: OperandDescriptor, value: number[]): Operand;

  // Create a Model object by identifying output operands.
  createModel(outputs: Operand[]): Promise<Model>;

  // ops
  add(a: Operand, b: Operand): Operand;
  matmul(a: Operand, b: Operand): Operand;
  mul(a: Operand, b: Operand): Operand;
}

// load webnn implementation
const { NeuralNetworkContext } = require('./webnn.tf2');
const nnctx = new NeuralNetworkContext();

export const ml = {
  getNeuralNetworkContext(): NeuralNetworkContextBase {
    return nnctx;
  }
};
