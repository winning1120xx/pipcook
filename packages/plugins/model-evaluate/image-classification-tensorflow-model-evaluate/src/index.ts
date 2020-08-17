import { ImageDataset, ModelEvaluateType, UniModel, EvaluateResult } from '@pipcook/pipcook-core';
import * as path from 'path';

const boa = require('@pipcook/boa');
const tf = boa.import('tensorflow');

const sys = boa.import('sys');
sys.path.insert(0, path.join(__dirname, '..'));
const { evaluate } = boa.import('pyscript.index');

/**
 * this is plugin used to train tfjs model with pascal voc data format for image classification problem.
 * @param data : train data
 * @param model : model loaded before
 * @param epochs : need to specify epochs
 * @param batchSize : need to specify batch size
 * @param optimizer : need to specify optimizer
 */
const ModelEvaluate: ModelEvaluateType = async (data: ImageDataset, model: UniModel): Promise<EvaluateResult> => {
  let batchSize = 16;

  const { testLoader } = data;

  // sample data must contain test data
  if (testLoader) {
    const count = await testLoader.len();
    const batches = parseInt(String(count / batchSize));

    let loss = 0;
    let accuracy = 0;

    for (let i = 0; i < batches; i++) {
      const dataBatch = await data.testLoader.nextBatch(batchSize);
      const evaluateRes = evaluate(dataBatch.map((ele) => ele.data), dataBatch.map((ele) => ele.label), model.model)
      if (typeof evaluateRes[0] === 'number') {
        loss += evaluateRes[0];  
      } else {
        loss += Number(evaluateRes[0].numpy());
      }
      if (typeof evaluateRes[1] === 'number') {
        accuracy += evaluateRes[1];
      } else {
        accuracy += Number(evaluateRes[1].numpy());
      }
    }

    loss /= batches;
    accuracy /= batches;

    console.log(`Evaluate Result: loss: ${loss} accuracy: ${accuracy}`);

    return {
      loss,
      accuracy
    };
  }
  // just skiped if no test loader.
  return { pass: true };
};

export default ModelEvaluate;
