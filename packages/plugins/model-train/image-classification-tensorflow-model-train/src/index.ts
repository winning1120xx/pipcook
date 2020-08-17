import { ImageDataset, ModelTrainType, UniModel, ModelTrainArgsType } from '@pipcook/pipcook-core';
import * as path from 'path';
import * as fs from 'fs-extra';

const boa = require('@pipcook/boa');
const tf = boa.import('tensorflow');

const sys = boa.import('sys');
sys.path.insert(0, path.join(__dirname, '..'));
const { train, evaluate } = boa.import('pyscript.index');

const config = tf.compat.v1.ConfigProto();
config.gpu_options.allow_growth = true;
tf.compat.v1.InteractiveSession(boa.kwargs({
  config:config
}));

/**
 * this is plugin used to train tfjs model with pascal voc data format for image classification problem.
 * @param data : train data
 * @param model : model loaded before
 * @param epochs : need to specify epochs
 * @param batchSize : need to specify batch size
 * @param optimizer : need to specify optimizer
 */
const ModelTrain: ModelTrainType = async (data: ImageDataset, model: UniModel, args: ModelTrainArgsType): Promise<UniModel> => {
  try {
    const {
      epochs = 10,
      batchSize = 16,
      modelPath
    } = args;

    const { trainLoader, validationLoader } = data;
    const count = await trainLoader.len();
    const valCount = await validationLoader.len();
    const batchesPerEpoch = Math.floor(count / batchSize);
    const valBatchesPerEpoch = Math.floor(valCount / batchSize);
    const trainModel = model.model;

    for (let i = 0; i < epochs; i++) {
      console.log(`Epoch ${i}/${epochs} start`);
      for (let j = 0; j < batchesPerEpoch; j++) {
        const dataBatch = await data.trainLoader.nextBatch(batchSize);
        train(dataBatch.map((ele) => ele.data), dataBatch.map((ele) => ele.label), trainModel, j, batchesPerEpoch)
      }
      let loss = 0;
      let accuracy = 0;
      for (let j = 0; j < valBatchesPerEpoch; j++) {
        const dataBatch = await validationLoader.nextBatch(batchSize);
        const evaluateRes = evaluate(dataBatch.map((ele) => ele.data), dataBatch.map((ele) => ele.label), trainModel)
        
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
      loss /= valBatchesPerEpoch;
      accuracy /= valBatchesPerEpoch;
      console.log(`Validation Result ${i}/${epochs} result --- loss: ${loss} accuracy: ${accuracy}`);
    }

    await fs.ensureDir(modelPath);
    await trainModel.save_weights(path.join(modelPath, 'weights.h5'));
    await trainModel.save(path.join(modelPath, 'model.h5'));

    const result: UniModel = {
      ...model,
      model: trainModel
    };
    return result;
  } catch (err) {
    console.error('occurs an error on model trainer', err);
    throw err;
  }
};

export default ModelTrain;
