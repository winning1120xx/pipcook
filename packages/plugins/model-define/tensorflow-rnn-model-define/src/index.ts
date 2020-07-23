/**
 * @file This is for the plugin to load Bayes Classifier model.
 */

import { ModelDefineType, UniModel, ModelDefineArgsType, CsvDataset, CsvSample } from '@pipcook/pipcook-core';
import * as assert from 'assert';
import * as path from 'path';
import { processPredictData } from './script';

const boa = require('@pipcook/boa');
const sys = boa.import('sys');
const tf = boa.import('tensorflow');
const { Adam } = boa.import('tensorflow.keras.optimizers');
const { Embedding, Bidirectional, LSTM, Dense, Dropout } = boa.import('tensorflow.keras.layers')
const { Sequential } = boa.import('tensorflow.keras')

/**
 * assertion test
 * @param data
 */
const assertionTest = (data: CsvDataset) => {
  assert.ok(data.metadata.feature && data.metadata.feature.names.length === 1,
    'feature should only have one dimension which is the feature name');
};

/**
 * Pipcook Plugin: bayes classifier model
 * @param data Pipcook uniform sample data
 * @param args args. If the model path is provided, it will restore the model previously saved
 */
const rnnClassifierModelDefine: ModelDefineType = async (data: CsvDataset, args: ModelDefineArgsType): Promise<UniModel> => {
  const {
    loss = 'categorical_crossentropy',
    metrics = [ 'accuracy' ],
    learningRate = 0.001,
    decay = 0.05,
    recoverPath
  } = args;

  sys.path.insert(0, path.join(__dirname, 'assets'));
  let classifier = Sequential([
    Embedding(1000, 60),
    Bidirectional(LSTM(64, boa.kwargs({
      return_sequences: true
    }))),
    Bidirectional(LSTM(32)),
    Dense(64, boa.kwargs({
      activation: 'relu'
    })),
    Dropout(.5).
    Dense(10)
  ]);

  classifier.compile(boa.kwargs({
    loss,
    metrics,
    optimizer: Adam(boa.kwargs({
      lr: learningRate,
      decay
    }))
  }));

  // if (!recoverPath) {
  //   assertionTest(data);
  //   classifier = getBayesModel();
  // } else {
  //   classifier = await loadModel(path.join(recoverPath, 'model.pkl'));
  // }

  const pipcookModel: UniModel = {
    model: classifier,
    predict: async function (text: CsvSample) {
      const processData = await processPredictData(text.data, path.join(recoverPath, 'feature_words.pkl'), path.join(recoverPath, 'stopwords.txt'));
      const pred = this.model.predict(processData);
      return pred.toString();
    }
  };
  return pipcookModel;
};

export default rnnClassifierModelDefine;
