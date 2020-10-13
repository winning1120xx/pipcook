import { app, mm, assert } from 'midway-mock/bootstrap';
import * as fs from 'fs-extra';
import * as sinon from 'sinon';
import * as pipcookApp from '@pipcook/app';
import { AppService } from '../../src/service/app';

describe('test the pipeline service', () => {
  afterEach(() => {
    mm.restore();
    sinon.restore();
  });

  it('#should return corresponding pipelines', async () => {
    const appService: AppService = await app.applicationContext.getAsync<AppService>('appService');
    const mockCompile = sinon.stub(pipcookApp, 'compile').resolves({
      pipelines: [
        {
          signature: 'mock signature',
          config: {
            plugins: {
              dataCollect: { package: 'mock'},
              dataAccess: { package: 'mock'},
              modelDefine: { package: 'mock'},
              modelTrain: { package: 'mock'},
              modelEvaluate: { package: 'mock'}
            }
          },
          namespace: {
              module: 'vision',
              method: 'mock'
          },
        }
      ],
      nlpReferences: [],
      visionReferences: []
    });
    sinon.stub(fs, 'readFile').resolves(Buffer.from('mock'));
    await appService.compile(`console.log('mock test')`);
    assert.ok(mockCompile.calledOnce, 'mock should be called once');
  });
});