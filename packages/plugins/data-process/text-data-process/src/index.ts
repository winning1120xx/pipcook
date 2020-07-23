import { DataProcessType, ArgsType, Sample, Metadata } from '@pipcook/pipcook-core'
import { MakeWordsSet } from './scripts';

const boa = require('@pipcook/boa');
const jieba = boa.import('jieba');

const textDataProcess: DataProcessType = async (data: Sample, metadata: Metadata, args: ArgsType): Promise<void> => {
  const {
    stopwordsPath = ""
  } = args;

  let stopwordsSet: Set<string>;

  if (stopwordsPath) {
    stopwordsSet = await MakeWordsSet(stopwordsPath);
  }

  const words: string[] = jieba.cut(data.data, boa.kwargs({cut_all: false}));
  if (stopwordsPath) data.data = words.filter((word) => !stopwordsSet.has(word));
  else data.data = words;
}

export default textDataProcess;
