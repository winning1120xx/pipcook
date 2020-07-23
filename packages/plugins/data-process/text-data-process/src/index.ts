import { DataProcessType, ArgsType, Sample, Metadata } from '@pipcook/pipcook-core'

const textDataProcess: DataProcessType = async (data: Sample, metadata: Metadata, args: ArgsType): Promise<void> => {
  const {
    stopwordsPath = ""
  } = args;
  
}

export default textDataProcess;
