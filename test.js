const boa = require('./packages/boa');
const glob = require('glob-promise');
const tf = boa.import('tensorflow');

function loadImage(path) {
  return path;
}


async function test(end) {
  let imageNames = await glob('/home/rickycao/Documents/work/pipcook/pipcook/pipcook-output/03260850-9d9a-11ea-bc41-b90899252ad8/data/train/*.jpeg');
  imageNames = imageNames.slice(0, end);
  const pathDs = tf.data.Dataset.from_tensor_slices(imageNames);
  const imageDs = pathDs.map(loadImage);
}

test(2000);