const boa = require('./packages/boa');
const glob = require('glob-promise');
const tf = boa.import('tensorflow');
const sys = boa.import('sys');

// sys.path.append('/home/rickycao/Documents/work/pipcook/pipcook/packages/plugins/model-train/image-classification-tensorflow-model-train/piploadlib');
// console.log(sys.path);

// const loadImage = boa.import('loadimage')
// console.log(loadImage.loadImage);

console.log(tf.config.list_physical_devices('GPU'))

const tf1 = boa.import('tensorflow');

console.log(tf1.config.list_physical_devices('GPU'))

// function loadImage(path) {
//   return path;
// }


// async function test() {
//   let imageNames = await glob('/home/rickycao/Documents/work/pipcook/pipcook/pipcook-output/03260850-9d9a-11ea-bc41-b90899252ad8/data/train/*.jpeg');
//   const pathDs = tf.data.Dataset.from_tensor_slices(imageNames);
//   const imageDs = pathDs.map(loadImage.loadImage);
// }

// test()