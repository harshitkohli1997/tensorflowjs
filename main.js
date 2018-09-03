 const tf =require('@tensorflow/tfjs');

// Define a model for linear regression.
const model = tf.sequential();
model.add(tf.layers.dense({units: 1, inputShape: [1]}));

// Prepare the model for training: Specify the loss and the optimizer.
model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

// Generate some synthetic data for training.
const xs = tf.tensor2d([1, 2, 3, 4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], [20, 1]);
const ys = tf.tensor2d([1, 4, 9, 16,25,36,49,64,81,100,121,144,169,196,225,256,289,324,361,400], [20, 1]);

// Train the model using the data.
model.fit(xs, ys, {epochs: 10}).then(() => {
  // Use the model to do inference on a data point the model hasn't seen before:
  model.predict(tf.tensor2d([21], [1, 1])).print();
});