//! the ordering is very important because the indexes correspond w/ eachother
// lat, long
const features = tf.tensor([
  [-121, 47],
  [-121.2, 46.5],
  [-122, 46.4],
  [-120.9, 46.7]
])

// house prices
const labels = tf.tensor([[200], [250], [215], [240]])

// prediction point of where we're trying to calc the price for
const predictionPoint = tf.tensor([-121, 47])

// oh the k value
const k = 2

features
  //! FIRST find distance between features and prediction point
  // we'd use the pythegrium theorum to do that
  // [[-121 , 47 ], [-121.1999969, 46.5 ], [-122 , 46.4000015], [-120.9000015, 46.7000008]]
  // so get the difference first by broadcasting the prediction point, smudging the feature values
  .sub(predictionPoint)
  // [[0 , 0 ], [-0.1999969, -0.5 ], [-1 , -0.5999985], [0.0999985 , -0.2999992]]
  // next part of pTheorum is to square each value
  .pow(2)
  // [[0 , 0 ], [0.0399988, 0.25 ], [1 , 0.3599982], [0.0099997, 0.0899995]]
  // then we add them all together (horizontally)
  .sum(1)
  // [0, 0.2899988, 1.3599982, 0.0999992]
  // finally we perform the square root operation of the pTheorum
  .pow(0.5)
  // [0, 0.5385153, 1.1661896, 0.3162265] - a 1D tensor
  //! SECOND.ONE - sort from lowest point to greatest - Merge distances w/ labels
  //  .concat(labels)
  // so we have different shapes that don't match so we can't concat together...
  // nows the time for expandDems to manipulate the shape so they can match passing 1 to change the axis
  .expandDims(1)
  // [[0 ], [0.5385153], [1.1661896], [0.3162265]]
  // and now we can concat our labels on to the values - passing 1 to concat on the horz axis
  .concat(labels, 1)
  // [[0 , 200], [0.5385153, 250], [1.1661896, 215], [0.3162265, 240]]
  //! SECOND.TWO - sort from lowest point to greatest - Merge distances w/ labels
  // so there's no sort function in the tensor library...
  // so we'll break it down into a bunch of individual tensors, 1 for each row, and make an array of them
  //	.unstack()[3] // [0.3162265, 240]
  .unstack()
  // so we are now no longer working with a tensor, but a normal JS ARRAY
  // now that we have the array, we just have to sort it
  // so we can now use the built-in JS .sort method
  .sort((a, b) => (a.get(0) > b.get(0) ? 1 : -1))
  // now they are sorted smallest to largest distance
  //! THIRD - TAKE TOP K RECORDS
  // for that we can use slice - this is the vanilla JS slice
  // and we'll grab the first 2 array items
  //	.slice(0, k).length // 2
  .slice(0, k)
  // now we just have an array with the top 2 closest tensors to the predicted point
  // now we pull our label out of each of these to find the average
  //! FINALLY - find the average values of the k prices
  // so get the sum and divide by k
  .reduce((acc, pair) => acc + pair.get(1), 0) / k
// 220 is our predicted price! 🥳
