require('@tensorflow/tfjs-node')
const tf = require('@tensorflow/tfjs')

const loadCSV = require('./load-csv')

function knn(features, labels, predictionPoint, k) {
  const { mean, variance } = tf.moments(features, 0)

  const scaledPrediction = predictionPoint.sub(mean).div(variance.pow(0.5))

  return (
    features
      .sub(mean)
      .div(variance.pow(0.5))
      // .sub(predictionPoint)
      .sub(scaledPrediction)
      // now that we've standardized, business as usual
      .pow(2)
      .sum(1)
      .pow(0.5)
      .expandDims(1)
      .concat(labels, 1)
      .unstack()
      .sort((a, b) => (a.get(0) > b.get(0) ? 1 : -1))
      .slice(0, k)
      .reduce((acc, pair) => acc + pair.get(1), 0) / k
  )
}

// ALL THESE LET VARIABLES ARE ARRAYS RIGHT NOW!
// LoadCSV is just gonna load, it knows nothing about what a tensor is
let { features, labels, testFeatures, testLabels } = loadCSV('kc_house_data.csv', {
  // shuffle all the data so we don't take biased data for our training data
  shuffle: true,
  // instruct the loader to give us 2 different sets of data
  // 1 for train, the other for testing - enter # of records to test
  splitTest: 10, // use a test of 10 records - so we get 10 records out as a test set, then the rest as a training set
  dataColumns: ['lat', 'long', 'sqft_lot', 'sqft_living'], //FEATURES EXCEPT FOR 10 IN OUR TEST GROUP columns to grab (by header name)
  labelColumns: ['price'] //LABEL like data columns, but extracts 1+ column and assigns them to our 'labels' data set - so... things we're trying to predict
})

// console.log(testFeatures)
// console.log(testLabels)

// ALL THE VARIABLES ARE ARRAYS RIGHT NOW!
// Sp we have to convert them before we pass them in to our function
//! HERE WE'LL MAKE THOSE ARRAYS FROM loadCSV INTO TENSORS
features = tf.tensor(features)
labels = tf.tensor(labels)
// we'll just take 1 of each from the original because we want to test 1 at a time
// testFeatures = tf.tensor(testFeatures)
// testLabels = tf.tensor(testLabels)

//! ITERATION #1
// now we can pass them all in
// function knn(features, labels, predictionPoint, k) { ...
// const result = knn(features, labels, tf.tensor(testFeatures[0]), 10)

// testLabels - an array of arrays - what the actual value of the property is (it's in testLabels) and grab the first row, inside that row we'll get the number value out of it
// console.log('Guess', result, testLabels[0][0])
// Guess 1421200 1085000
// we guessed it would be 1421200
// turned out to be worth 1085000

//we need to evaluate how good or bad any of our guesses is

//! ITERATION #2 - how good/bad the guess was
// const result = knn(features, labels, tf.tensor(testFeatures[0]), 10)
// const err = (testLabels[0][0] - result) / testLabels[0][0]
// console.log('Guess:', result, testLabels[0][0])
// console.log('Error:', err * 100)
// Guess: 1421200 1085000
// Error: -30.98617511520737

//! ITERATION #3 - now we'll loop over all our test samples...? features?
// testFeatures.forEach((testPoint, index) => {
//   const result = knn(features, labels, tf.tensor(testPoint), 10)
//   // look at the correct row of testLabels
//   const err = (testLabels[index][0] - result) / testLabels[index][0]
//   // console.log('Guess:', result, testLabels[0][0])
//   console.log('Error:', err * 100)
// })

// Error: -30.98617511520737
// Error: -52.95661953727506
// Error: -9.552941176470588
// Error: -28.528495575221243
// Error: -6.069828722002635
// Error: -9.855653270993358
// Error: -11.176432291666668
// Error: 43.34094616639478
// Error: -19.536472310319592
// Error: -5.603238866396762

// OK so obviously there is more to predicting home values than just lat/long...
// it's time to look at more factors!

//! ITERATION #4 - let's consider the sqft_lot feature and implement STANDARDIZATION!
testFeatures.forEach((testPoint, index) => {
  const result = knn(features, labels, tf.tensor(testPoint), 10)
  // look at the correct row of testLabels
  const err = (testLabels[index][0] - result) / testLabels[index][0]
  // console.log('Guess:', result, testLabels[0][0])
  console.log('Error:', err * 100)
})
