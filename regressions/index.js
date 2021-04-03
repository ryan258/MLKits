// we're going to
// - load up our data
// - reference our linear regression class
// - do our training
// - testing
// - validate accuracy

// we're going to put our actual code for the gradient descent algo and LinearRegression class will be in a seperate file
// we only need to require the -node version of tfjs, preferably at the root file that gets run by node
require('@tensorflow/tfjs-node')

const tf = require('@tensorflow/tfjs')
const loadCSV = require('./load-csv')
const LinearRegression = require('./linear-regression')

let { features, labels, testFeatures, testLabels } = loadCSV('./cars.csv', {
  shuffle: true,
  // in practice with huge data sets you'll use like 50% of the pool for a test set and the other 50% in the training data
  splitTest: 50,
  // columns to pull out and place in the features & test features array
  // we'll pick horsepower bc we're trying to prove there might be a relationship between horsepower & miles per gallon
  // miles per gallon = m * (car horsepower) + b
  dataColumns: ['horsepower'],
  // label columns - what in the CSV file is going to represent  date we want to use as our label, the thing we're trying to predict
  labelColumns: ['mpg']
})

// console.log(features, labels)

const regression = new LinearRegression(features, labels, {
  // override fallback defaults
  learningRate: 0.0001,
  iterations: 100
})

regression.train()

console.log('Updated M is:', regression.m, 'Updated B is:', regression.b)
