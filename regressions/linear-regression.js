const tf = require('@tensorflow/tfjs')
const _ = require('lodash')

class LinearRegression {
  // constructor is called whenever we make a new instance of this class
  constructor(features, labels, options) {
    // and initial set up of our class (traditionally) w/ feature set, labels, and relevant options for running this algo
    // we're assuming that when features and labels are passed in, that they will already be tensorflow tensors
    this.features = features
    this.labels = labels
    // and we'll also process the options
    this.options = Object.assign(
      {
        // here we can add default values - to make sure there's always a value provided for crucial components
        learningRate: 0.1,
        // specify maximum number of times we want to run our GDesc algo
        iterations: 1000
      },
      options
    )

    // create initial guesses for m & b
    this.m = 0
    this.b = 0
  }
  // gradient descent - purpose is to calc slope values for m & b - then use those to update our current guesses of m & b
  // - just 1 iterations
  // - then our training method will repeatedly run this function so we eventually get an optimal solution of m & b together
  gradientDescent() {
    const currentGuessesForMPG = this.features.map((row) => {
      // calculate that (mx + b) -- row[0] is the horsepower value
      return this.m * row[0] + this.b
    }) // now we have an array of data that represents all the mx+b terms
    // now we can iterate through all those guesses and subtract the actual values, then sum them all together -- then divide all that by the * of observations we have -- and then multiply it by 2
    // - i is the optional 2nd arguement of map that is the current index you are mapping over
    const bSlope =
      (_.sum(
        currentGuessesForMPG.map((guess, i) => {
          return guess - this.labels[i][0] // the actual MPG value
        })
      ) *
        2) /
      this.features.length

    const mSlope =
      (_.sum(
        currentGuessesForMPG.map((guess, i) => {
          return -1 * this.features[i][0] * (this.labels[i][0] - guess)
        })
      ) *
        2) /
      this.features.length

    // now we just need to take our slopes and multiply them by the learning rate, then subtract m & b by that product
    this.m = this.m - mSlope * this.options.learningRate
    this.b = this.b - bSlope * this.options.learningRate
  }

  // the train() will run gradiant decline until we get acceptable values for m and b
  train() {
    // it's possible that we'll never come across appropriate values for m and b
    for (let i = 0; i < this.options.iterations; i++) {
      this.gradientDescent()
    }
  }
}

module.exports = LinearRegression
