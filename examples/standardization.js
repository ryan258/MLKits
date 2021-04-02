const numbers = tf.tensor([
  [1, 2],
  [3, 4],
  [5, 6]
])

// we're going to perform standardization on a per column/feature
// basis, so we need to calc a mean for each column
// then adjust each column by their standard

// tf.moments(numbers)

//const {mean, variance} = tf.moments(numbers)
// mean = another name for the average
// variance = the square root of variance is a standard deviation

// tf.moments works the same way as the sum function, so instead of
// using all the values, we call out an axis

// mean
// 3.5
// variance
// 2.9166667461395264

const { mean, variance } = tf.moments(numbers, 0)

mean
// [3,4] - 3 avg for col 1, 4 avg for col 2
variance
// [2.66666, 2.66666]

// apply mean and variance to the numbers tensor
numbers.sub(mean).div(variance.pow(0.5))
// [[-1.2247449, -1.2247449], [0 , 0 ], [1.2247449 , 1.2247449 ]]

// so now we see things scaled down to where they fall in relation to deviations
