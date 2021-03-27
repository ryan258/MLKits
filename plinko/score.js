//const k = 3 // consider the closest k points - if bad results, massage this number, we'll write code to automatically try out different versions of k and get an accuracy %age.

const outputs = []

function onScoreUpdate(dropPosition, bounciness, size, bucketLabel) {
  // Ran every time a balls drops into a bucket
  outputs.push([dropPosition, bounciness, size, bucketLabel])
  // console.log(outputs)
}

function runAnalysis() {
  const testSetSize = 100
  const k = 10

  // const [testSet, trainingSet] = splitDataset(minMax(outputs, 3), testSetSize)

  /*let numberCorrect = 0
  for (let i = 0; i < testSet.length; i++) {
    const bucket = knn(trainingSet, testSet[i][0])
    // console.log(bucket, testSet[i][3]) // get individual pred bucket vs actual bucket
    if (bucket === testSet[i][3]) {
      numberCorrect++
    }
  }

  console.log('Accuracy:', numberCorrect / testSetSize)*/

  _.range(0, 3).forEach((feature) => {
    const data = _.map(outputs, (row) => [row[feature], _.last(row)])
    const [testSet, trainingSet] = splitDataset(minMax(data, 1), testSetSize)

    const accuracy = _.chain(testSet)
      // feature === 0, then feature === 1, then feature === 2
      // .filter((testPoint) => knn(trainingSet, testPoint[0], k) === testPoint[3]) --- we're removing the label
      .filter((testPoint) => knn(trainingSet, _.initial(testPoint), k) === _.last(testPoint))
      .size()
      .divide(testSetSize)
      .value()

    console.log('For feature of', feature, 'Accuracy is', accuracy)
  })
}

const knn = (data, point, k) => {
  // point has 3 values!!! label is gone!
  return (
    _.chain(data)
      // .map((row) => [distance(row[0], point), row[3]])
      .map((row) => {
        return [
          distance(_.initial(row), point), // the point is being made up of test data
          _.last(row)
        ]
      })
      .sortBy((row) => row[0])
      .slice(0, k)
      .countBy((row) => row[1])
      .toPairs()
      .sortBy((row) => row[1])
      .last()
      .first()
      .parseInt()
      .value()
  )
}

const distance = (pointA, pointB) => {
  // pointA = 300, pointB = 350
  // return Math.abs(pointA - pointB)
  //! make it accept any arbitrary # of features
  // pointA = [300, .5, 16], pointB = [350, .55, 16]
  return (
    _.chain(pointA)
      .zip(pointB)
      .map(([a, b]) => (a - b) ** 2)
      .sum()
      .value() ** 0.5
  )
}

// create a function for splitting data set between train and testing
// args (some existing dataset, # of records in our test set)
//  existing dataset, then remove x amount of records and test them against the remaining training results
const splitDataset = (data, testCount) => {
  //--shuffling data before splitting data is usually a good idea
  const shuffled = _.shuffle(data)
  // split data set into 2 and return it
  const testSet = _.slice(shuffled, 0, testCount)
  const trainingSet = _.slice(shuffled, testCount)

  return [testSet, trainingSet]
}

// data - number of initial columns we want to normalize
//        [dropPosition, bounciness, ballSize]
// featureCount - how many to include, we may want to increase or decrease inputs, and bucket is not to be normalized
const minMax = (data, featureCount) => {
  const clonedData = _.cloneDeep(data)
  // now we can normalize the data freely
  // time for the normalization process
  //!iterate over each column (feature)
  for (let i = 0; i < featureCount; i++) {
    const column = clonedData.map((row) => row[i]) // column will become an array of numbers
    const min = _.min(column)
    const max = _.max(column)

    // time to iterate over each value and modify them in place with the min/max function
    //!iterate over each row
    for (let j = 0; j < clonedData.length; j++) {
      //update each value in place - row J of column I - so it picks the row and then which column to update
      //yes we are completely mutating all the values in clonedData
      clonedData[j][i] = (clonedData[j][i] - min) / (max - min)
    }
  }

  return clonedData
}
