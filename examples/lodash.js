// lodash example #1

const outputs = [
  [10, 0.5, 16, 1],
  [200, 0.5, 16, 4],
  [350, 0.5, 16, 4],
  [600, 0.5, 16, 5]
]

const predictionPoint = 300
const k = 3

const distance = (point) => Math.abs(point - predicitionPoint)

_.chain(outputs)
  .map((row) => [distance(row[0]), row[3]])
  .sortBy((row) => row[0])
  .slice(0, k)
  .countBy((row) => row[1])
  .toPairs()
  .sortBy((row) => row[1])
  .last()
  .first()
  .parseInt()
  .value()
