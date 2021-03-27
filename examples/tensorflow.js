//! Elementwise
/*// tf = the main tensorflow object
// create a tensor

const data = tf.tensor([1, 2, 3])
// const data = tf.tensor([
//   [1, 2, 3]
// ])
// now we can call methods on that tensor

// time to care about the shape of the data obj
data.shape // logs [3]
// data.shape // logs [1, 3] - ROW, COLUMN!
console.log('2')
*/
//! Elementwise
/*const data = tf.tensor([
  [1, 2, 3],
  [4, 5, 6]
])
const otherData = tf.tensor([
  [4, 5, 6],
  [1, 2, 3]
])

// the big thing is two tensors and doing math ops on them together

data.add(otherData) 
// [5, 7, 9]
// this was an example of an elementwise operation, adding two tensors together to create a new tensor
// the output of a tensor operation on a tensor will result in a new tensor
data
otherData
// ^^ these 2 are still the same, so immutable! operations never change the originals, 
//    they always result in a new tensor

data.add(otherData)
// [-3, -3, -3]

*/
//! Broadcasting
//! TENSOR ACCESSORS
/*// const data = tf.tensor([10, 20, 30])
// data.get(0)
// 10

const data = tf.tensor([
  [10, 20, 30],
  [40, 50, 60],
])

data.get(1, 1)
// 50
// THERE IS NO .set() !
*/

//! CREATING SLICES OF DATA
/*// access many elements at 1 time

const data = tf.tensor([
  [10, 20, 30],
  [40, 50, 60],
  [10, 20, 30],
  [40, 50, 60],
  [10, 20, 30],
  [40, 50, 60],
  [10, 20, 30],
  [40, 50, 60]
])

// let's get just the center column
// data.slice([0, 1], [8, 1])
// [[20], [50], [20], [50], [20], [50], [20], [50]]

// but hardcoding the length is trouble... so!
data.shape
// [8, 3] - our first element will be the length
data.slice([0, 1], [data.shape[0], 1])
// [[20], [50], [20], [50], [20], [50], [20], [50]]
// :D

// and here's an alternate way of that!
// -1 is a special value that says "give me all the rows!"
data.slice([0, 1], [-1, 1])
// [[20], [50], [20], [50], [20], [50], [20], [50]]
// :D !!!

// so really it's usually going to look like this
// data.slice([0, 1], [-1, 1])
// and the thing that changes is the index of the column we want
// data.slice([0, X], [-1, 1])

*/

//! TENSOR CONCATENATION
/* // join together 2 tensors - concatenate

const tensorA = tf.tensor([
  [1, 2, 3],
  [4, 5, 6]
])

const tensorB = tf.tensor([
  [7, 8, 9],
  [10, 11, 12]
])

tensorA.concat(tensorB)
// [[1 , 2 , 3 ], [4 , 5 , 6 ], [7 , 8 , 9 ], [10, 11, 12]]
tensorA.concat(tensorB).shape
// [4, 3]

tensorA.concat(tensorB, 1)
// [[1, 2, 3, 7 , 8 , 9 ], [4, 5, 6, 10, 11, 12]]
tensorA.concat(tensorB, 1).shape
// [2, 6] -- 2 rows & 6 columns
*/

//! LOONG JUMP EXERCISE
// long jump competition
const jumpData = tf.tensor([
  [70, 70, 70],
  [70, 70, 70],
  [70, 70, 70],
  [70, 70, 70]
])

const playerData = tf.tensor([
  [1, 160],
  [2, 160],
  [3, 160],
  [4, 160]
])

// first we need to sum together the jump data for each player
jumpData.sum()
// 840 - by default it is adding together all the values in a tensor
// but we want to sum along an axis
jumpData.sum(1)
// [210, 210, 210, 210] - grabs the horizonal data and lines up sums
jumpData.sum(0)
// [280, 280, 280] - summed along y-axis

// nw we want to concat to player data
//jumpData.sum(1).concat(playerData)
// rank error: the concat we are trying to do is of different dimensions
// even though it seems like they are both 2 dimensional...

jumpData.sum(1).shape
// [4]
// ?
jumpData.shape
// [4,3]
// so when you call the .sum it reduces the dimension of a tensor...
// then when you concat, no dice

// simple not so useful solution
//// (optional 2nd argument) - the keep dimension argument
jumpData.sum(1, true)
// [[210], [210], [210], [210]]
jumpData.sum(1, true).shape
// [4, 1]
// now we can concat it
jumpData.sum(1, true).concat(playerData, 1)
// [[210, 1, 160], [210, 2, 160], [210, 3, 160], [210, 4, 160]]

// more complex but useful in more situations solition where methods reduce dimensions
//! ROBUST - but complex solution
// more complex but useful in more situations solition where methods reduce dimensions

jumpData.sum(1)
// [210, 210, 210, 210]
jumpData.sum(1).expandDims()
// [[210, 210, 210, 210],] - expends tensor dimension by 1
jumpData.sum(1).expandDims().shape
// [1, 4] - 1 row w/ 4 columns
jumpData.sum(1).expandDims(1).shape
// [4, 1] - 4 rows in a single column
jumpData.sum(1).expandDims(1)
// [[210], [210], [210], [210]]
jumpData.sum(1).expandDims(1).concat(playerData, 1)
// [[210, 1, 160], [210, 2, 160], [210, 3, 160], [210, 4, 160]]
// booya!
