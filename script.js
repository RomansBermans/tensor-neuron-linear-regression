import { TRAINING_DATA } from 'https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/TrainingData/real-estate-data.js'

const LEARNING_RATE = 0.01
const OPTIMIZER = tf.train.sgd (LEARNING_RATE)

const button = document.getElementById('button')
const info = document.getElementById('info')


function logProgress (epoch, logs) {
  info.innerText += `\n E${epoch + 1} A $${Math.round(Math.sqrt(logs.loss))} V $${Math.round(Math.sqrt(logs.val_loss))}`
  if (epoch === 20) {
    OPTIMIZER.setLearningRate(LEARNING_RATE / 2)
  } else if (epoch === 40) {
    OPTIMIZER.setLearningRate(LEARNING_RATE / 4)
  }
}


function normalize(tensor, min, max) {
  const result = tf.tidy(() => {
    const normalizedMinTensor = min || tensor.min(0)
    const normalizedMaxTensor = max || tensor.max(0)
    
    const subtractMinTensor = tensor.sub(normalizedMinTensor)
    
    const rangeSizeTensor = normalizedMaxTensor.sub(normalizedMinTensor)
    
    const normalizedTensor = subtractMinTensor.div(rangeSizeTensor)

    return { normalizedTensor, normalizedMinTensor, normalizedMaxTensor }
  })
  return result
}


function prepare(inputs, outputs) {
  const timeStart = performance.now()
  
  // Shuffle the two arrays in the same way so inputs still match outputs indexes
  tf.util.shuffleCombo(inputs, outputs)

  const inputsTensor = tf.tensor2d(inputs)
  const outputsTensor = tf.tensor1d(outputs)

  const { normalizedTensor, normalizedMinTensor, normalizedMaxTensor } = normalize(inputsTensor)
  info.innerText += `\n Normalized Inputs ${normalizedTensor.toString()}`
  info.innerText += `\n\n Normalized Min ${normalizedMinTensor.toString()}`
  info.innerText += `\n\n Normalized Max ${normalizedMaxTensor.toString()}`
  
  inputsTensor.dispose()

  const model = tf.sequential()
  model.add(tf.layers.dense({ inputShape: [2], units: 1 }))
  model.summary()
  
  const modelPrepareTime = performance.now() - timeStart
  info.innerText += `\n\n ⏳ ${Math.round(modelPrepareTime)}MS`
  
  return { model, normalizedTensor, normalizedMinTensor, normalizedMaxTensor, outputsTensor }
}


async function train(optimizer, model, normalizedTensor, outputsTensor) {
  const timeStart = performance.now()

  model.compile ({ 
    optimizer,
    loss: 'meanSquaredError'
  })

  const results = await model.fit(normalizedTensor, outputsTensor, {
    callbacks: { onEpochEnd: logProgress },
    validationSplit: 0.15,  // Take aside 15% of the data to use for validation testing
    shuffle: true,          // Ensure data is shuffled in case it was in an order
    batchSize: 64,          // As we have a lot of training data, batch size is set to 64
    epochs: 60              // Go over the data 10 times
  })
  
  normalizedTensor.dispose()
  outputsTensor.dispose()
  
  const modelTrainTime = performance.now() - timeStart
  info.innerText += `\n\n ⏳ ${Math.round(modelTrainTime)}MS`
}


async function evaluate(inputs, model, normalizedMinTensor, normalizedMaxTensor) {
  const timeStart = performance.now()
  
  const predictionTensor = tf.tidy(() => {
    const { normalizedTensor } = normalize(tf.tensor2d(inputs), normalizedMinTensor, normalizedMaxTensor)

    const result = model.predict(normalizedTensor)
    return result
  })
  
  const predictions = await predictionTensor.array()
  predictionTensor.dispose()
  
  predictions.forEach((prediction, index) => {
    info.innerText += `\n Result: [${inputs[index]}] = $${Math.round(prediction)}`
  })
  
  normalizedMinTensor.dispose()
  normalizedMaxTensor.dispose()
  model.dispose()
  
  const modelEvaluateTime = performance.now() - timeStart
  info.innerText += `\n\n ⏳ ${Math.round(modelEvaluateTime)}MS`
}


button.addEventListener('click', async event => {
  event.target.remove()

  info.innerText = '⎯⎯⎯⎯⎯⎯⎯ PREPARE ⎯⎯⎯⎯⎯⎯⎯\n'
  const { inputs, outputs } = TRAINING_DATA
  const { model, normalizedTensor, normalizedMinTensor, normalizedMaxTensor, outputsTensor } = prepare(inputs, outputs)
  
  info.innerText += '\n\n ⎯⎯⎯⎯⎯⎯⎯ TRAIN ⎯⎯⎯⎯⎯⎯⎯\n'
  await train(OPTIMIZER, model, normalizedTensor, outputsTensor)
  
  // TODO:
  // await model.save('downloads://my-model')
  // const model = await tf.loadLayersModel('http://yoursite.com/model.json')
  // await model.save ('localstorage://demo/newModelName')
  // const model = await tf.loadLayersModel('localstorage://demo/newModelName')
  
  info.innerText += '\n\n ⎯⎯⎯⎯⎯⎯⎯ EVALUATE ⎯⎯⎯⎯⎯⎯⎯\n'
  await evaluate([[750, 1], [750, 2], [750, 3], [1500, 1], [1500, 2], [1500, 3], [2500, 2], [2500, 4]], model, normalizedMinTensor, normalizedMaxTensor)
  
  info.innerText += '\n\n ⎯⎯⎯⎯⎯⎯⎯ MEMORY ⎯⎯⎯⎯⎯⎯⎯\n'
  
  info.innerText += `\n 💾 ${(tf.memory().numBytes / 1000000).toFixed(2)}MB (${tf.memory().numTensors})`
})