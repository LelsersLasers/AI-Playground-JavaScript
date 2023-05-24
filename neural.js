
class DataPoint {
	constructor(inputs, outputs) {
		this.inputs = inputs; // [X, Y] (0.0 - 1.0)
		this.outputs = outputs; // [R, G, B] (0.0 - 1.0)
	}
}

class Layer {
	constructor(numInputs, numOutputs, activationFunction, regularizationFunction, weightInitFunction) {
		this.numInputs = numInputs;
		this.numOutputs = numOutputs;
		
		this.weights = [];
		this.bias = [];

		this.weightGradients = [];
		this.biasGradients = [];


		for (let i = 0; i < numOutputs; i++) {
			this.weights.push([]);
			this.weightGradients.push([]);

			for (let j = 0; j < numInputs; j++) {
				this.weights[i].push(weightInitFunction());
				this.weightGradients[i].push(0);
			}

			this.bias.push(weightInitFunction());
			this.biasGradients.push(0);
		}

		this.activationFunction = activationFunction;
		this.regularizationFunction = regularizationFunction;
	}
	forwardPass(inputs) {
		const outputs = [];
		for (let i = 0; i < this.numOutputs; i++) {
			let sum = this.bias[i];
			for (let j = 0; j < this.numInputs; j++) {
				sum += inputs[j] * this.weights[i][j];
			}
			outputs.push(this.activationFunction(sum));
		}
		return outputs;
	}
	applyGradients(learningRate) {
		for (let i = 0; i < this.numOutputs; i++) {
			for (let j = 0; j < this.numInputs; j++) {
				this.weights[i][j] -= this.weightGradients[i][j] * learningRate;
				this.weightGradients[i][j] = 0;
			}
			this.bias[i] -= this.biasGradients[i] * learningRate;
			this.biasGradients[i] = 0;
		}
	}
	regularizationCost() {
		let cost = 0;
		for (let i = 0; i < this.numOutputs; i++) {
			for (let j = 0; j < this.numInputs; j++) {
				cost += this.regularizationFunction(this.weights[i][j]);
			}
			cost += this.regularizationFunction(this.bias[i]);
		}
		return cost / (this.numInputs * this.numOutputs + this.numOutputs)
	}
}

class NeuralNetwork {
	constructor(layers, learningRate, regularizationRate, h) {
		this.layers = layers;
		this.numLayers = layers.length;

		this.learningRate = learningRate;
		this.regularizationRate = regularizationRate;

		this.h = h;
	}
	forwardPass(inputs) {
		let outputs = inputs;
		for (let i = 0; i < this.layers.length; i++) {
			outputs = this.layers[i].forwardPass(outputs);
		}
		return outputs;
	}
	costOfOne(dataPoint) {
		const outputs = this.forwardPass(dataPoint.inputs);
		let cost = 0;
		for (let i = 0; i < outputs.length; i++) {
			cost += (outputs[i] - dataPoint.outputs[i]) ** 2;
		}
		return cost;
	}
	costOfAll(dataPoints) {
		let cost = 0;
		for (let i = 0; i < dataPoints.length; i++) {
			cost += this.costOfOne(dataPoints[i]);
		}
		for (let i = 0; i < this.layers.length; i++) {
			cost += this.layers[i].regularizationCost() * this.regularizationRate;
		}
		return cost / dataPoints.length;
	}
	applyGradients() {
		for (let i = 0; i < this.layers.length; i++) {
			this.layers[i].applyGradients(this.learningRate);
		}
	}
	learn(dataPoints) {
		const startCost = this.costOfAll(dataPoints);

		for (let l = 0; l < this.numLayers; l++) {
			for (let i = 0; i < this.layers[l].numOutputs; i++) {
				for (let j = 0; j < this.layers[l].numInputs; j++) {

					this.layers[l].weights[i][j] += this.h;
					const newCost = this.costOfAll(dataPoints);
					const gradient = (newCost - startCost) / this.h;
					this.layers[l].weights[i][j] -= this.h;

					this.layers[l].weightGradients[i][j] = gradient;
				}

				this.layers[l].bias[i] += this.h;
				const newCost = this.costOfAll(dataPoints);
				const gradient = (newCost - startCost) / this.h;
				this.layers[l].bias[i] -= this.h;

				this.layers[l].biasGradients[i] = gradient;
			}
		}
	}
	learnIterate(dataPoints, iterations) {
		for (let i = 0; i < iterations; i++) {
			this.learn(dataPoints);
			this.applyGradients();
		}
	}
}

const relu = x => Math.max(0, x);
const leakyRelu = x => Math.max(0.01 * x, x);
const sigmoid = x => 1 / (1 + Math.exp(-x));
const tanh = x => Math.tanh(x);
const linear = x => x;

const L1 = x => Math.abs(x);
const L2 = x => x ** 2;
const noRegulation = x => 0;

const randomWeight = () => (Math.random() - 0.5) / 2; // -0.25 to 0.25