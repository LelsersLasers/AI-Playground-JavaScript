const RADIUS_RATIO = 0.01;

let resolution = 100;

const canvas = document.getElementsByTagName("canvas")[0];
canvas.addEventListener("click", function(event) {
    const rect = canvas.getBoundingClientRect();

    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    const inputs = [x / canvas.width, y / canvas.height];
    const outputs = selectedColor;

    dataPoints.push(new DataPoint(inputs, outputs));
});

const context = canvas.getContext("2d");


let globalActivation = leakyRelu;
let globalRegularization = noRegulation;
let globalLayers = [
	new Layer(2, 6, globalActivation, globalRegularization, randomWeight),
	new Layer(6, 3, globalActivation, globalRegularization, randomWeight),
];
let globalLearningRate = 0.1;
let globalRegularizationRate = 0.001;
const H = 0.0001;
let network = new NeuralNetwork(globalLayers, globalLearningRate, globalRegularizationRate, H);

let selectedColor = [1.0, 0.0, 0.0];

const dataPoints = [
	// new DataPoint([0.2, 0.2], [1.0, 0.0, 0.0]), // red
    // new DataPoint([0.8, 0.8], [0.0, 1.0, 0.0]), // green
	// new DataPoint([0.2, 0.8], [0.0, 0.0, 1.0]), // blue
    // new DataPoint([0.8, 0.2], [1.0, 1.0, 0.0]), // yellow

    new DataPoint([0.15, 0.15], [1.0, 0.0, 0.0]), // red
    new DataPoint([0.85, 0.85], [0.0, 0.0, 1.0]), // blue


    // new DataPoint([0.15, 0.15], [1.0, 0.0, 0.0]), // red
    // new DataPoint([0.85, 0.15], [1.0, 0.0, 0.0]), // red

    // new DataPoint([0.15, 0.5], [0.0, 1.0, 0.0]), // green
    // new DataPoint([0.5, 0.5], [0.0, 0.0, 1.0]), // blue
    // new DataPoint([0.85, 0.5], [0.0, 1.0, 0.0]), // green

    // new DataPoint([0.15, 0.85], [0.0, 0.0, 1.0]), // blue
    // new DataPoint([0.85, 0.85], [0.0, 0.0, 1.0]), // blue
];


let paused = false;

function apply() {
    // NOTE: order matters

    updateActivation(); // sets globalActivation
    updateRegularization(); // sets globalRegularization

	updateLayers(); // sets globalLayers (uses globalActivation and globalRegularization)
    updateLearningRate(); // sets globalLearningRate
    updateRegularizationRate(); // sets globalRegularizationRate

    network = new NeuralNetwork(globalLayers, globalLearningRate, globalRegularizationRate, H);
    console.log(network);
}

function updateLayers() {
	const hiddenLayers = document.getElementById("HIDDEN_LAYERS");

	const valTrimmed = hiddenLayers.value.replace(/\s/g, '');
	let valSplit = valTrimmed.split(",");

	if (valTrimmed.length == 0) {
		valSplit = [];
	}

	for (let i = 0; i < valSplit.length; i++) {
		const numNeurons = parseInt(valSplit[i]);
		if (isNaN(numNeurons) || numNeurons < 0 || numNeurons > 100) {
            hiddenLayers.style.border = "2px solid #BF616A";
            return;
        }
        valSplit[i] = numNeurons;
	}

	const layers = [];
	let lastSize = 2;
	for (let i = 0; i < valSplit.length; i++) {
		layers.push(new Layer(lastSize, valSplit[i], globalActivation, globalRegularization, randomWeight));
		lastSize = valSplit[i];
	}
	layers.push(new Layer(lastSize, 3, globalActivation, globalRegularization, randomWeight));


    globalLayers = layers;

	hiddenLayers.innerHTML = valSplit.join(", ");
	hiddenLayers.style.border = "none";
}
function updateActivation() {
    const activationSelect = document.getElementById("ACTIVATION");
    switch (activationSelect.value) {
        case "relu": globalActivation = relu; break;
        case "leakyRelu": globalActivation = leakyRelu; break;
        case "sigmoid": globalActivation = sigmoid; break;
        case "tanh": globalActivation = tanh; break;
        case "linear": globalActivation = linear; break;
    }
}
function updateLearningRate() {
    const learningRate = document.getElementById("LEARNING_RATE");

    const parsed = parseFloat(learningRate.value);
    if (isNaN(parsed) || parsed < 0 || parsed > 1) {
        learningRate.style.border = "2px solid #BF616A";
        return;
    }

    globalLearningRate = parsed;
    
    learningRate.innerHTML = parsed;
    learningRate.style.border = "none";
}
function updateRegularization() {
    const regularizationSelect = document.getElementById("REGULARIZATION");
    switch (regularizationSelect.value) {
        case "noRegulation": globalRegularization = noRegulation; break;
        case "L1": globalRegularization = L1; break;
        case "L2": globalRegularization = L2; break;
    }
}
function updateRegularizationRate() {
    const regularizationRate = document.getElementById("REGULARIZATION_RATE");

    const parsed = parseFloat(regularizationRate.value);
    if (isNaN(parsed) || parsed < 0 || parsed > 1) {
        regularizationRate.style.border = "2px solid #BF616A";
        return;
    }

    globalRegularizationRate = parsed;

    regularizationRate.innerHTML = parsed;
    regularizationRate.style.border = "none";
}


function setOnChangeForRadioButtons() {
    const radioButtons = document.querySelectorAll("input[type=radio][name='COLOR']");
    radioButtons.forEach(radioButton => {
        radioButton.addEventListener("change", function() {
            switch (this.id) {
                case "red": selectedColor = [1.0, 0.0, 0.0]; break;
                case "green": selectedColor = [0.0, 1.0, 0.0]; break;
                case "blue": selectedColor = [0.0, 0.0, 1.0]; break;
                case "white": selectedColor = [1.0, 1.0, 1.0]; break;
                case "black": selectedColor = [0.0, 0.0, 0.0]; break;
                case "custom": selectedColor = applyColorChange("CUSTOM_COLOR"); break;
            }
        });
    });

    const customColor = document.getElementById("CUSTOM_COLOR");
    customColor.addEventListener("input", function() {
        if (document.getElementById("custom").checked) {
            selectedColor = applyColorChange("CUSTOM_COLOR");
        }
    });
}
function applyColorChange(id) {
    const val = document.getElementById(id).value;
    const valTrimmed = val.replace(/\s/g, '');
    let valSplit = valTrimmed.split(",");

    if (valSplit.length != 3) {
        document.getElementById(id).style.border = "2px solid #BF616A";
        return;
    }

    for (let i = 0; i < valSplit.length; i++) {
        let valSplitInt = parseInt(valSplit[i]);
        if (isNaN(valSplitInt) || valSplitInt < 0 || valSplitInt > 255) {
            document.getElementById(id).style.border = "2px solid #BF616A";
            return;
        }
        valSplit[i] = valSplitInt;
    }

    document.getElementById(id).value = valSplit.join(", ");
    document.getElementById(id).style.border = "none";

    return [valSplit[0] / 255, valSplit[1] / 255, valSplit[2] / 255];
}

function setOnChangeForResolution() {
    const resolutionElement = document.getElementById("RESOLUTION");
    resolutionElement.addEventListener("input", function() {
        const parsed = parseInt(resolutionElement.value);
        if (isNaN(parsed) || parsed < 5 || parsed > 250) {
            resolutionElement.style.border = "2px solid #BF616A";
            return;
        }
        resolution = parsed;

        resolutionElement.innerHTML = parsed;
        resolutionElement.style.border = "none";
    });
}


function resize() {
    if (canvas) {
        let maxWidth = (window.innerWidth - 34) * (2 / 3);
        let maxHeight = window.innerHeight - 34;

        let width = Math.min(maxWidth, maxHeight);
        let height = Math.min(maxHeight, maxWidth);

        canvas.width = width;
        canvas.height = height;
    }
}

function togglePause() {
	paused = !paused;
	let text = paused ? "Resume" : "Pause";
	document.getElementById("pauseButton").innerHTML = text;
}
function clearDataPoints() {
	dataPoints.length = 0;
}
function resetNetwork() {
    const newLayers = [];
    for (let i = 0; i < globalLayers.length; i++) {
        const layer = globalLayers[i];
        newLayers.push(new Layer(layer.numInputs, layer.numOutputs, layer.activationFunction, layer.regularizationFunction, randomWeight));
    }
    globalLayers = newLayers;

    network = new NeuralNetwork(globalLayers, globalLearningRate, globalRegularizationRate, H);
}


function rgbToFillStyle(r, g, b) {
	// Range: 0 to 1.0
	let scaledR = Math.floor(r * 255);
	let scaledG = Math.floor(g * 255);
	let scaledB = Math.floor(b * 255);

	return "rgb(" + scaledR + "," + scaledG + "," + scaledB + ")";
}

function renderDataPoints() {
	for (let i = 0; i < dataPoints.length; i++) {
		const dataPoint = dataPoints[i];
		const inputs = dataPoint.inputs;
		const outputs = dataPoint.outputs;

		context.fillStyle = rgbToFillStyle(outputs[0], outputs[1], outputs[2]);

		context.beginPath();
		context.arc(inputs[0] * canvas.width, inputs[1] * canvas.height, canvas.width * RADIUS_RATIO, 0, 2 * Math.PI, false);
		
		context.fill();
		context.lineWidth = 2;
		context.strokeStyle = '#000000';
		context.stroke();
	}
}

function render() {
	context.fillStyle = "#3B4252";
	context.fillRect(0, 0, canvas.width, canvas.height);

    const w = Math.ceil(canvas.width / resolution);
    for (let x = 0; x < resolution; x++) {
        for (let y = 0; y < resolution; y++) {

            const inputs = [x / resolution, y / resolution];
            const outputs = network.forwardPass(inputs);

            context.fillStyle = rgbToFillStyle(outputs[0], outputs[1], outputs[2]);

            context.fillRect(Math.floor(x * w), Math.floor(y * w), w, w);
        }
    }

    renderDataPoints();

	if (!paused && dataPoints.length > 0) {
		network.learnIterate(dataPoints, 1);
	}


	t1 = performance.now();
	delta = t1 - t0;
	t0 = performance.now();


	document.getElementById("fpsText").innerHTML = "FPS: " + Math.round(1000 / delta);
	document.getElementById("costText").innerHTML = "Cost: " + network.costOfAll(dataPoints).toFixed(3);


	window.requestAnimationFrame(render);
}


setOnChangeForRadioButtons();
setOnChangeForResolution();

var t0 = performance.now();
var t1 = performance.now();
var delta = 1 / 60;

window.requestAnimationFrame(render);