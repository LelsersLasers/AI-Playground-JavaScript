const RADIUS_RATIO = 0.01;

let resolution = 75;
let iterations = 1;

const canvas = document.getElementsByTagName("canvas")[0];
canvas.addEventListener("click", function (event) {
    const rect = canvas.getBoundingClientRect();

    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    const inputs = [x / canvas.width, y / canvas.height];
    const outputs = selectedColor;

    dataPoints.push(new DataPoint(inputs, outputs));
});

const context = canvas.getContext("2d");

const inputOptions = [
    (inputs) => inputs[0], // X
    (inputs) => inputs[1], // Y
    (inputs) => inputs[0] * inputs[0], // X^2
    (inputs) => inputs[1] * inputs[1], // Y^2
    (inputs) => inputs[0] * inputs[1], // X*Y
    (inputs) => Math.sin(inputs[0]), // SIN(X)
    (inputs) => Math.sin(inputs[1]), /// SIN(Y)
];
const toggledInputs = [0, 1]; // indexes to inputOptions

let epoches = 0;

let globalActivation = leakyRelu;
let globalRegularization = noRegulation;
let globalInitialWeight = randomWeight;
let globalMomentum = 0.9;
let globalLayers = [
    new Layer(
        2,
        6,
        globalActivation,
        globalRegularization,
        globalInitialWeight,
        globalMomentum
    ),
    new Layer(
        6,
        6,
        globalActivation,
        globalRegularization,
        globalInitialWeight,
        globalMomentum
    ),
    new Layer(
        6,
        3,
        globalActivation,
        globalRegularization,
        globalInitialWeight,
        globalMomentum
    ),
];
let globalLearningRate = 0.1;
let globalRegularizationRate = 0.001;
const H = 0.00001;
let network = new NeuralNetwork(
    globalLayers,
    globalLearningRate,
    globalRegularizationRate,
    H
);

let selectedColor = [1.0, 0.0, 0.0];

const dataPoints = [];
{
    function equallySpacedPointsOfCircle(radius, numPoints) {
        const points = [];
        for (let i = 0; i < numPoints; i++) {
            const angle = (i * 2 * Math.PI) / numPoints;
            points.push([radius * Math.cos(angle), radius * Math.sin(angle)]);
        }
        return points;
    }

    const numPoints = 20;

    const redPoints = equallySpacedPointsOfCircle(0.4, numPoints);
    const bluePoints = equallySpacedPointsOfCircle(0.25, numPoints);

    for (let i = 0; i < numPoints; i++) {
        const redPoint = [0.5 + redPoints[i][0], 0.5 + redPoints[i][1]];
        dataPoints.push(new DataPoint(redPoint, [1.0, 0.0, 0.0]));

        const bluePoint = [0.5 + bluePoints[i][0], 0.5 + bluePoints[i][1]];
        dataPoints.push(new DataPoint(bluePoint, [0.0, 0.0, 1.0]));
    }
}

let paused = false;

function apply() {
    // NOTE: order matters

    updateActivation(); // sets globalActivation
    updateRegularization(); // sets globalRegularization
    updateInit(); // sets globalInitialWeight

    updateLayers(); // sets globalLayers (uses globalActivation, globalRegularization, globalInitialWeight, globalMomentum)

    updateRegularizationRate(); // sets globalRegularizationRate

    network = new NeuralNetwork(
        globalLayers,
        globalLearningRate,
        globalRegularizationRate,
        H
    );
    console.log(network);

    epoches = 0;
}

function updateLayers() {
    const hiddenLayers = document.getElementById("HIDDEN_LAYERS");

    const valTrimmed = hiddenLayers.value.replace(/\s/g, "");
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
    let lastSize = toggledInputs.length;
    for (let i = 0; i < valSplit.length; i++) {
        layers.push(
            new Layer(
                lastSize,
                valSplit[i],
                globalActivation,
                globalRegularization,
                globalInitialWeight,
                globalMomentum
            )
        );
        lastSize = valSplit[i];
    }
    layers.push(
        new Layer(
            lastSize,
            3,
            globalActivation,
            globalRegularization,
            globalInitialWeight,
            globalMomentum
        )
    );

    globalLayers = layers;

    hiddenLayers.innerHTML = valSplit.join(", ");
    hiddenLayers.style.border = "none";
}
function updateActivation() {
    const activationSelect = document.getElementById("ACTIVATION");
    switch (activationSelect.value) {
        case "relu":
            globalActivation = relu;
            break;
        case "leakyRelu":
            globalActivation = leakyRelu;
            break;
        case "sigmoid":
            globalActivation = sigmoid;
            break;
        case "tanh":
            globalActivation = tanh;
            break;
        case "linear":
            globalActivation = linear;
            break;
    }
}
function setOnChangeForLearningRate() {
    const learningRateElement = document.getElementById("LEARNING_RATE");
    learningRateElement.addEventListener("input", function () {
        globalLearningRate = parseFloat(learningRateElement.value);
        network.learningRate = globalLearningRate;
    });
}
function setOnChangeForMomentum() {
    const momentumElement = document.getElementById("MOMENTUM");
    momentumElement.addEventListener("input", function () {
        globalMomentum = parseFloat(momentumElement.value);
        for (let i = 0; i < network.layers.length; i++) {
            network.layers[i].momentum = globalMomentum;
        }
    });
}

function updateRegularization() {
    const regularizationSelect = document.getElementById("REGULARIZATION");
    switch (regularizationSelect.value) {
        case "noRegulation":
            globalRegularization = noRegulation;
            break;
        case "L1":
            globalRegularization = L1;
            break;
        case "L2":
            globalRegularization = L2;
            break;
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
function updateInit() {
    const initSelect = document.getElementById("INIT");
    switch (initSelect.value) {
        case "random":
            globalInitialWeight = randomWeight;
            break;
        case "xavier":
            globalInitialWeight = xavierWeight;
            break;
        case "he":
            globalInitialWeight = heWeight;
            break;
    }
}

function setOnChangeForRadioButtons() {
    function hexToRgb(hex) {
        const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
        return result ? {
            r: parseInt(result[1], 16),
            g: parseInt(result[2], 16),
            b: parseInt(result[3], 16)
        } : null;
    }


    const radioButtons = document.querySelectorAll(
        "input[type=radio][name='COLOR']"
    );
    radioButtons.forEach((radioButton) => {
        radioButton.addEventListener("change", function () {
            switch (this.id) {
                case "red":
                    selectedColor = [1.0, 0.0, 0.0];
                    break;
                case "green":
                    selectedColor = [0.0, 1.0, 0.0];
                    break;
                case "blue":
                    selectedColor = [0.0, 0.0, 1.0];
                    break;
                case "white":
                    selectedColor = [1.0, 1.0, 1.0];
                    break;
                case "black":
                    selectedColor = [0.0, 0.0, 0.0];
                    break;
                case "custom": {
                    const rgb = hexToRgb(document.getElementById("CUSTOM_COLOR").value);
                    selectedColor = [rgb.r / 255, rgb.g / 255, rgb.b / 255];
                    break;
                }
            }
        });
    });

    const customColor = document.getElementById("CUSTOM_COLOR");
    customColor.addEventListener("input", function () {
        if (document.getElementById("custom").checked) {
            const rgb = hexToRgb(document.getElementById("CUSTOM_COLOR").value);
            selectedColor = [rgb.r / 255, rgb.g / 255, rgb.b / 255];
        }
    });
}

function setOnChangeForResolution() {
    const resolutionElement = document.getElementById("RESOLUTION");
    resolutionElement.addEventListener("input", function () {
        resolution = parseInt(resolutionElement.value);
    });
}
function setOnChangeForIterations() {
    const iterationsElement = document.getElementById("ITERATIONS");
    iterationsElement.addEventListener("input", function () {
        iterations = parseInt(iterationsElement.value);
    });
}

function setOnChangeForInputs() {
    const inputsElement = document.querySelectorAll("input[type=checkbox]");
    inputsElement.forEach((inputElement) => {
        inputElement.addEventListener("change", function () {
            const id = this.id.replace("INPUT-", "");
            if (this.checked) {
                switch (id) {
                    case "X":
                        toggledInputs.push(0);
                        break;
                    case "Y":
                        toggledInputs.push(1);
                        break;
                    case "X^2":
                        toggledInputs.push(2);
                        break;
                    case "Y^2":
                        toggledInputs.push(3);
                        break;
                    case "X*Y":
                        toggledInputs.push(4);
                        break;
                    case "SIN(X)":
                        toggledInputs.push(5);
                        break;
                    case "SIN(Y)":
                        toggledInputs.push(6);
                        break;
                }
            } else {
                switch (id) {
                    case "X":
                        toggledInputs.filter((x) => x != 0);
                        break;
                    case "Y":
                        toggledInputs.filter((x) => x != 1);
                        break;
                    case "X^2":
                        toggledInputs.filter((x) => x != 2);
                        break;
                    case "Y^2":
                        toggledInputs.filter((x) => x != 3);
                        break;
                    case "X*Y":
                        toggledInputs.filter((x) => x != 4);
                        break;
                    case "SIN(X)":
                        toggledInputs.filter((x) => x != 5);
                        break;
                    case "SIN(Y)":
                        toggledInputs.filter((x) => x != 6);
                        break;
                }
            }
        });
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
        newLayers.push(
            new Layer(
                layer.numInputs,
                layer.numOutputs,
                layer.activationFunction,
                layer.regularizationFunction,
                layer.weightInitFunction,
                layer.momentum
            )
        );
    }
    globalLayers = newLayers;

    network = new NeuralNetwork(
        globalLayers,
        globalLearningRate,
        globalRegularizationRate,
        H
    );

    epoches = 0;
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
        context.arc(
            inputs[0] * canvas.width,
            inputs[1] * canvas.height,
            canvas.width * RADIUS_RATIO,
            0,
            2 * Math.PI,
            false
        );

        context.fill();
        context.lineWidth = 2;
        context.strokeStyle = "#000000";
        context.stroke();
    }
}

let gradientMagnitude = 0;
function render() {
    context.fillStyle = "#3B4252";
    context.fillRect(0, 0, canvas.width, canvas.height);

    const w = canvas.width / resolution;
    const ceilW = Math.ceil(w);

    const inputFns = [];
    for (let i = 0; i < toggledInputs.length; i++) {
        const fn = inputOptions[toggledInputs[i]];
        inputFns.push(fn);
    }

    for (let x = 0; x < resolution + 1; x++) {
        for (let y = 0; y < resolution + 1; y++) {
            const baseInputs = [x / resolution, y / resolution];
            const inputs = [];
            for (let i = 0; i < inputFns.length; i++) {
                const fn = inputFns[i];
                inputs.push(fn(baseInputs));
            }

            const outputs = network.forwardPass(inputs);

            context.fillStyle = rgbToFillStyle(
                outputs[0],
                outputs[1],
                outputs[2]
            );

            context.fillRect(
                Math.floor((x - 0.5) * w),
                Math.floor((y - 0.5) * w),
                ceilW,
                ceilW
            );
        }
    }

    renderDataPoints();

    const transformedDataPoints = [];
    for (let i = 0; i < dataPoints.length; i++) {
        const dataPoint = dataPoints[i];

        const baseInputs = dataPoint.inputs;
        const inputs = [];
        for (let i = 0; i < inputFns.length; i++) {
            const fn = inputFns[i];
            inputs.push(fn(baseInputs));
        }
        transformedDataPoints.push(new DataPoint(inputs, dataPoint.outputs));
    }

    if (!paused && dataPoints.length > 0) {
        gradientMagnitude = network.learnIterate(
            transformedDataPoints,
            iterations
        );
        epoches += iterations;
    }

    t1 = performance.now();
    delta = t1 - t0;
    t0 = performance.now();

    document.getElementById("fpsText").innerHTML =
        "FPS: " + Math.round(1000 / delta);
    document.getElementById("costText").innerHTML =
        "Cost: " + network.costOfAll(transformedDataPoints).toFixed(3);
    document.getElementById("epochesText").innerHTML = "Epoches: " + epoches;
    document.getElementById("gradientText").innerHTML =
        "Gradient Magnitude: " + gradientMagnitude.toFixed(3);

    window.requestAnimationFrame(render);
}

setOnChangeForRadioButtons();
setOnChangeForInputs();

setOnChangeForResolution();
setOnChangeForIterations();

setOnChangeForLearningRate();
setOnChangeForMomentum();

var t0 = performance.now();
var t1 = performance.now();
var delta = 1 / 60;

window.requestAnimationFrame(render);
