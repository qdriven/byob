// Function to  be plotted
function himmelblau(x, y) {
    return Math.pow(Math.pow(x, 2) + y - 11, 2) + Math.pow(x + Math.pow(y, 2) - 7, 2)
}
himmelblau.name = 'himmelblau';

function rastrigin(x, y) {
    return 20 + Math.pow(x, 2) + Math.pow(y, 2) - 10 * (Math.cos(2 * Math.PI * x) + Math.cos(2 * Math.PI * y));
}
rastrigin.name = 'rastrigin';

function rosenbrock(x, y) {
    return Math.pow(1 - x, 2) + 100 * Math.pow(y - Math.pow(x, 2), 2);
}
rosenbrock.name = 'rosenbrock';

function booth(x, y) {
    return Math.pow(x + 2 * y - 7, 2) + Math.pow(2 * x + y - 5, 2);
}
booth.name = 'booth';

function beale(x, y) {
    return Math.pow(1.5 - x + x * y, 2) + Math.pow(2.25 - x + x * Math.pow(y, 2), 2) + Math.pow(2.625 - x + x * Math.pow(y, 3), 2);
}
beale.name = 'beale';

function threehumpcamel(x, y) {
    return 2 * Math.pow(x, 2) - 1.05 * Math.pow(x, 4) + (Math.pow(x, 6) / 6) + x * y + Math.pow(y, 2);
}
threehumpcamel.name = 'threehumpcamel';

function matyas(x, y) {
    return 0.26 * (Math.pow(x, 2) + Math.pow(y, 2)) - 0.48 * x * y;
}
matyas.name = 'matyas';


// Current active function
let currentFunction = himmelblau;

// Function metadata
const functionInfo = {
    himmelblau: {
        name: "Himmelblau Function",
        description: "A multi-modal function with 4 identical local minima",
        minima: [
            { x: 3, y: 2 },
            { x: -2.805118, y: 3.131312 },
            { x: -3.779310, y: -3.283186 },
            { x: 3.584428, y: -1.848126 }
        ]
    },
    rastrigin: {
        name: "Rastrigin Function",
        description: "A highly multimodal function with many local minima",
        minima: [{ x: 0, y: 0 }]
    },
    rosenbrock: {
        name: "Rosenbrock Function (Banana Function)",
        description: "A unimodal function with a narrow valley",
        minima: [{ x: 1, y: 1 }]
    },
    booth: {
        name: "Booth Function",
        description: "A simple quadratic function with one global minimum",
        minima: [{ x: 1, y: 3 }]
    },
    beale: {
        name: "Beale Function",
        description: "A multimodal function with sharp peaks",
        minima: [{ x: 3, y: 0.5 }]
    },
    threehumpcamel: {
        name: "Three-Hump Camel Function",
        description: "A complex surface with multiple valleys and peaks",
        minima: [{ x: 0, y: 0 }]
    },
    matyas: {
        name: "Matyas Function",
        description: "A relatively simple form and a smooth surface",
        minima: [{ x: 0, y: 0 }]
    }
};

// Data for plotting
var precision = 0.1;
var x = d3.range(-6, 6, precision);
var y = d3.range(-6, 6, precision);

var n = x.length;
var m = y.length;

var h = 1e-7;

function updateContourData() {
    var z = [];
    for (var i = m - 1; i >= 0; i--) {
        for (var j = 0; j < n; j++) {
            var k = j + (m - 1 - i) * n;
            z[k] = currentFunction(x[j], y[i]);
        }
    }
    return z;
}

var z = updateContourData();

// Function to change the current optimization function
function setFunction(funcName) {
    // Update current function
    currentFunction = window[funcName];
    currentFunction.name = funcName;  // Explicitly set the function name

    // Clear previous optimization paths and points
    if (window.graphArea) {
        window.graphArea.selectAll(".optimization-path").remove();
        window.graphArea.selectAll("#current-point").remove();
    }

    // Update contour plot data
    z = updateContourData();

    // Update minima points with current function's minima
    if (window.updateCurrentMinima) {
        window.updateCurrentMinima(funcName);
    }

    // Update contour plot
    if (window.contourPlot) {
        window.contourPlot
            .data(window.contours(z))
            .attr("d", d3.geoPath(d3.geoIdentity().scale(window.width / m)))
            .attr("fill", function (d) { return window.color(d.value); });
    }

    return functionInfo[funcName];
}

// Returns gradient of the selected function at (x, y)
function grad_f(x, y) {
    switch (currentFunction.name) {
        case 'himmelblau':
            return [
                4 * x * (x * x + y - 11) + 2 * (x + y * y - 7),
                2 * (x * x + y - 11) + 4 * y * (x + y * y - 7)
            ];
        case 'rastrigin':
            return [
                2 * x + 20 * Math.PI * Math.sin(2 * Math.PI * x),
                2 * y + 20 * Math.PI * Math.sin(2 * Math.PI * y)
            ];
        case 'rosenbrock':
            return [
                2 * (x - 1) + 200 * x * (y - x * x),
                100 * (y - x * x)
            ];
        case 'booth':
            return [
                2 * (x + 2 * y - 7) + 4 * (2 * x + y - 5),
                4 * (x + 2 * y - 7) + 2 * (2 * x + y - 5)
            ];
        case 'beale':
            return [
                2 * (1.5 - x + x * y) * (-1 + y) +
                2 * (2.25 - x + x * y * y) * (-1 + y * y) +
                2 * (2.625 - x + x * y * y * y) * (-1 + y * y * y),
                2 * (1.5 - x + x * y) * x +
                2 * (2.25 - x + x * y * y) * 2 * x * y +
                2 * (2.625 - x + x * y * y * y) * 3 * x * y * y
            ];
        case 'threehumpcamel':
            return [
                4 * x - 4.2 * x * x * x + x * x * x * x * x + y,
                x + 2 * y
            ];
        case 'matyas':
            return [
                0.52 * x - 0.48 * y,
                0.52 * y - 0.48 * x
            ];

        default:
            console.error('Unknown function:', currentFunction.name);
            return [0, 0];
    }
}

// Algorithm descriptions
const algorithmInfo = {
    sgd: {
        name: "Stochastic Gradient Descent",
        description: "Basic gradient descent that updates parameters in the direction of steepest descent."
    },
    momentum: {
        name: "SGD with Momentum",
        description: "Adds momentum to gradient descent, helping to overcome local minima and speed up convergence."
    },
    rmsprop: {
        name: "RMSProp",
        description: "Adapts learning rates based on the moving average of squared gradients, helping with non-stationary objectives."
    },
    adam: {
        name: "Adam",
        description: "Combines momentum and RMSProp, using both first and second moments of gradients for adaptive learning rates."
    },
    nag: {
        name: "Nesterov Accelerated Gradient",
        description: "An extension of gradient descent with momentum and achieve faster convergence in some situations by looking ahead in the direction of the momentum."
    }
};

// Get optimization path based on selected algorithm
function getOptimizationPath(x0, y0, learning_rate, num_steps) {
    const algorithm = document.getElementById('algorithm-select').value;

    switch (algorithm) {
        case 'sgd':
            return get_sgd_data(x0, y0, learning_rate, num_steps);
        case 'momentum':
            return get_momentum_data(x0, y0, learning_rate, num_steps);
        case 'rmsprop':
            return get_rmsprop_data(x0, y0, learning_rate, num_steps);
        case 'adam':
            return get_adam_data(x0, y0, learning_rate, num_steps);
        case 'nag':
            return get_nag_data(x0, learning_rate, num_steps);
        default:
            console.error('Unknown algorithm:', algorithm);
            return [];
    }
}

// Export necessary variables and functions
window.currentFunction = currentFunction;
window.functionInfo = functionInfo;
window.setFunction = setFunction;
window.z = z;
window.updateContourData = updateContourData;
window.grad_f = grad_f;
window.getOptimizationPath = getOptimizationPath;
window.algorithmInfo = algorithmInfo;

