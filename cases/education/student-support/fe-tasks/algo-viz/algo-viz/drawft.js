//helper Function
// Returns gradient of f at (x, y) for different functions
function grad_f(x, y) {
    const functionName = window.currentFunction.name;
    
    switch(functionName) {
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
                2 * (x - 1) + 400 * x * (x * x - y),
                200 * (y - x * x)
            ];
        case 'booth':
            return [
                2 * (x + 2*y - 7) + 2 * (2*x + y - 5),
                2 * (x + 2*y - 7) * 2 + 2 * (2*x + y - 5)
            ];
        case 'beale':
            return [
                2 * (1.5 - x + x*y) * (-1 + y) +
                2 * (2.25 - x + x*y*y) * (-1 + y*y) +
                2 * (2.625 - x + x*y*y*y) * (-1 + y*y*y),
                2 * (1.5 - x + x*y) * x +
                2 * (2.25 - x + x*y*y) * 2*x*y +
                2 * (2.625 - x + x*y*y*y) * 3*x*y*y
            ];
		case 'threehumpcamel':
			return [
				4 * x - 4.2 * Math.pow(x, 3)+Math.pow(x, 5)+y,
				x + 2 * y
			];
		case 'matyas':
			return [
				0.52 * x - 0.48 * y,
				0.52 * y - 0.48 * x
			];
			
        default:
            return [0, 0];
    }
}

//SGD algorithms
function get_sgd_data(x0, y0, learning_rate, num_steps) {
    let path = [[x0, y0]];
    let x = x0;
    let y = y0;
    let converged = false;
    let convergenceCount = 0;
    const convergenceThreshold = 1e-5;
    const requiredStableSteps = 5;

    for (let i = 0; i < num_steps && !converged; i++) {
        let [dx, dy] = grad_f(x, y);
        let new_x = x - learning_rate * dx;
        let new_y = y - learning_rate * dy;
        
        // Bound checking
        new_x = Math.max(-6, Math.min(6, new_x));
        new_y = Math.max(-6, Math.min(6, new_y));
        
        // Check for convergence
        const change = Math.sqrt(Math.pow(new_x - x, 2) + Math.pow(new_y - y, 2));
        if (change < convergenceThreshold) {
            convergenceCount++;
            if (convergenceCount >= requiredStableSteps) {
                converged = true;
            }
        } else {
            convergenceCount = 0;
        }
        
        x = new_x;
        y = new_y;
        path.push([x, y]);
    }
    
    return path;
}

function get_momentum_data(x0, y0, learning_rate, num_steps) {
    const dampening = 0.5;
    const momentum = 0.9;
    let path = [[x0, y0]];
    let vx = 0, vy = 0;
    let x = x0, y = y0;

    for (let i = 0; i < num_steps; i++) {
        const [gx, gy] = grad_f(x, y);
        
        // Update velocity with momentum and dampening
        vx = momentum * vx - (1 - dampening) * learning_rate * gx;
        vy = momentum * vy - (1 - dampening) * learning_rate * gy;
        
        // Update position
        x = Math.max(-6, Math.min(6, x + vx));
        y = Math.max(-6, Math.min(6, y + vy));
        
        path.push([x, y]);
    }
    return path;
}

function get_rmsprop_data(x0, y0, learning_rate, num_steps) {
    const decay_rate = 0.9;
    const epsilon = 1e-8;
    let path = [[x0, y0]];
    let x = x0, y = y0;
    let squared_grad_x = 0;
    let squared_grad_y = 0;

    for (let i = 0; i < num_steps; i++) {
        const [gx, gy] = grad_f(x, y);
        
        // Update squared gradients
        squared_grad_x = decay_rate * squared_grad_x + (1 - decay_rate) * gx * gx;
        squared_grad_y = decay_rate * squared_grad_y + (1 - decay_rate) * gy * gy;
        
        // Compute adaptive learning rates
        const dx = learning_rate * gx / (Math.sqrt(squared_grad_x) + epsilon);
        const dy = learning_rate * gy / (Math.sqrt(squared_grad_y) + epsilon);
        
        // Update position
        x = Math.max(-6, Math.min(6, x - dx));
        y = Math.max(-6, Math.min(6, y - dy));
        
        path.push([x, y]);
    }
    return path;
}

function get_adam_data(x0, y0, learning_rate, num_steps) {
    const beta1 = 0.9;
    const beta2 = 0.999;
    const epsilon = 1e-8;
    let path = [[x0, y0]];
    let x = x0, y = y0;
    let m_x = 0, m_y = 0;  // First moment estimates
    let v_x = 0, v_y = 0;  // Second moment estimates
    
    for (let i = 1; i <= num_steps; i++) {
        const [gx, gy] = grad_f(x, y);
        
        // Update biased first moment estimates
        m_x = beta1 * m_x + (1 - beta1) * gx;
        m_y = beta1 * m_y + (1 - beta1) * gy;
        
        // Update biased second moment estimates
        v_x = beta2 * v_x + (1 - beta2) * gx * gx;
        v_y = beta2 * v_y + (1 - beta2) * gy * gy;
        
        // Compute bias-corrected first and second moment estimates
        const m_x_hat = m_x / (1 - Math.pow(beta1, i));
        const m_y_hat = m_y / (1 - Math.pow(beta1, i));
        const v_x_hat = v_x / (1 - Math.pow(beta2, i));
        const v_y_hat = v_y / (1 - Math.pow(beta2, i));
        
        // Update position
        x = Math.max(-6, Math.min(6, x - learning_rate * m_x_hat / (Math.sqrt(v_x_hat) + epsilon)));
        y = Math.max(-6, Math.min(6, y - learning_rate * m_y_hat / (Math.sqrt(v_y_hat) + epsilon)));
        
        path.push([x, y]);
    }
    return path;
}

function get_nag_data(x0, y0, learning_rate, num_steps) {
    const momentum = 0.9;
    let path = [[x0, y0]];
    let x = x0, y = y0;
    let vx = 0, vy = 0;

    for (let i = 0; i < num_steps; i++) {
        // Look ahead to approximate future position
        const look_ahead_x = x + momentum * vx;
        const look_ahead_y = y + momentum * vy;
        
        // Get gradient at the look-ahead position
        const [gx, gy] = grad_f(look_ahead_x, look_ahead_y);
        
        // Update velocity using NAG formula
        vx = momentum * vx - learning_rate * gx;
        vy = momentum * vy - learning_rate * gy;
        
        // Update position
        x = Math.max(-6, Math.min(6, x + vx));
        y = Math.max(-6, Math.min(6, y + vy));
        
        path.push([x, y]);
    }
    return path;
}

// Get selected algorithms from checkboxes
function getSelectedAlgorithms() {
    const checkboxes = document.querySelectorAll('.algorithm-option input[type="checkbox"]');
    const selected = [];
    checkboxes.forEach(checkbox => {
        if (checkbox.checked) {
            selected.push(checkbox.value);
        }
    });
    return selected;
}

// Get optimization path based on selected algorithms
function getOptimizationPath(x0, y0, learning_rate, num_steps) {
    const selectedAlgorithms = getSelectedAlgorithms();
    if (selectedAlgorithms.length === 0) {
        console.error('No algorithms selected');
        return [];
    }
    
    // If only one algorithm is selected, return its path directly
    if (selectedAlgorithms.length === 1) {
        const algorithm = selectedAlgorithms[0];
        switch(algorithm) {
            case 'sgd':
                return get_sgd_data(x0, y0, learning_rate, num_steps);
            case 'momentum':
                return get_momentum_data(x0, y0, learning_rate, num_steps);
            case 'rmsprop':
                return get_rmsprop_data(x0, y0, learning_rate, num_steps);
            case 'adam':
                return get_adam_data(x0, y0, learning_rate, num_steps);
            case 'nag':
                return get_nag_data(x0, y0, learning_rate, num_steps);
            default:
                console.error('Unknown algorithm:', algorithm);
                return [];
        }
    }
    
    // For multiple algorithms, generate paths for all selected ones
    const paths = {};
    selectedAlgorithms.forEach(algorithm => {
        switch(algorithm) {
            case 'sgd':
                paths[algorithm] = get_sgd_data(x0, y0, learning_rate, num_steps);
                break;
            case 'momentum':
                paths[algorithm] = get_momentum_data(x0, y0, learning_rate, num_steps);
                break;
            case 'rmsprop':
                paths[algorithm] = get_rmsprop_data(x0, y0, learning_rate, num_steps);
                break;
            case 'adam':
                paths[algorithm] = get_adam_data(x0, y0, learning_rate, num_steps);
                break;
            case 'nag':
                paths[algorithm] = get_nag_data(x0, y0, learning_rate, num_steps);
                break;
        }
    });
    
    // Return the longest path for animation tracking
    let maxLength = 0;
    let longestPath = [];
    for (const algo in paths) {
        if (paths[algo].length > maxLength) {
            maxLength = paths[algo].length;
            longestPath = paths[algo];
        }
    }
    
    return longestPath;
}

// Animation control
let animationId = null;
let currentStep = 0;
let optimizationPath = [];
let animationSpeed = 5;
let isAnimating = false;
let activeTransitions = 0;  // Track active transitions
let animationTimeoutId = null; // Track animation timeouts

// Function to clean up all animations and elements
function cleanupAnimation() {
    // Cancel any ongoing animation frame
    if (animationId) {
        cancelAnimationFrame(animationId);
        animationId = null;
    }
    
    // Clear animation timeout
    if (animationTimeoutId) {
        clearTimeout(animationTimeoutId);
        animationTimeoutId = null;
    }
    
    // Clear all transitions and timeouts related to animation
    if (typeof d3 !== 'undefined') {
        d3.timerFlush(); // Force completion of active transitions
    }
    
    // Clear all existing paths and points with immediate effect
    window.graphArea.selectAll(".optimization-path-group").remove();
    window.graphArea.selectAll("#current-point").remove();
    window.graphArea.selectAll('[id^="current-point-"]').remove(); // Remove all algorithm-specific points
    window.graphArea.selectAll(".final-point").remove();
    window.graphArea.selectAll(".start-point-marker").remove();
    
    // Reset animation state
    currentStep = 0;
    optimizationPath = [];
    isAnimating = false;
    activeTransitions = 0;
}

// Function to animate one step of the optimization
function animateStep(path, index) {
    if (!path || !Array.isArray(path) || index >= path.length) {
        return;
    }

    const selectedAlgorithms = getSelectedAlgorithms();
    if (selectedAlgorithms.length === 0) {
        cleanupAnimation();
        return;
    }

    // If only one algorithm is selected, use the original animation
    if (selectedAlgorithms.length === 1) {
        const algorithm = selectedAlgorithms[0];
        const pathColor = getAlgorithmColor(algorithm);
        const prevPoint = index > 0 ? path[index - 1] : null;
        const currentPoint = path[index];
        
        if (prevPoint) {
            drawPathSegment(prevPoint, currentPoint);
        }
        
        // Update the current point
        let currentPointElem = window.graphArea.select('#current-point');
        
        if (currentPointElem.empty()) {
            currentPointElem = window.graphArea.append("circle")
                .attr("id", "current-point")
                .attr("r", 5)
                .attr("fill", pathColor)
                .style("stroke", "#fff")
                .style("stroke-width", 1);
        }
        
        // Update position with transition
        activeTransitions++;
        currentPointElem
            .transition()
            .duration(50)
            .ease(d3.easeLinear)
            .attr("cx", window.currScale.x(currentPoint[0]))
            .attr("cy", window.currScale.y(currentPoint[1]))
            .on("end", function() {
                activeTransitions--;
            });
        
        // Update coordinates display
        updateCoordinates(currentPoint[0], currentPoint[1]);
    } else {
        // For multiple algorithms, draw each path with its own color
        selectedAlgorithms.forEach(algorithm => {
            let algorithmPath;
            
            // Generate the path for each algorithm
            switch(algorithm) {
                case 'sgd':
                    algorithmPath = get_sgd_data(window.clickX, window.clickY, 
                        parseFloat(document.getElementById('learning-rate').value),
                        parseInt(document.getElementById('num-steps').value));
                    break;
                case 'momentum':
                    algorithmPath = get_momentum_data(window.clickX, window.clickY, 
                        parseFloat(document.getElementById('learning-rate').value),
                        parseInt(document.getElementById('num-steps').value));
                    break;
                case 'rmsprop':
                    algorithmPath = get_rmsprop_data(window.clickX, window.clickY, 
                        parseFloat(document.getElementById('learning-rate').value),
                        parseInt(document.getElementById('num-steps').value));
                    break;
                case 'adam':
                    algorithmPath = get_adam_data(window.clickX, window.clickY, 
                        parseFloat(document.getElementById('learning-rate').value),
                        parseInt(document.getElementById('num-steps').value));
                    break;
                case 'nag':
                    algorithmPath = get_nag_data(window.clickX, window.clickY, 
                        parseFloat(document.getElementById('learning-rate').value),
                        parseInt(document.getElementById('num-steps').value));
                    break;
            }
            
            // If the current index is within this algorithm's path, draw it
            if (algorithmPath && index < algorithmPath.length) {
                const prevPoint = index > 0 ? algorithmPath[index - 1] : null;
                const currentPoint = algorithmPath[index];
                
                // Validate points to prevent NaN errors
                if (prevPoint && currentPoint && 
                    isValidPoint(prevPoint) && isValidPoint(currentPoint)) {
                    drawPathSegmentForAlgorithm(prevPoint, currentPoint, algorithm);
                    
                    // Update the current point for this algorithm
                    let currentPointElem = window.graphArea.select(`#current-point-${algorithm}`);
                    
                    if (currentPointElem.empty()) {
                        currentPointElem = window.graphArea.append("circle")
                            .attr("id", `current-point-${algorithm}`)
                            .attr("r", 5)
                            .attr("fill", getAlgorithmColor(algorithm));
                    }
                    
                    currentPointElem
                        .attr("cx", window.currScale.x(currentPoint[0]))
                        .attr("cy", window.currScale.y(currentPoint[1]));
                    
                    // Handle final point if we're at the last step for this algorithm
                    if (index === algorithmPath.length - 1) {
                        handleFinalPointForAlgorithm(currentPoint, window.currentMinima, algorithm);
                    }
                }
            }
        });
        
        // Update coordinates with the reference path's current point
        if (index < path.length && isValidPoint(path[index])) {
            updateCoordinates(path[index][0], path[index][1]);
        }
    }
    
    // Continue animation with speed control only if we're still animating
    if (index < path.length - 1 && isAnimating) {
        const delay = Math.max(20, Math.min(200, 200 / animationSpeed));
        animationTimeoutId = setTimeout(() => animateStep(path, index + 1), delay);
    }
}

// Helper function to validate a point
function isValidPoint(point) {
    return Array.isArray(point) && 
           point.length >= 2 && 
           typeof point[0] === 'number' && 
           typeof point[1] === 'number' && 
           !isNaN(point[0]) && 
           !isNaN(point[1]);
}

// Helper function to get color for each algorithm
function getAlgorithmColor(algorithm) {
    const colors = {
        'sgd': '#ff5722',        // Orange
        'momentum': '#2196f3',   // Blue
        'rmsprop': '#4caf50',    // Green
        'adam': '#9c27b0',       // Purple
        'nag': '#ffeb3b'         // Yellow
    };
    return colors[algorithm] || '#ffffff';
}

// Helper function to draw a path segment for single algorithm
function drawPathSegment(prevPoint, currentPoint) {
    if (!prevPoint) return;
    
    // Get the currently selected algorithm for color
    const selectedAlgorithms = getSelectedAlgorithms();
    if (selectedAlgorithms.length !== 1) return;
    
    const algorithm = selectedAlgorithms[0];
    const pathColor = getAlgorithmColor(algorithm);
    
    // Create a path group for the single algorithm
    let pathGroup = window.graphArea.select('.optimization-path-group');
    
    if (pathGroup.empty()) {
        pathGroup = window.graphArea.append("g")
            .attr("class", "optimization-path-group")
            .attr("transform", d3.zoomTransform(window.graphArea.node()));
    }
    
    // Draw the path segment
    pathGroup.append("line")
        .attr("x1", window.currScale.x(prevPoint[0]))
        .attr("y1", window.currScale.y(prevPoint[1]))
        .attr("x2", window.currScale.x(currentPoint[0]))
        .attr("y2", window.currScale.y(currentPoint[1]))
        .attr("stroke", pathColor)
        .attr("stroke-width", 2)
        .attr("opacity", 0)
        .transition()
        .duration(1000 / animationSpeed)
        .attr("opacity", 1);
}

// Function to handle final point after animation completes
function handleFinalPoint(lastPoint, minima) {
    // Get the currently selected algorithm for color
    const selectedAlgorithms = getSelectedAlgorithms();
    if (selectedAlgorithms.length !== 1) return;
    
    const algorithm = selectedAlgorithms[0];
    const pathColor = getAlgorithmColor(algorithm);
    
    // Add final marker
    window.graphArea.append("circle")
        .attr("class", "final-point")
        .attr("cx", window.currScale.x(lastPoint[0]))
        .attr("cy", window.currScale.y(lastPoint[1]))
        .attr("r", 5)
        .attr("fill", "none")
        .attr("stroke", pathColor)
        .attr("stroke-width", 2)
        .attr("opacity", 0)
        .transition()
        .duration(1000 / animationSpeed)
        .attr("opacity", 1)
        .attr("r", 8);
    
    // Update coordinate display with final point information
    updateCoordinates(lastPoint[0], lastPoint[1]);
}

// Helper function to draw path segment with color based on algorithm
function drawPathSegmentForAlgorithm(prevPoint, currentPoint, algorithm) {
    if (!prevPoint || !isValidPoint(prevPoint) || !isValidPoint(currentPoint)) return;
    
    const pathColor = getAlgorithmColor(algorithm);
    const pathGroupClass = `optimization-path-group-${algorithm}`;
    
    // Check if we have a path group for this algorithm
    let pathGroup = window.graphArea.select(`.${pathGroupClass}`);
    
    if (pathGroup.empty()) {
        // Create a new path group for this algorithm
        pathGroup = window.graphArea.append("g")
            .attr("class", `optimization-path-group ${pathGroupClass}`)
            .attr("transform", d3.zoomTransform(window.graphArea.node())); // Apply current zoom transform
        
        // Add a label for the algorithm
        pathGroup.append("text")
            .attr("class", "algorithm-label")
            .attr("x", window.currScale.x(currentPoint[0]) + 5)
            .attr("y", window.currScale.y(currentPoint[1]) - 5)
            .attr("fill", pathColor)
            .text(algorithm.toUpperCase());
    }
    
    // Draw the path segment using the current scaled coordinates
    pathGroup.append("line")
        .attr("x1", window.currScale.x(prevPoint[0]))
        .attr("y1", window.currScale.y(prevPoint[1]))
        .attr("x2", window.currScale.x(currentPoint[0]))
        .attr("y2", window.currScale.y(currentPoint[1]))
        .attr("stroke", pathColor)
        .attr("stroke-width", 2)
        .attr("opacity", 0)
        .transition()
        .duration(1000 / animationSpeed)
        .attr("opacity", 1);
    
    // Update the algorithm label position
    pathGroup.select(".algorithm-label")
        .attr("x", window.currScale.x(currentPoint[0]) + 5)
        .attr("y", window.currScale.y(currentPoint[1]) - 5);
}

// Function to handle final point for specific algorithm
function handleFinalPointForAlgorithm(lastPoint, minima, algorithm) {
    if (!isValidPoint(lastPoint)) return;
    
    const pathColor = getAlgorithmColor(algorithm);
    
    // Add final marker
    window.graphArea.append("circle")
        .attr("class", "final-point")
        .attr("cx", window.currScale.x(lastPoint[0]))
        .attr("cy", window.currScale.y(lastPoint[1]))
        .attr("r", 5)
        .attr("fill", "none")
        .attr("stroke", pathColor)
        .attr("stroke-width", 2)
        .attr("opacity", 0)
        .transition()
        .duration(1000 / animationSpeed)
        .attr("opacity", 1)
        .attr("r", 8);
    
    // Update coordinate display with final point information
    updateCoordinates(lastPoint[0], lastPoint[1]);
}

// Function to handle mouse click and start optimization
function animatealgorithms() {
    // Clean up any existing animation first
    cleanupAnimation();
    
    // Get click coordinates
    const coords = d3.mouse(this);
    if (!coords || coords.length < 2) return;
    
    const x0 = window.currScale.x.invert(coords[0]);
    const y0 = window.currScale.y.invert(coords[1]);
    
    if (isNaN(x0) || isNaN(y0)) return;
    
    // Store click coordinates for use in algorithm paths
    window.clickX = x0;
    window.clickY = y0;
    
    // Set animation flag to true
    isAnimating = true;
    
    // Determine color based on selected algorithm(s)
    let fillColor = "#ff4444"; // Default color
    const selectedAlgorithms = getSelectedAlgorithms();
    if (selectedAlgorithms.length === 1) {
        fillColor = getAlgorithmColor(selectedAlgorithms[0]);
    }
    
    // Add starting point with stable positioning
    const startPoint = window.graphArea.append("circle")
        .attr("id", "current-point")
        .attr("r", 4)
        .attr("fill", fillColor)
        .style("stroke", "#fff")
        .style("stroke-width", 1)
        .style("opacity", 0)
        .attr("cx", window.currScale.x(x0))
        .attr("cy", window.currScale.y(y0));
    
    // Use transition for smooth appearance
    activeTransitions++;
    startPoint
        .transition()
        .duration(200)
        .style("opacity", 1)
        .on("end", function() {
            activeTransitions--;
            // Start optimization after point is fully visible
            if (isAnimating) { // Only start if animation hasn't been cancelled
                startOptimization(x0, y0);
            }
        });
}

// Function to start the optimization animation
function startOptimization(x0, y0) {
    if (!isAnimating || typeof x0 === 'undefined' || typeof y0 === 'undefined' || 
        isNaN(x0) || isNaN(y0)) {
        cleanupAnimation();
        return;
    }
    
    // Get parameters from UI
    const learningRate = parseFloat(document.getElementById("learning-rate").value);
    const numSteps = parseInt(document.getElementById("num-steps").value);
    animationSpeed = parseInt(document.getElementById("animation-speed").value);
    
    if (isNaN(learningRate) || isNaN(numSteps) || isNaN(animationSpeed)) {
        cleanupAnimation();
        return;
    }
    
    // Generate optimization path
    optimizationPath = getOptimizationPath(x0, y0, learningRate, numSteps);
    if (!optimizationPath || !Array.isArray(optimizationPath)) {
        cleanupAnimation();
        return;
    }
    
    // Start animation only if still flagged as animating
    if (isAnimating) {
        currentStep = 0;
        animateStep(optimizationPath, currentStep);
    }
}

// Function to update coordinates display
function updateCoordinates(x, y) {
    if (typeof x === 'undefined' || typeof y === 'undefined' || isNaN(x) || isNaN(y)) {
        return;
    }
    const f = window.currentFunction ? window.currentFunction(x, y) : 0;
    
    d3.select("#x-coord").text(x.toFixed(3));
    d3.select("#y-coord").text(y.toFixed(3));
    d3.select("#f-val").text(f.toFixed(3));
}

// Initialize event listeners
document.addEventListener("DOMContentLoaded", function() {
    // Update displayed values when sliders change
    document.getElementById("learning-rate").addEventListener("input", function() {
        document.getElementById("learning-rate-value").textContent = this.value;
    });
    
    document.getElementById("num-steps").addEventListener("input", function() {
        document.getElementById("steps-value").textContent = this.value;
    });
    
    document.getElementById("animation-speed").addEventListener("input", function() {
        document.getElementById("speed-value").textContent = this.value;
        animationSpeed = parseInt(this.value);
    });
    
    // Handle function selection
    document.getElementById("function-select").addEventListener("change", function() {
        const functionName = this.value;
        
        // Stop any ongoing animation and set the flag to false
        isAnimating = false;
        
        // Clean up all animation traces thoroughly
        cleanupAnimation();
        
        // Additionally, remove all final points and any SVG elements related to animation
        window.graphArea.selectAll(".final-point").remove();
        window.graphArea.selectAll(".optimization-path-group").remove();
        window.graphArea.selectAll("#current-point").remove();
        window.graphArea.selectAll('[id^="current-point-"]').remove(); // Remove all algorithm-specific points
        window.graphArea.selectAll(".start-point-marker").remove();
        
        // Reset animation state
        currentStep = 0;
        optimizationPath = [];
        
        // Update function
        window.setFunction(functionName);
        document.getElementById("function-description").textContent = window.functionInfo[functionName].description;
    });
    
    // Handle algorithm checkbox changes for cleanup
    const algoCheckboxes = document.querySelectorAll('.algorithm-option input[type="checkbox"]');
    algoCheckboxes.forEach(checkbox => {
        checkbox.addEventListener('change', function() {
            // Stop any ongoing animation and set the flag to false
            isAnimating = false;
            
            // Clean up all animation traces when algorithms are changed
            cleanupAnimation();
            
            // Reset any related animation elements
            window.graphArea.selectAll(".final-point").remove();
            window.graphArea.selectAll(".optimization-path-group").remove();
            window.graphArea.selectAll("#current-point").remove();
            window.graphArea.selectAll('[id^="current-point-"]').remove();
            window.graphArea.selectAll(".start-point-marker").remove();
        });
    });
    
    // Set initial descriptions
    document.getElementById("function-description").textContent = window.functionInfo.himmelblau.description;
    
    // Handle graph clicks to start animation
    const graphContainer = document.getElementById("graph-container");
    if (graphContainer) {
        graphContainer.addEventListener("mousedown", function(event) {
            // Only handle left click
            if (event.button === 0) {
                event.stopPropagation();  // Prevent zoom from interfering
                animatealgorithms.call(this);
            }
        });
    }
});

// Function to update the start point marker
function updateStartPointMarker(x, y) {
    // Remove any existing start markers
    window.graphArea.selectAll(".start-point-marker").remove();
    
    // Add a new marker
    window.graphArea.append("circle")
        .attr("class", "start-point-marker")
        .attr("r", 5)
        .attr("fill", "#333")
        .style("stroke", "#fff")
        .style("stroke-width", 1)
        .attr("cx", window.currScale.x(x))
        .attr("cy", window.currScale.y(y))
        .style("opacity", 0)
        .transition()
        .duration(200)
        .style("opacity", 1);
    
    // Update coordinates display
    updateCoordinates(x, y);
}