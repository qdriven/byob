// Display properties
var crosshairWidth = 0.5;
var minRadius = 0.05, radarRadius = 0.2;
var minZoom = 1, maxZoom = 3;
var simulate = false;
var radarColor = "#ff0000";
var helpTextColor = "#fff";
var coordBoxColor = "#ffffb8";
var coordTextColor = "#000";
var contourLineColor = "#666";
var axisLabelWeight = "600";
var crosshairColor = "#fff";
var cutoffWidth = 500; 

// Make these variables globally accessible
window.xScale = null;
window.yScale = null;
window.currScale = null;
window.graphArea = null;
window.coordText = null;
window.coordBox = null;
window.updateCurrentMinima = updateCurrentMinima;  // Export the function

// Define minima for each function
window.functionMinima = {
    himmelblau: [
        { x: 3.0, y: 2.0 },
        { x: -2.805118, y: 3.131312 },
        { x: -3.779310, y: -3.283186 },
        { x: 3.584428, y: -1.848126 }
    ],
    rastrigin: [
        { x: 0.0, y: 0.0 }
    ],
    rosenbrock: [
        { x: 1.0, y: 1.0 }
    ],
    booth: [
        { x: 1.0, y: 3.0 }
    ],
    beale: [
        { x: 3.0, y: 0.5 }
    ],
	threehumpcamel: [
		{ x: 0.0, y: 0.0 }
	],
	matyas: [
		{ x: 0.0, y: 0.0 }
	]
};

// Update currentMinima reference
let currentMinima = window.functionMinima;

// Function to update current minima based on selected function
function updateCurrentMinima(functionName) {
    currentMinima = window.functionMinima[functionName];
    
    // Update stationary minimum points
    const minPoints = window.graphArea.selectAll(".stationary-min")
        .data(currentMinima);
    
    // Remove old points
    minPoints.exit().remove();
    
    // Update existing points
    minPoints
        .attr("cx", d => window.currScale.x(d.x))
        .attr("cy", d => window.currScale.y(d.y))
        .attr("r", scaledDistance("x", minRadius));
    
    // Add new points
    minPoints.enter()
        .append("circle")
        .attr("class", "stationary-min")
        .attr("fill", radarColor)
        .attr("r", scaledDistance("x", minRadius))
        .attr("cx", d => window.currScale.x(d.x))
        .attr("cy", d => window.currScale.y(d.y));

    // Update radar circle points
    const radarPoints = window.graphArea.selectAll(".min")
        .data(currentMinima);
    
    // Remove old points
    radarPoints.exit().remove();
    
    // Update existing points
    radarPoints
        .attr("cx", d => window.currScale.x(d.x))
        .attr("cy", d => window.currScale.y(d.y))
        .attr("r", scaledDistance("x", radarRadius));
    
    // Add new points
    radarPoints.enter()
        .append("circle")
        .attr("class", "min")
        .attr("fill", radarColor)
        .attr("stroke", radarColor)
        .attr("r", scaledDistance("x", radarRadius))
        .attr("cx", d => window.currScale.x(d.x))
        .attr("cy", d => window.currScale.y(d.y));
}

// Contour plot properties
var thresholds = d3.merge([d3.range(d3.min(z), 50, 10), d3.range(100, d3.max(z), 100)]);
var contours = d3.contours()
                 .size([n, m])
                 .thresholds(thresholds);
var color = d3.scaleLinear()
              .domain(d3.extent(thresholds))
              .range(['#222', 'yellow'])
              .interpolate(d3.interpolateHcl);

 // D3 elements
var xAxis, yAxis; // Axis building functions and svg groups containing the axes
var xLabel, yLabel; // Axis labels
var svg, clipArea, gX, gY,menu_g, gradient_path_g; // SVG elements containing different parts of the plot
var zoom, zoomHelpText; // Handles zooming
var minCircle, radarCircle; // Display minimum values of Himmelblau function
var margin; // Margins around graph
var labelPositions; // Position of x/y axis labels
var height, width; // Height and width of graph area

// Calculate axis distance under current scale
function scaledDistance(axis, dist){
    return currScale[[axis]](dist) - currScale[[axis]](0);
}


// Set plot margins based on screen size
function setMargins(){
    var viewPortWidth = window.innerWidth;

    if (viewPortWidth > cutoffWidth){
        margin = {
            left: 40,
            right: 20,
            top: 20,
            bottom: 40
        };
    }else{
        margin = {
            left: 30,
            right: 15,
            top: 15,
            bottom: 30
        };
    }
}

// Set axis label positions based on current margins
function setLabelPositions(){
    labelPosition = {
        x: margin.top + height + (3 * margin.bottom / 4),
        y: margin.left / 4
    };  
}

//Scale and translate graph when user zooms
function zoomed() {
    if (zoomHelpText) {
        zoomHelpText.remove();
    }

    // Get current zoom scale
    currScale = {
        x: d3.event.transform.rescaleX(xScale),
        y: d3.event.transform.rescaleY(yScale)
    };
    window.currScale = currScale; // Update global reference

    // Transform contour plot according to zoom
    contourPlot.attr("transform", d3.event.transform);

    // Update axes
    gX.call(xAxis.scale(currScale.x));
    gY.call(yAxis.scale(currScale.y));

    // Update minimum points with new scale
    window.graphArea.selectAll(".stationary-min")
        .attr("cx", d => currScale.x(d.x))
        .attr("cy", d => currScale.y(d.y))
        .attr("r", scaledDistance("x", minRadius));

    window.graphArea.selectAll(".min")
        .attr("cx", d => currScale.x(d.x))
        .attr("cy", d => currScale.y(d.y))
        .attr("r", scaledDistance("x", radarRadius));

    // Update paths and points with the new transform
    window.graphArea.selectAll(".optimization-path-group")
        .attr("transform", d3.event.transform);
        
    window.graphArea.selectAll("#current-point")
        .attr("transform", d3.event.transform);
        
    // Update algorithm-specific points and final points
    window.graphArea.selectAll("[id^='current-point-']")
        .attr("transform", d3.event.transform);
        
    window.graphArea.selectAll(".final-point")
        .attr("transform", d3.event.transform);
        
    window.graphArea.selectAll(".start-point-marker")
        .attr("transform", d3.event.transform);

    // Hide coordinate display while zooming
    showCoords(this, false);
}

// Display current coordinates and function value on mouse over
function showCoords(e, show) {
    if (!show) {
        d3.select("#coordinates").style("opacity", "0");
        return;
    }

    var coords = d3.mouse(e);
    var currX = window.currScale.x.invert(coords[0]);
    var currY = window.currScale.y.invert(coords[1]);
    var currF = currentFunction(currX, currY);

    // Update coordinate display
    d3.select("#x-coord").text(currX.toFixed(3));
    d3.select("#y-coord").text(currY.toFixed(3));
    d3.select("#f-val").text(currF.toFixed(3));
    d3.select("#coordinates").style("opacity", "1");
}

// Initialize graph on page load
document.addEventListener('DOMContentLoaded', function() {
    // Check if required functions are available
    if (typeof animatealgorithms === 'undefined') {
        console.error('Required function animatealgorithms is not loaded. Please check if drawft.js is properly loaded.');
        return;
    }
    
    try {
        initGraph();
    } catch (error) {
        console.error('Error initializing graph:', error);
    }
});

function initGraph(){
    // Set size and margin of plotting area
    var container = d3.select("#graph-container").node();
    var containerSize = Math.min(container.clientWidth, container.clientHeight);
    
    setMargins();
    width = containerSize - margin.left - margin.right;
    height = containerSize - margin.top - margin.bottom;

    // Create axes, labels and scales
    window.xScale = d3.scaleLinear() 
               .domain([d3.min(x), d3.max(x)])
               .range([0, width]);

    window.yScale = d3.scaleLinear() 
               .domain([d3.min(y), d3.max(y)])
               .range([height, 0]);

    window.currScale = {
        x: window.xScale,
        y: window.yScale
    };

    // Set zoom behaviour
    zoom = d3.zoom()
             .scaleExtent([minZoom, maxZoom])
             .extent([[0, 0], [width, height]])
             .translateExtent([[0, 0], [width, height]])
             .on("zoom", zoomed);

    // Create plot elements
    svg = d3.select("#graph-container")
             .append("svg")
               .attr("width", width + margin.left + margin.right)
               .attr("height", height + margin.top + margin.bottom);

    clipArea = svg.append("defs")
                   .append("clipPath")
                   .attr("id", "clip")
                   .append("rect")
                   .attr("width", width)
                   .attr("height", height);

    window.graphArea = svg.append("g")
                      .attr("id", "graph")
                      .attr("height", height)
                      .attr("width", width)
                      .attr("transform", "translate(" + margin.left + "," + margin.top + ")")
                      .attr("clip-path", "url(#clip)")
                      .on("mouseover", function(){
                          showCoords(this, true);
                      })
                      .on("mousemove", function(){
                          showCoords(this, true);
                      })
                      .on("mouseout", function(){
                          showCoords(this, false);
                      })
                      .on("mousedown", function(event) {
                          // Only handle left click
                          if (d3.event.button === 0) {
                              d3.event.stopPropagation();  // Prevent zoom from interfering
                              animatealgorithms.call(this);
                          }
                      })
                      .call(zoom);

    contourPlot = window.graphArea.selectAll("path")
                           .data(contours(z))
                           .enter()
                           .append("path")
                           .attr("d", d3.geoPath(d3.geoIdentity().scale(width / m)))
                           .attr("fill", function(d){return color(d.value);})
                           .attr("stroke", contourLineColor);

    // Initialize with Himmelblau function minima
    currentMinima = window.functionMinima['himmelblau'];
    
    // Set minimum points with larger radius
    minCircle = window.graphArea.selectAll(".stationary-min")
                           .data(currentMinima)
                           .enter()
                           .append("circle")
                           .attr("class", "stationary-min")
                           .attr("fill", radarColor)
                           .attr("r", 4)  // Fixed size instead of scaled
                           .attr("cx", function(d){return window.xScale(d.x)})
                           .attr("cy", function(d){return window.yScale(d.y)});

    radarCircle = window.graphArea.selectAll(".min")
                             .data(currentMinima)
                             .enter()
                             .append("circle")
                             .attr("class", "min")
                             .attr("fill", "none")
                             .attr("stroke", radarColor)
                             .attr("stroke-width", 1)
                             .attr("r", 8)  // Larger radius for outer circle
                             .attr("cx", function(d){return window.xScale(d.x)})
                             .attr("cy", function(d){return window.yScale(d.y)});

    xAxis = d3.axisBottom(window.xScale).tickSizeOuter(0);
    yAxis = d3.axisLeft(window.yScale).tickSizeOuter(0);

    gY = svg.append("g")
               .attr("transform", "translate(" + margin.left + "," + margin.top + ")")
               .call(yAxis);

    gX = svg.append("g")
               .attr("transform", "translate(" + margin.left + "," + (margin.top + height) + ")")
               .call(xAxis);

    setLabelPositions();

    yLabel = svg.append("text")
                   .attr("text-anchor", "middle")
                   .attr("transform", "translate(" + labelPosition.y + "," + (margin.top + (height / 2)) + ")")
                   .attr("font-weight", axisLabelWeight)
                   .text("Y");

    xLabel = svg.append("text")
                   .attr("text-anchor", "middle")
                   .attr("transform", "translate(" + (margin.left + (width / 2)) + "," + labelPosition.x + ")")
                   .attr("font-weight", axisLabelWeight)
                   .text("X");

    // Create display to hold current coordinates/function value
    window.coordBox = svg.append("rect")
                     .style("opacity", 0)
                     .attr("x", margin.left + width / 8)
                     .attr("y", margin.top / 4)
                     .attr("width", 3 * width / 4)
                     .attr("height", margin.top / 2)
                     .attr("stroke", coordTextColor)
                     .attr("stroke-width", 0.3)
                     .attr("fill", coordBoxColor);

    window.coordText = svg.append("text")
                      .style("opacity", 0)
                      .style("white-space", "pre")
                      .attr("text-anchor", "middle")
                      .attr("dy", "0.35em")
                      .attr("transform", "translate(" + (margin.left + width / 2) + "," + margin.top / 2 + ")");

    window.coordText.append("tspan")
               .text("x:  ")
               .style("font-weight", "bold");

    window.coordText.append("tspan")
               .attr("id", "x-coord");

    window.coordText.append("tspan")
               .text("     y:  ")
               .style("font-weight", "bold");

    window.coordText.append("tspan")
               .attr("id", "y-coord");

    window.coordText.append("tspan")
               .text("     f(x,y):  ")
               .style("font-weight", "bold");

    window.coordText.append("tspan")
               .attr("id", "f-val");

    // Help text telling user they can zoom
    if (window.innerWidth > cutoffWidth){
        var scrollAction = "Scroll";
    }else{
        var scrollAction = "Pinch";
    }

    zoomHelpText = window.graphArea.append("text")
                              .attr("text-anchor", "middle")
                              .attr("fill", helpTextColor)
                              .attr("transform", "translate(" + (width / 2) + "," + (height / 2) + ")")
                              .text("Use " + scrollAction + " to Zoom")
                              .on("mouseover", function(){zoomHelpText.remove();});
}

// Redraw graph after window resizes
function reDraw(){
    // Reset zoom 
    svg.call(zoom.transform, d3.zoomIdentity);

    // Set new size and margin of plotting area
    var container = d3.select("#graph-container").node();
    var containerSize = Math.min(container.clientWidth, container.clientHeight);
    
    setMargins();
    width = containerSize - margin.left - margin.right;
    height = containerSize - margin.top - margin.bottom;

    // Adjust zoom behaviour
    zoom.extent([[0, 0], [width, height]])
    .translateExtent([[0, 0], [width, height]]);

    // Adjust size of plot elements
    svg.attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom);

    clipArea.attr("width", width)
            .attr("height", height);

    window.graphArea.attr("height", height)
             .attr("width", width)
             .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

    contourPlot.attr("d", d3.geoPath(d3.geoIdentity().scale(width / m)))
               .attr("fill", function(d){return color(d.value);})
               .attr("stroke", "#666");

    // Adjust axes, labels and scales
    window.xScale.range([0, width]);
    window.yScale.range([height, 0]);

    window.currScale = {
        x: window.xScale,
        y: window.yScale
    };

    xAxis = d3.axisBottom(window.xScale).tickSizeOuter(0);
    yAxis = d3.axisLeft(window.yScale).tickSizeOuter(0);

    gY.attr("transform", "translate(" + margin.left + "," + margin.top + ")").call(yAxis);
    gX.attr("transform", "translate(" + margin.left + "," + (margin.top + height) + ")").call(xAxis);

    setLabelPositions();

    yLabel.attr("transform", "translate(" + labelPosition.y  + "," + (margin.top + (height / 2)) + ")");
    xLabel.attr("transform", "translate(" + (margin.left + (width / 2)) + "," + labelPosition.x + ")");

    // Adjust minimum points
    minCircle.attr("r", scaledDistance("x", minRadius))
             .attr("cx", function(d){return window.xScale(d.x)})
             .attr("cy", function(d){return window.yScale(d.y)});

    radarCircle.attr("cx", function(d){return window.xScale(d.x)})
               .attr("cy", function(d){return window.yScale(d.y)});


    // Adjust coordinates display
    window.coordText.attr("transform", "translate(" + (margin.left + width / 2) + "," + margin.top / 2 + ")");
    window.coordBox.attr("x", margin.left + width / 8)
            .attr("y", margin.top / 4)
            .attr("width", 3 * width / 4)
            .attr("height", margin.top / 2);

    // Adjust zoom help text
    zoomHelpText.attr("transform", "translate(" + (margin.left + width / 2) + "," + (margin.top + height / 2) + ")");
}

window.addEventListener("resize", reDraw);

// Add event listeners for algorithm checkboxes
document.addEventListener('DOMContentLoaded', function() {
    // Find all algorithm checkboxes
    const algoCheckboxes = document.querySelectorAll('.algorithm-option input[type="checkbox"]');
    
    // Add event listeners to each checkbox
    algoCheckboxes.forEach(checkbox => {
        checkbox.addEventListener('change', updateAlgorithmDescription);
    });
    
    // Initial algorithm description update (SGD is checked by default)
    updateAlgorithmDescription();
});

// Function to update algorithm description based on selected checkboxes
function updateAlgorithmDescription() {
    const descriptionElement = document.getElementById('algorithm-description');
    const selectedAlgorithms = getSelectedAlgorithms();
    
    if (selectedAlgorithms.length === 0) {
        descriptionElement.textContent = "Please select at least one algorithm.";
        return;
    }
    
    if (selectedAlgorithms.length === 1) {
        // If only one algorithm is selected, show its description
        const algorithm = selectedAlgorithms[0];
        descriptionElement.textContent = algorithmInfo[algorithm].description;
    } else {
        // If multiple algorithms are selected, show a comparison message
        descriptionElement.textContent = "Multiple algorithms selected. Compare their convergence patterns.";
    }
}