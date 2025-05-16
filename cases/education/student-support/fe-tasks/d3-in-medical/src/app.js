document.addEventListener('DOMContentLoaded', () => {
    const sankeyChart = new SankeyChart('#sankey-chart');
    const parallelChart = new ParallelChart('#parallel-chart');
    const heatmapChart = new HeatmapChart('#heatmap-chart');

    // 初始渲染
    sankeyChart.render(medicalData);
    parallelChart.render(medicalData);
    heatmapChart.render(medicalData);
    
    // Sankey图表的滑块事件监听
    const sankeySlider = document.getElementById('sankey-severity-slider');
    const sankeySeverityValue = document.querySelector('.sankey-severity-value');
    
    if (sankeySlider && sankeySeverityValue) {
        sankeySlider.addEventListener('input', (event) => {
            const severity = parseInt(event.target.value);
            sankeySeverityValue.textContent = `${severity}级`;
            
            // 直接调用渲染和高亮
            sankeyChart.currentSeverity = severity;
            sankeyChart.render(medicalData);
            sankeyChart.highlightPaths(severity);
        });
    }

    // 热力图的过滤条件监听
    const heatmapSlider = document.getElementById('heatmap-severity-slider');
    const heatmapSeverityValue = document.querySelector('.heatmap-severity-value');
    const heatmapAgeMin = document.getElementById('heatmap-age-min');
    const heatmapAgeMax = document.getElementById('heatmap-age-max');
    const heatmapAgeValue = document.querySelector('.heatmap-age-value');
    const heatmapGenderSelect = document.getElementById('heatmap-gender-select');
    
    function updateHeatmapChart() {
        const filters = {
            severity: parseInt(heatmapSlider.value),
            gender: heatmapGenderSelect.value,
            ageRange: {
                min: parseInt(heatmapAgeMin.value),
                max: parseInt(heatmapAgeMax.value)
            }
        };
        heatmapChart.update(medicalData, filters);
    }

    heatmapSlider.addEventListener('input', (event) => {
        heatmapSeverityValue.textContent = `${event.target.value}级`;
        updateHeatmapChart();
    });

    heatmapAgeMin.addEventListener('input', (event) => {
        if (parseInt(event.target.value) > parseInt(heatmapAgeMax.value)) {
            heatmapAgeMin.value = heatmapAgeMax.value;
        }
        heatmapAgeValue.textContent = `${heatmapAgeMin.value}-${heatmapAgeMax.value}岁`;
        updateHeatmapChart();
    });

    heatmapAgeMax.addEventListener('input', (event) => {
        if (parseInt(event.target.value) < parseInt(heatmapAgeMin.value)) {
            heatmapAgeMax.value = heatmapAgeMin.value;
        }
        heatmapAgeValue.textContent = `${heatmapAgeMin.value}-${heatmapAgeMax.value}岁`;
        updateHeatmapChart();
    });

    heatmapGenderSelect.addEventListener('change', updateHeatmapChart);

    // 移除重复的 Sankey 图表监听代码
    
    // 平行坐标图的事件监听保持不变
    // Sankey图表的滑块事件监听
// Remove duplicate declaration since sankeySlider is already defined above
    //const sankeySeverityValue = document.querySelector('.sankey-severity-value');
    
    sankeySlider.addEventListener('input', (event) => {
        const value = parseInt(event.target.value);
        sankeySeverityValue.textContent = `${value}级`;
        sankeyChart.update(medicalData, value);
    });

    // 平行坐标图的滑块事件监听
    // 平行坐标图的过滤条件监听
    const parallelSlider = document.getElementById('parallel-severity-slider');
    const parallelSeverityValue = document.querySelector('.parallel-severity-value');
    const parallelAgeMin = document.getElementById('parallel-age-min');
    const parallelAgeMax = document.getElementById('parallel-age-max');
    const parallelAgeValue = document.querySelector('.parallel-age-value');
    const parallelGenderSelect = document.getElementById('parallel-gender-select');
    
    // 更新函数
    function updateParallelChart() {
        const filters = {
            severity: parseInt(parallelSlider.value),
            gender: parallelGenderSelect.value,
            ageRange: {
                min: parseInt(parallelAgeMin.value),
                max: parseInt(parallelAgeMax.value)
            }
        };
        parallelChart.update(medicalData, filters);
    }

    // 事件监听
    parallelSlider.addEventListener('input', (event) => {
        parallelSeverityValue.textContent = `${event.target.value}级`;
        updateParallelChart();
    });

    parallelAgeMin.addEventListener('input', (event) => {
        if (parseInt(event.target.value) > parseInt(parallelAgeMax.value)) {
            parallelAgeMin.value = parallelAgeMax.value;
        }
        parallelAgeValue.textContent = `${parallelAgeMin.value}-${parallelAgeMax.value}岁`;
        updateParallelChart();
    });

    parallelAgeMax.addEventListener('input', (event) => {
        if (parseInt(event.target.value) < parseInt(parallelAgeMin.value)) {
            parallelAgeMax.value = parallelAgeMin.value;
        }
        parallelAgeValue.textContent = `${parallelAgeMin.value}-${parallelAgeMax.value}岁`;
        updateParallelChart();
    });

    parallelGenderSelect.addEventListener('change', updateParallelChart);
});