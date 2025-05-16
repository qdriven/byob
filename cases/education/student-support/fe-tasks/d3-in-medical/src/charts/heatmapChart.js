class HeatmapChart {
    constructor(container) {
        this.container = container;
        this.width = 600;
        this.height = 400;
        this.margin = { top: 40, right: 30, bottom: 60, left: 120 };
        
        this.tooltip = d3.select('body')
            .append('div')
            .attr('class', 'heatmap-tooltip')
            .style('opacity', 0);
            
        this.init();
        this.currentSeverity = 4;
        this.currentGender = 'all';
        this.ageRange = { min: 25, max: 70 };
    }

    init() {
        d3.select(this.container).html("");
        
        this.svg = d3.select(this.container)
            .append("svg")
            .attr("width", this.width)
            .attr("height", this.height);
    }

    processData(data) {
        // 多重过滤条件
        const filteredData = data.filter(d => {
            const severityMatch = +d['症状严重程度（1-10级）'] === this.currentSeverity;
            const genderMatch = this.currentGender === 'all' || d['性别'] === this.currentGender;
            const ageMatch = +d['年龄'] >= this.ageRange.min && +d['年龄'] <= this.ageRange.max;
            return severityMatch && genderMatch && ageMatch;
        });
        
        // 获取所有药物
        const drugs = [...new Set(filteredData.map(d => d['使用药物']))];
        
        // 计算每种药物的效果分布
        const heatmapData = drugs.map(drug => {
            const drugData = filteredData.filter(d => d['使用药物'] === drug);
            const total = drugData.length;
            
            return {
                drug,
                '无效': (drugData.filter(d => d['治疗效果（0：无效，1：有效，2：显著有效）'] === 0).length / total) || 0,
                '有效': (drugData.filter(d => d['治疗效果（0：无效，1：有效，2：显著有效）'] === 1).length / total) || 0,
                '显著有效': (drugData.filter(d => d['治疗效果（0：无效，1：有效，2：显著有效）'] === 2).length / total) || 0,
                total
            };
        }).filter(d => d.total > 0);

        return heatmapData;
    }

    // render 方法基本保持不变，只修改标题显示
    render(data) {
        const processedData = this.processData(data);
        const effects = ['无效', '有效', '显著有效'];
        
        // 计算实际绘图区域
        const width = this.width - this.margin.left - this.margin.right;
        const height = this.height - this.margin.top - this.margin.bottom;

        // 创建比例尺
        const xScale = d3.scaleBand()
            .domain(effects)
            .range([0, width])
            .padding(0.05);

        const yScale = d3.scaleBand()
            .domain(processedData.map(d => d.drug))
            .range([0, height])
            .padding(0.05);

        // 创建颜色比例尺
        const colorScale = d3.scaleSequential()
            .domain([0, 1])
            .interpolator(d3.interpolateRdYlBu);

        // 清除之前的内容
        this.svg.selectAll("*").remove();

        // 创建主绘图组
        const g = this.svg.append("g")
            .attr("transform", `translate(${this.margin.left},${this.margin.top})`);

        // 添加标题
        // 更新标题，添加过滤条件信息
        const genderText = this.currentGender === 'all' ? '全部性别' : this.currentGender;
        this.svg.append("text")
            .attr("x", this.width / 2)
            .attr("y", 20)
            .attr("text-anchor", "middle")
            .style("font-size", "16px")
            .text(`症状${this.currentSeverity}级 - ${genderText} - ${this.ageRange.min}-${this.ageRange.max}岁 - 药物疗效分布`);

        // 绘制热力图单元格
        effects.forEach(effect => {
            g.selectAll(`.${effect}-cells`)
                .data(processedData)
                .join("rect")
                .attr("x", xScale(effect))
                .attr("y", d => yScale(d.drug))
                .attr("width", xScale.bandwidth())
                .attr("height", yScale.bandwidth())
                .attr("fill", d => colorScale(d[effect]))
                .on("mouseover", (event, d) => {
                    const percentage = (d[effect] * 100).toFixed(1);
                    this.tooltip
                        .style("opacity", 1)
                        .html(`
                            <div>药物: ${d.drug}</div>
                            <div>效果: ${effect}</div>
                            <div>比例: ${percentage}%</div>
                            <div>样本数: ${d.total}</div>
                        `)
                        .style("left", (event.pageX + 10) + "px")
                        .style("top", (event.pageY - 28) + "px");
                })
                .on("mouseout", () => {
                    this.tooltip.style("opacity", 0);
                });
        });

        // 添加X轴
        g.append("g")
            .attr("transform", `translate(0,${height})`)
            .call(d3.axisBottom(xScale))
            .selectAll("text")
            .style("text-anchor", "middle");

        // 添加Y轴
        g.append("g")
            .call(d3.axisLeft(yScale))
            .selectAll("text")
            .style("text-anchor", "end");
    }

    update(data, filters) {
        if (filters.severity !== undefined) this.currentSeverity = filters.severity;
        if (filters.gender !== undefined) this.currentGender = filters.gender;
        if (filters.ageRange !== undefined) this.ageRange = filters.ageRange;
        this.render(data);
    }
}