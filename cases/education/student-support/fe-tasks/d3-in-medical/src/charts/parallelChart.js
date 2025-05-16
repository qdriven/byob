class ParallelChart {
    constructor(container) {
        this.container = container;
        this.width = 1200;
        this.height = 400;
        this.margin = { top: 30, right: 50, bottom: 30, left: 50 };
        
        this.tooltip = d3.select('body')
            .append('div')
            .attr('class', 'parallel-tooltip')
            .style('opacity', 0);
            
        this.init();
        this.currentSeverity = 4;
        this.currentGender = 'all';
        this.ageRange = { min: 25, max: 70 };  // 修改默认年龄范围
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
            const genderMatch = this.currentGender === 'all' || d['性别'] === this.currentGender;
            const ageMatch = +d['年龄'] >= this.ageRange.min && +d['年龄'] <= this.ageRange.max;
            // 移除症状等级过滤
            return genderMatch && ageMatch;
        });
        
        return filteredData.map(d => ({
            性别: d['性别'] === '男' ? 1 : 0,
            年龄: +d['年龄'] || 0,
            症状等级: +d['症状严重程度（1-10级）'] || 0,
            用药剂量: +d['用药剂量（mg）'] || 0,
            治疗效果: +d['治疗效果（0：无效，1：有效，2：显著有效）'] || 0,
            药物: d['使用药物'],
            原始性别: d['性别']
        })).filter(d => 
            !isNaN(d.年龄) && 
            !isNaN(d.症状等级) && 
            !isNaN(d.用药剂量) && 
            !isNaN(d.治疗效果)
        );
    }

    render(data) {
        const processedData = this.processData(data);
        const dimensions = ['性别', '年龄', '症状等级', '用药剂量', '治疗效果'];
        
        const width = this.width - this.margin.left - this.margin.right;
        const height = this.height - this.margin.top - this.margin.bottom;

        // 创建比例尺
        const scales = {};
        dimensions.forEach(dim => {
            scales[dim] = d3.scaleLinear()
                .domain(d3.extent(processedData, d => d[dim]))
                .range([height, 0]);
        });

        // 创建坐标轴
        const xScale = d3.scalePoint()
            .domain(dimensions)
            .range([0, width]);

        // 创建线条生成器
        // Update line generator to handle missing values
        const line = d3.line()
            .defined(d => !isNaN(d.value) && d.value !== null)
            .x(d => xScale(d.dimension))
            .y(d => scales[d.dimension](d.value));

        // 清除之前的内容
        this.svg.selectAll("*").remove();

        // 创建主绘图组
        const g = this.svg.append("g")
            .attr("transform", `translate(${this.margin.left},${this.margin.top})`);

        // 绘制坐标轴
        // 修改坐标轴绘制部分
        dimensions.forEach(dim => {
            const axis = d3.axisLeft(scales[dim]);
            if (dim === '性别') {
                // 为性别添加特殊刻度
                axis.tickFormat(d => d === 1 ? '男' : '女');
            }
            g.append("g")
                .attr("transform", `translate(${xScale(dim)},0)`)
                .call(axis)
                .append("text")
                .attr("y", -10)
                .attr("text-anchor", "middle")
                .text(dim)
                .style("fill", "black");
        });

        // 绘制线条
        g.selectAll(".parallel-line")
            .data(processedData)
            .join("path")
            .attr("class", "parallel-line")
            .attr("d", d => {
                const points = dimensions.map(dim => ({
                    dimension: dim,
                    value: d[dim]
                }));
                return line(points);
            })
            .style("fill", "none")
            .style("stroke", d => {
                // 根据治疗效果设置颜色
                const effectColors = ["#ff7f0e", "#2ca02c", "#1f77b4"];
                return effectColors[d.治疗效果];
            })
            .style("stroke-width", 1.5)
            .style("opacity", 0.5)
            .on("mouseover", (event, d) => {
                // 高亮显示当前线条
                d3.select(event.target)
                    .style("stroke-width", 3)
                    .style("opacity", 1);
                
                // 显示tooltip
                this.tooltip
                    .style("opacity", 1)
                    .html(`
                        <div>药物: ${d.药物}</div>
                        <div>性别: ${d.原始性别}</div>
                        <div>年龄: ${d.年龄}</div>
                        <div>症状等级: ${d.症状等级}</div>
                        <div>用药剂量: ${d.用药剂量}</div>
                        <div>治疗效果: ${['无效', '有效', '显著有效'][d.治疗效果]}</div>
                    `)
                    .style("left", (event.pageX + 10) + "px")
                    .style("top", (event.pageY - 28) + "px");
            })
            .on("mouseout", (event) => {
                // 恢复线条样式
                d3.select(event.target)
                    .style("stroke-width", 1.5)
                    .style("opacity", 0.5);
                
                // 隐藏tooltip
                this.tooltip.style("opacity", 0);
            });
    }

    normalizeAmount(amount) {
        if (!amount || amount === 'undefined') return 0;
        const str = amount.toString().trim();
        console.log('处理前的剂量:', str);

        // 移除所有空格
        const cleanStr = str.replace(/\s+/g, '');
        
        // 处理不同单位的剂量
        if (cleanStr.includes('万单位')) {
            const num = cleanStr.match(/[\d.]+/);
            return num ? parseFloat(num[0]) * 10000 : 0;
        }
        
        if (cleanStr.toLowerCase().includes('mg')) {
            const num = cleanStr.match(/[\d.]+/);
            return num ? parseFloat(num[0]) : 0;
        }
        
        if (cleanStr.toLowerCase().endsWith('g')) {
            const num = cleanStr.match(/[\d.]+/);
            return num ? parseFloat(num[0]) * 1000 : 0;
        }
        
        // 处理纯数字
        const num = cleanStr.match(/[\d.]+/);
        console.log('提取的数字:', num);
        return num ? parseFloat(num[0]) : 0;
    }

    update(data, filters) {
        if (filters.severity !== undefined) this.currentSeverity = filters.severity;
        if (filters.gender !== undefined) this.currentGender = filters.gender;
        if (filters.ageRange !== undefined) this.ageRange = filters.ageRange;
        this.render(data);
    }
}