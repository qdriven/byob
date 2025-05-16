class SankeyChart {
    constructor(container) {
        this.container = container;
        this.width = 1200;
        this.height = 600;
        this.margin = { top: 20, right: 20, bottom: 20, left: 20 };
        
        this.tooltip = d3.select('body')
            .append('div')
            .attr('class', 'sankey-tooltip')
            .style('opacity', 0);
            
        this.init();
        
        // 添加过滤状态
        this.currentSeverity = 4;
    }

    init() {
        d3.select(this.container).html("");
        
        this.svg = d3.select(this.container)
            .append("svg")
            .attr("width", this.width)
            .attr("height", this.height);

        this.sankey = d3.sankey()
            .nodeWidth(15)
            .nodePadding(20)
            .extent([[this.margin.left, this.margin.top], 
                    [this.width - this.margin.right, this.height - this.margin.bottom]]);
    }

    processData(data) {
        const nodes = [];
        const links = [];
        
        // 处理症状严重程度节点
        const severities = [...new Set(data.map(d => d['症状严重程度（1-10级）']))].sort();
        severities.forEach(severity => {
            nodes.push({
                id: `症状${severity}级`,
                name: `症状${severity}级`,
                category: 'severity'
            });
        });
    
        // 处理药物节点
        const drugs = [...new Set(data.map(d => d['使用药物']))].filter(drug => drug && drug !== 'undefined');
        drugs.forEach(drug => {
            nodes.push({
                id: drug,
                name: drug,
                category: 'drug'
            });
        });
    
        // 处理治疗效果节点
        const effects = ['无效', '有效', '显著有效'];
        effects.forEach(effect => {
            nodes.push({
                id: effect,
                name: effect,
                category: 'effect'
            });
        });
    
        // 创建症状到药物的链接
        data.forEach(d => {
            const severity = d['症状严重程度（1-10级）'];
            const drug = d['使用药物'];
            
            if (severity && drug && drug !== 'undefined') {
                const sourceId = `症状${severity}级`;
                const sourceNode = nodes.find(n => n.id === sourceId);
                const targetNode = nodes.find(n => n.id === drug);
                
                if (sourceNode && targetNode) {
                    const existingLink = links.find(l => 
                        (l.source === sourceNode && l.target === targetNode) ||
                        (l.source.id === sourceId && l.target.id === drug)
                    );
                    if (existingLink) {
                        existingLink.value++;
                    } else {
                        links.push({
                            source: sourceNode,
                            target: targetNode,
                            value: 1
                        });
                    }
                }
            }
        });

        // 创建药物到治疗效果的链接
        data.forEach(d => {  // Changed from filteredData to data
            const drug = d['使用药物'];
            const effect = ['无效', '有效', '显著有效'][d['治疗效果（0：无效，1：有效，2：显著有效）']];
            
            if (drug && drug !== 'undefined' && effect) {
                const sourceNode = nodes.find(n => n.id === drug);
                const targetNode = nodes.find(n => n.id === effect);
                
                if (sourceNode && targetNode) {
                    const existingLink = links.find(l => 
                        (l.source === sourceNode && l.target === targetNode) ||
                        (l.source.id === drug && l.target.id === effect)
                    );
                    if (existingLink) {
                        existingLink.value++;
                    } else {
                        links.push({
                            source: sourceNode,
                            target: targetNode,
                            value: 1
                        });
                    }
                }
            }
        });

        return { nodes, links };
    }

    highlightPaths(severity) {
        this.currentSeverity = severity;
        const sourceNode = `症状${severity}级`;
        
        // 找到与当前症状相关的所有药物节点
        const relatedDrugs = new Set();
        this.svg.selectAll("path")
            .filter(d => d.source.name === sourceNode)
            .each(d => relatedDrugs.add(d.target.name));

        // 更新连接的样式
        this.svg.selectAll("path")
            .classed("path-highlight", d => {
                // 高亮症状到药物的路径
                if (d.source.name === sourceNode) return true;
                // 高亮相关药物到效果的路径
                if (relatedDrugs.has(d.source.name)) return true;
                return false;
            })
            .classed("path-dimmed", d => {
                // 淡化非相关路径
                if (d.source.name === sourceNode) return false;
                if (relatedDrugs.has(d.source.name)) return false;
                return true;
            });
            
        // 更新节点的样式
        this.svg.selectAll("rect")
            .style("opacity", d => {
                if (d.name === sourceNode) return 1;
                if (relatedDrugs.has(d.name)) return 1;
                // 检查是否是相关药物连接的效果节点
                const isEffectNode = Array.from(relatedDrugs).some(drug => {
                    return this.svg.selectAll("path")
                        .filter(p => p.source.name === drug && p.target.name === d.name)
                        .size() > 0;
                });
                return isEffectNode ? 1 : 0.3;
            });

        // 更新标签的样式
        this.svg.selectAll("text")
            .style("opacity", d => {
                if (d.name === sourceNode) return 1;
                if (relatedDrugs.has(d.name)) return 1;
                const isEffectNode = Array.from(relatedDrugs).some(drug => {
                    return this.svg.selectAll("path")
                        .filter(p => p.source.name === drug && p.target.name === d.name)
                        .size() > 0;
                });
                return isEffectNode ? 1 : 0.3;
            });
    }

    render(data) {
        // 清除之前的内容
        this.svg.selectAll("*").remove();
        
        const processedData = this.processData(data);
        const { nodes, links } = this.sankey(processedData);

        // 计算总流量用于百分比显示
        const totalFlow = d3.sum(links, d => d.value);

        // 定义节点颜色
        const colorScale = d3.scaleOrdinal()
            .domain(['severity', 'drug', 'effect'])
            .range(['#ff7f0e', '#2ca02c', '#1f77b4']);

        // 绘制连接
        this.svg.append("g")
            .selectAll("path")
            .data(links)
            .join("path")
            .attr("d", d3.sankeyLinkHorizontal())
            .attr("stroke-width", d => Math.max(1, d.width))
            .attr("stroke", "#aaa")
            .attr("fill", "none")
            .attr("opacity", 0.5)
            .on("mouseover", (event, d) => {
                const percentage = ((d.value / totalFlow) * 100).toFixed(1);
                this.tooltip
                    .style("opacity", 1)
                    .html(`
                        <div>从 ${d.source.name} 到 ${d.target.name}</div>
                        <div>数量: ${d.value} (${percentage}%)</div>
                    `)
                    .style("left", (event.pageX + 10) + "px")
                    .style("top", (event.pageY - 28) + "px");
            })
            .on("mouseout", () => {
                this.tooltip.style("opacity", 0);
            });

        // 绘制节点
        this.svg.append("g")
            .selectAll("rect")
            .data(nodes)
            .join("rect")
            .attr("x", d => d.x0)
            .attr("y", d => d.y0)
            .attr("height", d => d.y1 - d.y0)
            .attr("width", d => d.x1 - d.x0)
            .attr("fill", d => colorScale(d.category))
            .attr("opacity", 0.8);

        // 添加节点标签
        // 修改节点标签部分
        this.svg.append("g")
            .selectAll("text")
            .data(nodes)
            .join("text")
            .attr("x", d => d.x0 < this.width / 2 ? d.x1 + 6 : d.x0 - 6)
            .attr("y", d => (d.y1 + d.y0) / 2)
            .attr("dy", "0.35em")
            .attr("text-anchor", d => d.x0 < this.width / 2 ? "start" : "end")
            .text(d => {
                // 计算该节点的总流量
                const nodeFlow = d.value || 0;
                const percentage = ((nodeFlow / totalFlow) * 100).toFixed(1);
                return `${d.name} (${percentage}%)`;
            })
            .style("font-size", "12px");
    }
    // Add after render method
    update(data, filters) {
        if (filters.severity !== undefined) {
            this.currentSeverity = filters.severity;
        }
        // 不需要重新初始化，只需要重新渲染
        this.render(data);
        // Apply highlighting
        this.highlightPaths(this.currentSeverity);
    }
}