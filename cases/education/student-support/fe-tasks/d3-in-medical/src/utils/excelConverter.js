const XLSX = require('xlsx');
const fs = require('fs');
const path = require('path');

function convertExcelToJson() {
    // 读取 Excel 文件
    const workbook = XLSX.readFile(path.join(__dirname, '../../data/medical.xlsx'));
    const sheetName = workbook.SheetNames[0];
    const worksheet = workbook.Sheets[sheetName];
    
    // 转换为 JSON 数据
    const rawData = XLSX.utils.sheet_to_json(worksheet);
    
    // 处理副作用数据
    const sideEffectData = {};
    const drugs = [...new Set(rawData.map(item => item['使用药物']))];
    
    drugs.forEach(drug => {
        const drugData = rawData.filter(item => item['使用药物'] === drug);
        const sideEffects = {};
        
        drugData.forEach(item => {
            if (item['副作用'] && item['副作用'] !== '无') {
                sideEffects[item['副作用']] = (sideEffects[item['副作用']] || 0) + 1;
            }
        });
        
        sideEffectData[drug] = Object.entries(sideEffects).map(([effect, count]) => ({
            "副作用": effect,
            "发生率": Number((count / drugData.length).toFixed(2))
        }));
    });
    
    // 生成输出文件内容
    const outputContent = `// 这是从 Excel 导入的数据
const medicalData = ${JSON.stringify(rawData, null, 4)};

// 副作用数据
const sideEffectData = ${JSON.stringify(sideEffectData, null, 4)};

// 导出数据
module.exports = {
    medicalData,
    sideEffectData
};
`;
    
    // 写入文件
    fs.writeFileSync(
        path.join(__dirname, '../../data/sampleData.js'),
        outputContent,
        'utf8'
    );
    
    console.log('数据转换完成！');
}

// 执行转换
convertExcelToJson();