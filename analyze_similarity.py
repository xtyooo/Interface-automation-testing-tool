import pandas as pd
import numpy as np
import re
import jieba
from collections import Counter
import math
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from datetime import datetime

# 允许通过命令行参数指定输入文件
if len(sys.argv) > 1:
    excel_path = sys.argv[1]
else:
    excel_path = '效果测试.xlsx'  # 默认文件名

print(f"正在读取文件: {excel_path}")
df = pd.read_excel(excel_path)

# 获取列名
columns = df.columns.tolist()
if len(columns) < 3:
    print("错误：Excel文件至少需要三列数据（问题、预期结果和真实结果）")
    exit(1)

# 第二列是预期结果，第三列是真实结果
expected_col = columns[1]  # 第二列
actual_col = columns[2]    # 第三列

print(f"预期结果列: {expected_col}")
print(f"真实结果列: {actual_col}")

# 文本预处理函数
def preprocess_text(text):
    if pd.isna(text):
        return []
    # 转为字符串
    text = str(text)
    # 去除标点符号和特殊字符
    text = re.sub(r'[^\w\s]', '', text)
    # 分词
    words = jieba.lcut(text)
    # 去除停用词（简单处理）
    stopwords = {'的', '了', '是', '在', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}
    words = [word for word in words if word not in stopwords and len(word.strip()) > 0]
    return words

# 计算余弦相似度
def cosine_similarity(vec1, vec2):
    intersection = set(vec1) & set(vec2)
    if not intersection:
        return 0.0
    
    # 计算词频向量
    counter1 = Counter(vec1)
    counter2 = Counter(vec2)
    
    # 计算分子（点积）
    numerator = sum(counter1[x] * counter2[x] for x in intersection)
    
    # 计算分母（向量长度的乘积）
    sum1 = sum(counter1[x] ** 2 for x in counter1.keys())
    sum2 = sum(counter2[x] ** 2 for x in counter2.keys())
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    
    if not denominator:
        return 0.0
    
    return numerator / denominator

# 计算Jaccard相似度
def jaccard_similarity(vec1, vec2):
    set1 = set(vec1)
    set2 = set(vec2)
    
    if not set1 or not set2:
        return 0.0
    
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    return intersection / union

# 定义评估函数
def evaluate_similarity(expected, actual):
    # 计算语义相似度
    if pd.isna(expected) or pd.isna(actual):
        return "数据缺失"
    
    if actual == "请求失败":
        return "请求失败"
    
    # 文本预处理
    expected_words = preprocess_text(expected)
    actual_words = preprocess_text(actual)
    
    if not expected_words or not actual_words:
        return "文本为空"
    
    # 计算余弦相似度
    cos_sim = cosine_similarity(expected_words, actual_words)
    
    # 计算Jaccard相似度
    jac_sim = jaccard_similarity(expected_words, actual_words)
    
    # 综合相似度（加权平均）
    similarity = 0.7 * cos_sim + 0.3 * jac_sim
    
    # 根据相似度评估效果
    if similarity >= 0.9:
        return f"优秀 ({similarity:.2f}): 语义几乎完全一致"
    elif similarity >= 0.8:
        return f"良好 ({similarity:.2f}): 语义非常接近"
    elif similarity >= 0.7:
        return f"较好 ({similarity:.2f}): 语义基本相符"
    elif similarity >= 0.6:
        return f"一般 ({similarity:.2f}): 语义有一定差异"
    elif similarity >= 0.5:
        return f"较差 ({similarity:.2f}): 语义存在明显差异"
    else:
        return f"差距较大 ({similarity:.2f}): 语义相差很大"

# 添加评估结果列
print("正在计算语义相似度...")
df['效果评估'] = df.apply(lambda row: evaluate_similarity(row[expected_col], row[actual_col]), axis=1)

# 计算整体评估统计
total_rows = len(df)
excellent = len(df[df['效果评估'].str.contains('优秀', na=False)])
good = len(df[df['效果评估'].str.contains('良好', na=False)])
fair = len(df[df['效果评估'].str.contains('较好', na=False)])
average = len(df[df['效果评估'].str.contains('一般', na=False)])
poor = len(df[df['效果评估'].str.contains('较差', na=False)])
bad = len(df[df['效果评估'].str.contains('差距较大', na=False)])
failed = len(df[df['效果评估'] == "请求失败"])
missing = len(df[df['效果评估'].str.contains('数据缺失|文本为空', na=False, regex=True)])

# 计算有效样本数（排除请求失败和数据缺失的情况）
valid_samples = total_rows - failed - missing
valid_excellent = excellent / valid_samples * 100 if valid_samples > 0 else 0
valid_good = good / valid_samples * 100 if valid_samples > 0 else 0
valid_fair = fair / valid_samples * 100 if valid_samples > 0 else 0

print("\n整体评估统计:")
print(f"总样本数: {total_rows}")
print(f"有效样本数: {valid_samples}")
print(f"优秀: {excellent} ({excellent/total_rows*100:.1f}% 总体, {valid_excellent:.1f}% 有效)")
print(f"良好: {good} ({good/total_rows*100:.1f}% 总体, {valid_good:.1f}% 有效)")
print(f"较好: {fair} ({fair/total_rows*100:.1f}% 总体, {valid_fair:.1f}% 有效)")
print(f"一般: {average} ({average/total_rows*100:.1f}% 总体)")
print(f"较差: {poor} ({poor/total_rows*100:.1f}% 总体)")
print(f"差距较大: {bad} ({bad/total_rows*100:.1f}% 总体)")
print(f"请求失败: {failed} ({failed/total_rows*100:.1f}% 总体)")
print(f"数据缺失或为空: {missing} ({missing/total_rows*100:.1f}% 总体)")

# 计算合格率（优秀+良好+较好）
qualified_rate = (excellent + good + fair) / valid_samples * 100 if valid_samples > 0 else 0
print(f"\n合格率: {qualified_rate:.1f}%")

# 生成可视化图表
print("\n正在生成评估结果图表...")

# 创建输出目录
output_dir = "analysis_results"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 获取输入文件的基本名称，用于生成输出文件名
input_basename = os.path.basename(excel_path)
input_name, _ = os.path.splitext(input_basename)

# 使用输入文件名作为输出文件名的一部分
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
chart_path = os.path.join(output_dir, f'{input_name}_评估分布_{timestamp}.png')
output_path = os.path.join(output_dir, f'{input_name}_评估结果_{timestamp}.xlsx')
html_output = os.path.join(output_dir, f'{input_name}_评估报告_{timestamp}.html')

# 饼图数据
labels = ['优秀', '良好', '较好', '一般', '较差', '差距较大', '请求失败', '数据缺失/为空']
sizes = [excellent, good, fair, average, poor, bad, failed, missing]
colors = ['#4CAF50', '#8BC34A', '#CDDC39', '#FFC107', '#FF9800', '#F44336', '#9E9E9E', '#607D8B']
explode = (0.1, 0, 0, 0, 0, 0, 0, 0)  # 突出显示"优秀"部分

# 设置中文字体
try:
    font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf")
except:
    print("警告: 无法加载中文字体，图表中的中文可能显示不正确")
    font = FontProperties()

plt.figure(figsize=(12, 8))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=90, textprops={'fontproperties': font})
plt.axis('equal')  # 保证饼图是圆的
plt.title('效果评估结果分布', fontproperties=font, fontsize=16)
plt.tight_layout()

# 保存饼图
plt.savefig(chart_path)
print(f"图表已保存至: {chart_path}")

# 保存结果到新的Excel文件
print(f"\n正在保存结果到: {output_path}")
df.to_excel(output_path, index=False)

# 生成HTML报告
print(f"正在生成HTML报告: {html_output}")

html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>效果测试评估报告</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        .summary {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        .summary-item {{ margin: 10px 0; }}
        .progress {{ background-color: #e0e0e0; height: 20px; border-radius: 10px; }}
        .progress-bar {{ background-color: #4CAF50; height: 20px; border-radius: 10px; text-align: center; color: white; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .excellent {{ color: #4CAF50; }}
        .good {{ color: #8BC34A; }}
        .fair {{ color: #CDDC39; }}
        .average {{ color: #FFC107; }}
        .poor {{ color: #FF9800; }}
        .bad {{ color: #F44336; }}
        .failed {{ color: #9E9E9E; }}
        .missing {{ color: #607D8B; }}
        .chart-container {{ text-align: center; margin: 20px 0; }}
    </style>
</head>
<body>
    <h1>效果测试评估报告</h1>
    <div class="summary">
        <div class="summary-item"><strong>测试文件:</strong> {os.path.basename(excel_path)}</div>
        <div class="summary-item"><strong>生成时间:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>
        <div class="summary-item"><strong>总样本数:</strong> {total_rows}</div>
        <div class="summary-item"><strong>有效样本数:</strong> {valid_samples}</div>
        <div class="summary-item">
            <strong>合格率:</strong> {qualified_rate:.1f}%
            <div class="progress">
                <div class="progress-bar" style="width: {qualified_rate}%">{qualified_rate:.1f}%</div>
            </div>
        </div>
    </div>
    
    <h2>评估结果统计</h2>
    <table>
        <tr>
            <th>评估等级</th>
            <th>数量</th>
            <th>占总体比例</th>
            <th>占有效样本比例</th>
        </tr>
        <tr class="excellent">
            <td>优秀</td>
            <td>{excellent}</td>
            <td>{excellent/total_rows*100:.1f}%</td>
            <td>{valid_excellent:.1f}%</td>
        </tr>
        <tr class="good">
            <td>良好</td>
            <td>{good}</td>
            <td>{good/total_rows*100:.1f}%</td>
            <td>{valid_good:.1f}%</td>
        </tr>
        <tr class="fair">
            <td>较好</td>
            <td>{fair}</td>
            <td>{fair/total_rows*100:.1f}%</td>
            <td>{valid_fair:.1f}%</td>
        </tr>
        <tr class="average">
            <td>一般</td>
            <td>{average}</td>
            <td>{average/total_rows*100:.1f}%</td>
            <td>{average/valid_samples*100:.1f}%</td>
        </tr>
        <tr class="poor">
            <td>较差</td>
            <td>{poor}</td>
            <td>{poor/total_rows*100:.1f}%</td>
            <td>{poor/valid_samples*100:.1f}%</td>
        </tr>
        <tr class="bad">
            <td>差距较大</td>
            <td>{bad}</td>
            <td>{bad/total_rows*100:.1f}%</td>
            <td>{bad/valid_samples*100:.1f}%</td>
        </tr>
        <tr class="failed">
            <td>请求失败</td>
            <td>{failed}</td>
            <td>{failed/total_rows*100:.1f}%</td>
            <td>-</td>
        </tr>
        <tr class="missing">
            <td>数据缺失或为空</td>
            <td>{missing}</td>
            <td>{missing/total_rows*100:.1f}%</td>
            <td>-</td>
        </tr>
    </table>
    
    <div class="chart-container">
        <h2>评估结果分布</h2>
        <img src="{os.path.basename(chart_path)}" alt="评估结果分布图" style="max-width: 100%;">
    </div>
    
    <h2>详细评估结果</h2>
    <p>详细结果已保存至Excel文件: {os.path.basename(output_path)}</p>
</body>
</html>
"""

with open(html_output, 'w', encoding='utf-8') as f:
    f.write(html_content)

print("HTML报告已生成")
print("处理完成！") 