import pandas as pd
import os

# 创建一个新文件夹用于保存Excel文件
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 创建示例数据
data = {
    "测试问题": [
        "大学生普通门诊报销标准",
        "医保卡如何办理",
        "异地就医如何报销",
        "生育保险报销流程",
        "医保卡余额查询方式"
    ],
    "测试答案": [
        "大学生普通门诊报销标准如下：\n\n1. 报销条件：\n- 在定点医疗机构就诊\n- 符合基本医疗保险药品目录、诊疗项目和医疗服务设施标准\n\n2. 报销比例：50%\n\n3. 最高支付限额：每年2000元",
        "这里是医保卡办理的预期结果",
        "这里是异地就医报销的预期结果",
        "这里是生育保险报销流程的预期结果",
        "这里是医保卡余额查询方式的预期结果"
    ]
}

# 创建DataFrame
df = pd.DataFrame(data)

# 保存为Excel文件
output_file = os.path.join(output_dir, "测试问题样例.xlsx")
df.to_excel(output_file, index=False)

print(f"示例Excel文件已创建: {output_file}") 