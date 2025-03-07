# 接口自动化测试工具

一个基于Python的GUI接口自动化测试工具，支持批量测试、结果分析和报告生成。

## 功能特点

- 📝 支持Excel格式的测试用例批量导入
- 🔄 可配置多个智能体Token，方便切换测试环境
- ⏱️ 可自定义请求间隔，避免频率限制
- 📊 自动生成测试报告，包含详细的统计分析
- 📈 支持相似度分析，评估测试结果质量
- 🎨 美观的图形用户界面，操作简单直观
- 📋 实时日志显示，测试过程可视化
- 🛑 支持随时暂停/停止测试
- 💾 自动保存测试结果，支持后续分析

## 系统要求

- Python 3.6 或更高版本
- Windows/Linux/MacOS 操作系统

## 安装说明

1. 克隆仓库到本地：
```bash
git clone https://github.com/xtyooo/Interface-automation-testing-tool.git
cd Interface-automation-testing-tool
```

2. 安装依赖包：
```bash
pip install -r requirements.txt
```
## 快速开始

1. 双击运行 `start.bat`（Windows）或执行 `python gui_test.py`（所有平台）
2. 在界面上选择测试用例文件（Excel格式）
3. 输入或选择智能体Token
4. 设置输出文件路径和请求间隔
5. 点击"开始测试"按钮开始测试

## 测试用例格式

测试用例Excel文件需包含以下列：
- `问题`：测试输入
- `预期结果`：预期的输出结果（可选）
- `真实测试结果`：将由工具自动填充

## 测试报告

测试完成后会自动生成：
- Excel格式的详细测试结果
- HTML格式的分析报告
- 结果分布饼图
- 相似度评估结果

## 配置说明

### 智能体Token配置
可在GUI界面直接配置，也可以修改代码中的预设值：
```python
self.preset_agents = {
    "默认智能体": "app-CmBCgYDKd9yGjmgV1PnNSeZ4",
    "智能体2": "app-另一个Token",
    "智能体3": "app-第三个Token"
}
```

### 请求间隔
- 默认值：3秒
- 可在界面上实时调整
- 建议根据服务器负载情况适当调整

## 目录结构

```
api-test-tool/
├── gui_test.py          # 主程序
├── analyze_similarity.py # 结果分析脚本
├── start.bat            # 启动脚本
├── requirements.txt     # 依赖包列表
├── output/             # 测试结果输出目录
└── analysis_results/   # 分析报告输出目录
```

## 常见问题

1. **Q: 程序无法启动？**
   A: 请确保已安装Python和所有依赖包。

2. **Q: 测试结果分析失败？**
   A: 检查Excel文件格式是否正确，确保包含必要的列。

3. **Q: 请求总是失败？**
   A: 检查网络连接和Token是否正确。

## 开发计划

- [ ] 支持更多数据格式（JSON、CSV等）
- [ ] 添加更多数据分析功能
- [ ] 支持自定义请求头和参数
- [ ] 添加测试用例管理功能
- [ ] 支持导出更多格式的报告

## 贡献指南

欢迎提交Issue和Pull Request！

1. Fork 本仓库
2. 创建新的分支 `git checkout -b feature/your-feature`
3. 提交更改 `git commit -am 'Add some feature'`
4. 推送到分支 `git push origin feature/your-feature`
5. 创建Pull Request

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 联系方式

如有问题或建议，欢迎通过以下方式联系：
- 提交 Issue
- 发送邮件至：[1286214601@qq.com]

## 致谢

感谢所有贡献者的支持！ 
