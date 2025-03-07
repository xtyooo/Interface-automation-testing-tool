import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import os
import threading
import pandas as pd
import sys
import logging
import http.client
import json
import time
import re

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("test_log.log"),
        logging.StreamHandler()
    ]
)

def extract_answer(response_text):
    """
    从响应中提取answer字段的非<details>部分
    """
    try:
        # 将响应文本解析为JSON
        response_json = json.loads(response_text)
        
        # 获取answer字段
        answer = response_json.get("answer", "")
        
        # 移除<details>部分
        details_pattern = r'<details.*?</details>\s*\n*'
        clean_answer = re.sub(details_pattern, '', answer, flags=re.DOTALL)
        
        # 清理开头的空行
        clean_answer = clean_answer.lstrip('\n')
        
        return clean_answer
    except Exception as e:
        logging.error(f"提取答案时出错: {e}")
        return "提取失败"

class AutoTestGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("接口自动化测试工具")
        self.root.geometry("800x600")
        self.root.resizable(True, True)
        
        # 创建输出目录
        self.output_dir = "output"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # 创建主框架
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建标题
        title_label = ttk.Label(self.main_frame, text="接口自动化测试工具", font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # 创建设置框架
        settings_frame = ttk.LabelFrame(self.main_frame, text="测试设置", padding="10")
        settings_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # 文件选择
        file_frame = ttk.Frame(settings_frame)
        file_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(file_frame, text="测试文件:").pack(side=tk.LEFT, padx=5)
        self.file_path_var = tk.StringVar()
        file_entry = ttk.Entry(file_frame, textvariable=self.file_path_var, width=50)
        file_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        browse_button = ttk.Button(file_frame, text="浏览...", command=self.browse_file)
        browse_button.pack(side=tk.LEFT, padx=5)
        
        # 智能体选择
        agent_frame = ttk.Frame(settings_frame)
        agent_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(agent_frame, text="智能体Token:").pack(side=tk.LEFT, padx=5)
        self.agent_token_var = tk.StringVar(value="app-CmBCgYDKd9yGjmgV1PnNSeZ4")
        agent_entry = ttk.Entry(agent_frame, textvariable=self.agent_token_var, width=50)
        agent_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # 预设智能体下拉菜单
        self.preset_agents = {
            "默认智能体": "app-CmBCgYDKd9yGjmgV1PnNSeZ4",
            "智能体2": "app-另一个Token",
            "智能体3": "app-第三个Token"
        }
        
        ttk.Label(agent_frame, text="预设:").pack(side=tk.LEFT, padx=5)
        self.preset_var = tk.StringVar()
        preset_combo = ttk.Combobox(agent_frame, textvariable=self.preset_var, 
                                    values=list(self.preset_agents.keys()), width=15)
        preset_combo.pack(side=tk.LEFT, padx=5)
        preset_combo.bind("<<ComboboxSelected>>", self.on_preset_selected)
        
        # 输出文件设置
        output_frame = ttk.Frame(settings_frame)
        output_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(output_frame, text="输出文件:").pack(side=tk.LEFT, padx=5)
        self.output_path_var = tk.StringVar()
        output_entry = ttk.Entry(output_frame, textvariable=self.output_path_var, width=50)
        output_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        output_browse_button = ttk.Button(output_frame, text="浏览...", command=self.browse_output_file)
        output_browse_button.pack(side=tk.LEFT, padx=5)
        
        # 高级设置
        advanced_frame = ttk.LabelFrame(self.main_frame, text="高级设置", padding="10")
        advanced_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # 请求间隔
        interval_frame = ttk.Frame(advanced_frame)
        interval_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(interval_frame, text="请求间隔(秒):").pack(side=tk.LEFT, padx=5)
        self.interval_var = tk.StringVar(value="3")
        interval_entry = ttk.Entry(interval_frame, textvariable=self.interval_var, width=10)
        interval_entry.pack(side=tk.LEFT, padx=5)
        
        # 详细日志
        self.verbose_var = tk.BooleanVar(value=False)
        verbose_check = ttk.Checkbutton(advanced_frame, text="显示详细日志", variable=self.verbose_var)
        verbose_check.pack(anchor=tk.W, padx=5, pady=5)
        
        # 操作按钮
        button_frame = ttk.Frame(self.main_frame)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.run_button = ttk.Button(button_frame, text="开始测试", command=self.start_test)
        self.run_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(button_frame, text="停止测试", command=self.stop_test, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        analyze_button = ttk.Button(button_frame, text="分析结果", command=self.analyze_existing_results)
        analyze_button.pack(side=tk.LEFT, padx=5)
        
        clear_button = ttk.Button(button_frame, text="清空日志", command=self.clear_log)
        clear_button.pack(side=tk.LEFT, padx=5)
        
        # 日志显示
        log_frame = ttk.LabelFrame(self.main_frame, text="测试日志", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.log_text = tk.Text(log_frame, wrap=tk.WORD, height=10)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=scrollbar.set)
        
        # 状态栏
        self.status_var = tk.StringVar(value="就绪")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # 测试线程
        self.test_thread = None
        self.stop_event = threading.Event()
        
        # 自定义日志处理器
        self.log_handler = LogHandler(self.log_text)
        logging.getLogger().addHandler(self.log_handler)
        
        # 初始日志
        self.log("接口自动化测试工具已启动")
        self.log("请选择测试文件并设置智能体Token")
    
    def browse_file(self):
        file_path = filedialog.askopenfilename(
            title="选择测试文件",
            filetypes=[("Excel文件", "*.xlsx"), ("所有文件", "*.*")]
        )
        if file_path:
            self.file_path_var.set(file_path)
            # 自动设置输出文件名
            file_name = os.path.basename(file_path)
            name, ext = os.path.splitext(file_name)
            output_path = os.path.join(self.output_dir, f"{name}_结果{ext}")
            self.output_path_var.set(output_path)
    
    def browse_output_file(self):
        output_path = filedialog.asksaveasfilename(
            title="保存测试结果",
            defaultextension=".xlsx",
            filetypes=[("Excel文件", "*.xlsx"), ("所有文件", "*.*")],
            initialdir=self.output_dir
        )
        if output_path:
            self.output_path_var.set(output_path)
    
    def on_preset_selected(self, event):
        selected = self.preset_var.get()
        if selected in self.preset_agents:
            self.agent_token_var.set(self.preset_agents[selected])
    
    def log(self, message):
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
    
    def clear_log(self):
        self.log_text.delete(1.0, tk.END)
    
    def start_test(self):
        # 检查输入
        file_path = self.file_path_var.get().strip()
        if not file_path:
            messagebox.showerror("错误", "请选择测试文件")
            return
        
        if not os.path.exists(file_path):
            messagebox.showerror("错误", f"文件不存在: {file_path}")
            return
        
        agent_token = self.agent_token_var.get().strip()
        if not agent_token:
            messagebox.showerror("错误", "请输入智能体Token")
            return
        
        try:
            interval = float(self.interval_var.get())
            if interval < 0:
                raise ValueError("间隔时间不能为负数")
        except ValueError:
            messagebox.showerror("错误", "请输入有效的请求间隔时间")
            return
        
        output_path = self.output_path_var.get().strip()
        if not output_path:
            # 自动生成输出文件名
            file_name = os.path.basename(file_path)
            name, ext = os.path.splitext(file_name)
            output_path = os.path.join(self.output_dir, f"{name}_结果_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}{ext}")
            self.output_path_var.set(output_path)
        
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 设置日志级别
        if self.verbose_var.get():
            logging.getLogger().setLevel(logging.DEBUG)
        else:
            logging.getLogger().setLevel(logging.INFO)
        
        # 禁用运行按钮，启用停止按钮
        self.run_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.status_var.set("测试运行中...")
        
        # 重置停止事件
        self.stop_event.clear()
        
        # 启动测试线程
        self.test_thread = threading.Thread(
            target=self.run_test_thread,
            args=(file_path, output_path, agent_token, interval)
        )
        self.test_thread.daemon = True
        self.test_thread.start()
    
    def run_test_thread(self, input_file, output_file, agent_token, interval):
        try:
            self.log(f"开始测试，读取文件: {input_file}")
            
            # 读取Excel文件
            df = pd.read_excel(input_file)
            
            # 确保有"问题"列
            if "问题" not in df.columns:
                self.log("错误: 输入文件中没有'问题'列")
                return
            
            # 添加"真实测试结果"列
            df["真实测试结果"] = ""
            
            # 遍历每个问题
            total = len(df)
            success_count = 0
            fail_count = 0
            
            self.log(f"共有 {total} 个测试用例")
            
            for i, row in df.iterrows():
                # 检查是否需要停止
                if self.stop_event.is_set():
                    self.log("测试已手动停止")
                    break
                
                query = row["问题"]
                self.log(f"处理问题 {i+1}/{total}: {query[:50]}...")
                
                # 发送请求，使用自定义的agent_token
                response_text = self.custom_send_request(query, agent_token)
                
                if response_text:
                    # 提取答案
                    answer = extract_answer(response_text)
                    df.at[i, "真实测试结果"] = answer
                    
                    # 请求成功就算成功
                    success_count += 1
                else:
                    df.at[i, "真实测试结果"] = "请求失败"
                    fail_count += 1
                
                # 每次请求之间稍作暂停，避免请求过于频繁
                for _ in range(int(interval * 2)):
                    if self.stop_event.is_set():
                        break
                    time.sleep(0.5)
            
            # 保存结果
            df.to_excel(output_file, index=False)
            self.log(f"测试完成，结果已保存至: {output_file}")
            self.log(f"测试结果: 总计 {total} 个测试用例，通过 {success_count} 个，失败 {fail_count} 个")
            
            # 自动调用分析脚本
            self.analyze_results(output_file)
            
        except Exception as e:
            self.log(f"运行测试时出错: {e}")
        finally:
            # 恢复按钮状态
            self.root.after(0, self.reset_buttons)
    
    def custom_send_request(self, query, agent_token):
        """
        使用自定义token发送HTTP请求并返回响应
        """
        conn = http.client.HTTPConnection("127.0.0.1")
        
        payload = json.dumps({
            "inputs": {},
            "query": query,
            "user": "testAPI"
        })
        
        headers = {
            'Authorization': f'Bearer {agent_token}',
            'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
            'Content-Type': 'application/json',
            'Accept': '*/*',
            'Host': '127.0.0.1',
            'Connection': 'keep-alive'
        }
        
        try:
            logging.info(f"发送请求: {query[:50]}...")
            conn.request("POST", "/v1/chat-messages", payload, headers)
            res = conn.getresponse()
            
            status = res.status
            logging.info(f"响应状态码: {status}")
            
            if status != 200:
                logging.error(f"请求失败，状态码: {status}")
                return None
                
            data = res.read()
            response_text = data.decode("utf-8")
            logging.debug(f"收到响应: {response_text[:100]}...")
            return response_text
        except Exception as e:
            logging.error(f"发送请求时出错: {e}")
            return None
        finally:
            conn.close()
    
    def analyze_results(self, output_file):
        """
        调用分析脚本分析测试结果
        """
        try:
            self.log("开始分析测试结果...")
            
            # 检查分析脚本是否存在
            analyze_script = "analyze_similarity.py"
            if not os.path.exists(analyze_script):
                self.log(f"错误: 找不到分析脚本 {analyze_script}")
                return
            
            # 使用子进程调用分析脚本
            import subprocess
            process = subprocess.Popen(
                [sys.executable, analyze_script, output_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # 读取输出
            stdout, stderr = process.communicate()
            
            if process.returncode == 0:
                self.log("测试结果分析完成")
                
                # 提取分析结果中的关键信息
                import re
                
                # 提取合格率
                qualified_match = re.search(r'合格率: (\d+\.\d+)%', stdout)
                if qualified_match:
                    qualified_rate = qualified_match.group(1)
                    self.log(f"合格率: {qualified_rate}%")
                
                # 提取HTML报告路径
                html_match = re.search(r'正在生成HTML报告: (.+\.html)', stdout)
                if html_match:
                    html_path = html_match.group(1)
                    self.log(f"HTML报告已生成: {html_path}")
                    
                    # 询问是否打开HTML报告
                    if messagebox.askyesno("分析完成", "测试结果分析已完成，是否打开HTML报告?"):
                        import webbrowser
                        webbrowser.open(f"file://{os.path.abspath(html_path)}")
            else:
                self.log(f"分析过程中出错: {stderr}")
                
        except Exception as e:
            self.log(f"调用分析脚本时出错: {e}")
    
    def stop_test(self):
        if self.test_thread and self.test_thread.is_alive():
            self.log("正在停止测试...")
            self.stop_event.set()
            self.stop_button.config(state=tk.DISABLED)
    
    def reset_buttons(self):
        self.run_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_var.set("就绪")

    def analyze_existing_results(self):
        """
        分析已有的测试结果文件
        """
        file_path = filedialog.askopenfilename(
            title="选择测试结果文件",
            filetypes=[("Excel文件", "*.xlsx"), ("所有文件", "*.*")],
            initialdir=self.output_dir
        )
        
        if file_path:
            self.analyze_results(file_path)


class LogHandler(logging.Handler):
    def __init__(self, text_widget):
        logging.Handler.__init__(self)
        self.text_widget = text_widget
    
    def emit(self, record):
        msg = self.format(record)
        def append():
            self.text_widget.insert(tk.END, msg + '\n')
            self.text_widget.see(tk.END)
        # 在主线程中更新UI
        self.text_widget.after(0, append)


if __name__ == "__main__":
    root = tk.Tk()
    app = AutoTestGUI(root)
    root.mainloop() 