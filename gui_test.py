import tkinter as tk
from tkinter import filedialog, ttk, messagebox, simpledialog
import os
import threading
import pandas as pd
import sys
import logging
import http.client
import json
import time
import re
import numpy as np
from collections import Counter
import math
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from datetime import datetime
import jieba

# 设置标准输出编码为UTF-8
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 尝试导入transformers库，用于BERT模型
try:
    from transformers import BertTokenizer, BertModel
    import torch
    from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
    BERT_AVAILABLE = True
    print("[成功] BERT模型可用，将使用深度学习方法进行相似度评估")
except ImportError:
    BERT_AVAILABLE = False
    print("[警告] 未安装transformers或torch库，将使用传统方法进行相似度评估")
    print("  如需使用BERT模型，请安装必要的库: pip install transformers torch scikit-learn")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("test_log.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# 设置jieba日志级别，减少不必要的输出
jieba.setLogLevel(logging.WARNING)

# 初始化BERT模型（如果可用）
if BERT_AVAILABLE:
    try:
        print("\n[加载] 正在加载BERT模型...")
        model_load_start = time.time()
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        model = BertModel.from_pretrained('bert-base-chinese')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()  # 设置为评估模式
        model_load_time = time.time() - model_load_start
        print(f"[成功] BERT模型加载完成，使用设备: {device}，耗时 {model_load_time:.2f} 秒")
    except Exception as e:
        BERT_AVAILABLE = False
        print(f"[错误] 加载BERT模型失败: {e}")
        print("  将使用传统方法进行相似度评估")

def get_bert_embedding(text):
    if pd.isna(text) or not text:
        return None
    
    # 转为字符串并截断（BERT有最大长度限制）
    text = str(text)[:512]
    
    # 对文本进行编码
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 获取BERT输出
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 使用[CLS]标记的输出作为整个序列的表示
    embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    return embeddings[0]

def bert_similarity(text1, text2):
    if pd.isna(text1) or pd.isna(text2) or not text1 or not text2:
        return 0.0
    
    # 获取文本嵌入
    embedding1 = get_bert_embedding(text1)
    embedding2 = get_bert_embedding(text2)
    
    if embedding1 is None or embedding2 is None:
        return 0.0
    
    # 计算余弦相似度
    embedding1 = embedding1.reshape(1, -1)
    embedding2 = embedding2.reshape(1, -1)
    similarity = sklearn_cosine_similarity(embedding1, embedding2)[0][0]
    
    return similarity

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

# 计算传统余弦相似度
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
def evaluate_similarity(expected, actual, log_function=None):
    # 计算语义相似度
    if pd.isna(expected) or pd.isna(actual):
        if log_function:
            log_function("数据缺失")
        return "数据缺失"
    
    if actual == "请求失败":
        if log_function:
            log_function("请求失败")
        return "请求失败"
    
    # 文本预处理
    expected_words = preprocess_text(expected)
    actual_words = preprocess_text(actual)
    
    if not expected_words or not actual_words:
        if log_function:
            log_function("文本为空")
        return "文本为空"
    
    # 尝试使用BERT模型（如果可用）
    start_time = time.time()
    if BERT_AVAILABLE:
        try:
            bert_sim = bert_similarity(expected, actual)
            bert_time = time.time() - start_time
            if log_function:
                log_function(f"使用BERT模型，相似度 {bert_sim:.4f}，耗时 {bert_time:.2f}秒")
            similarity = bert_sim
            method = "BERT模型"
        except Exception as e:
            if log_function:
                log_function(f"BERT模型计算失败: {e}，切换到传统方法")
            # 如果BERT失败，使用传统方法
            cos_sim = cosine_similarity(expected_words, actual_words)
            jac_sim = jaccard_similarity(expected_words, actual_words)
            similarity = 0.7 * cos_sim + 0.3 * jac_sim
            method = "传统方法"
            trad_time = time.time() - start_time
            if log_function:
                log_function(f"使用{method}，相似度 {similarity:.4f}，耗时 {trad_time:.2f}秒")
    else:
        # 使用传统方法
        cos_sim = cosine_similarity(expected_words, actual_words)
        jac_sim = jaccard_similarity(expected_words, actual_words)
        similarity = 0.7 * cos_sim + 0.3 * jac_sim
        method = "传统方法"
        trad_time = time.time() - start_time
        if log_function:
            log_function(f"使用{method}，相似度 {similarity:.4f}，耗时 {trad_time:.2f}秒")
    
    # 根据相似度评估效果
    if similarity >= 0.9:
        result = f"优秀 ({similarity:.2f}): 语义几乎完全一致 [{method}]"
    elif similarity >= 0.8:
        result = f"良好 ({similarity:.2f}): 语义非常接近 [{method}]"
    elif similarity >= 0.7:
        result = f"较好 ({similarity:.2f}): 语义基本相符 [{method}]"
    elif similarity >= 0.6:
        result = f"一般 ({similarity:.2f}): 语义有一定差异 [{method}]"
    elif similarity >= 0.5:
        result = f"较差 ({similarity:.2f}): 语义存在明显差异 [{method}]"
    else:
        result = f"差距较大 ({similarity:.2f}): 语义相差很大 [{method}]"
    
    return result

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
        
        # 创建分析结果目录
        self.analysis_dir = "analysis_results"
        if not os.path.exists(self.analysis_dir):
            os.makedirs(self.analysis_dir)
        
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
        
        # 线程数量设置
        threads_frame = ttk.Frame(advanced_frame)
        threads_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(threads_frame, text="并发线程数:").pack(side=tk.LEFT, padx=5)
        self.threads_var = tk.StringVar(value="3")
        threads_entry = ttk.Entry(threads_frame, textvariable=self.threads_var, width=10)
        threads_entry.pack(side=tk.LEFT, padx=5)
        threads_info = ttk.Label(threads_frame, text="(建议不超过10，避免请求过于频繁)")
        threads_info.pack(side=tk.LEFT, padx=5)
        
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
        
        self.analyze_button = ttk.Button(button_frame, text="分析结果", command=self.analyze_existing_results)
        self.analyze_button.pack(side=tk.LEFT, padx=5)
        
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
            if "测试问题" not in df.columns:
                self.log("错误: 输入文件中没有'测试问题'列")
                return
            
            # 添加"真实测试结果"列
            df["真实测试结果"] = ""
            
            # 获取线程数
            try:
                num_threads = int(self.threads_var.get())
                if num_threads <= 0:
                    raise ValueError("线程数必须大于0")
            except ValueError:
                self.log("错误: 无效的线程数，将使用默认值3")
                num_threads = 3
            
            # 创建线程池
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            # 遍历每个问题
            total = len(df)
            success_count = 0
            fail_count = 0
            
            self.log(f"共有 {total} 个测试用例，使用 {num_threads} 个线程处理")
            
            # 创建任务列表
            tasks = []
            for i, row in df.iterrows():
                tasks.append((i, row["测试问题"]))
            
            # 使用线程池处理请求
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                # 提交所有任务
                future_to_idx = {
                    executor.submit(self.custom_send_request, query, agent_token): (idx, query)
                    for idx, query in tasks
                }
                
                # 处理完成的任务
                for future in as_completed(future_to_idx):
                    idx, query = future_to_idx[future]
                    
                    # 检查是否需要停止
                    if self.stop_event.is_set():
                        self.log("测试已手动停止")
                        executor.shutdown(wait=False)
                        break
                    
                    try:
                        response_text = future.result()
                        if response_text:
                            # 提取答案
                            answer = extract_answer(response_text)
                            df.at[idx, "真实测试结果"] = answer
                            success_count += 1
                            self.log(f"完成问题 {idx+1}/{total}: {query[:50]}...")
                        else:
                            df.at[idx, "真实测试结果"] = "请求失败"
                            fail_count += 1
                            self.log(f"问题 {idx+1}/{total} 请求失败")
                        
                        # 每处理一批请求后暂停一下，避免请求过于频繁
                        time.sleep(interval)
                        
                    except Exception as e:
                        df.at[idx, "真实测试结果"] = f"处理出错: {str(e)}"
                        fail_count += 1
                        self.log(f"处理问题 {idx+1} 时出错: {e}")
            
            # 保存结果
            df.to_excel(output_file, index=False)
            self.log(f"测试完成，结果已保存至: {output_file}")
            self.log(f"测试结果: 总计 {total} 个测试用例，通过 {success_count} 个，失败 {fail_count} 个")
            
            # 询问用户是否要进行分析
            self.root.after(0, lambda: self.ask_for_analysis(output_file))
            
        except Exception as e:
            self.log(f"运行测试时出错: {e}")
        finally:
            # 恢复按钮状态
            self.root.after(0, self.reset_buttons)
    
    def ask_for_analysis(self, output_file):
        """
        询问用户是否要分析测试结果
        """
        if messagebox.askyesno("测试完成", "测试已完成，是否要分析测试结果?"):
            self.analyze_results(output_file)
    
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
        分析测试结果
        """
        try:
            self.log("开始分析测试结果...")
            
            # 询问用户是否使用智能评估
            if messagebox.askyesno("选择评估方式", "是否使用智能评估系统进行评估？\n选择'是'使用智能评估，选择'否'使用传统相似度评估"):
                self.smart_evaluate_results(output_file)
            else:
                # 使用内置相似度分析实现
                self.similarity_analyze_results(output_file)
                
        except Exception as e:
            self.log(f"分析结果时出错: {e}")
    
    def similarity_analyze_results(self, input_file):
        """
        使用内置相似度分析算法分析测试结果
        """
        try:
            self.log("[开始] 开始计算语义相似度...")
            self.status_var.set("正在分析中...")
            
            # 禁用运行和分析按钮
            self.run_button.config(state=tk.DISABLED)
            self.analyze_button.config(state=tk.DISABLED)
            
            # 创建进度条窗口
            self.progress_window = tk.Toplevel(self.root)
            self.progress_window.title("分析进度")
            self.progress_window.geometry("400x150")
            self.progress_window.resizable(False, False)
            self.progress_window.transient(self.root)
            self.progress_window.grab_set()
            
            # 添加进度条
            ttk.Label(self.progress_window, text="正在进行相似度分析，请稍候...", font=("Arial", 10)).pack(pady=10)
            self.progress_var = tk.DoubleVar()
            self.progress_bar = ttk.Progressbar(self.progress_window, variable=self.progress_var, maximum=100, length=350)
            self.progress_bar.pack(pady=10)
            
            # 添加进度文本
            self.progress_text = tk.StringVar(value="准备中...")
            ttk.Label(self.progress_window, textvariable=self.progress_text).pack(pady=5)
            
            # 添加取消按钮
            self.analysis_stop_event = threading.Event()
            ttk.Button(self.progress_window, text="取消分析", 
                      command=lambda: self.analysis_stop_event.set()).pack(pady=10)
            
            # 创建并启动分析线程
            self.analysis_thread = threading.Thread(
                target=self._run_similarity_analysis,
                args=(input_file,)
            )
            self.analysis_thread.daemon = True
            self.analysis_thread.start()
            
        except Exception as e:
            self.log(f"启动分析时出错: {e}")
            self._close_progress_window()
            self.reset_buttons()
    
    def _run_similarity_analysis(self, input_file):
        """
        在独立线程中运行相似度分析
        """
        try:
            # 读取Excel文件
            start_time = time.time()
            self.root.after(0, lambda: self.log(f"[读取] 正在读取文件: {input_file}"))
            df = pd.read_excel(input_file)
            total_rows = len(df)
            self.root.after(0, lambda: self.log(f"[完成] 文件读取完成，共 {total_rows} 行数据"))
            
            # 获取列名
            columns = df.columns.tolist()
            if len(columns) < 3:
                self.root.after(0, lambda: self.log("[错误] Excel文件至少需要三列数据（问题、预期结果和真实结果）"))
                self.root.after(0, self._close_progress_window)
                self.root.after(0, self.reset_buttons)
                return
            
            # 第二列是预期结果，第三列是真实结果
            expected_col = columns[1]  # 第二列
            actual_col = columns[2]    # 第三列
            
            self.root.after(0, lambda: self.log(f"[信息] 数据列信息:"))
            self.root.after(0, lambda: self.log(f"  - 预期结果列: {expected_col}"))
            self.root.after(0, lambda: self.log(f"  - 真实结果列: {actual_col}"))
            
            # 逐行处理，显示进度
            results = []
            for i, row in df.iterrows():
                # 检查是否取消
                if self.analysis_stop_event.is_set():
                    self.root.after(0, lambda: self.log("[取消] 分析已取消"))
                    break
                
                # 更新进度
                progress = (i + 1) / total_rows * 100
                self.root.after(0, lambda p=progress: self.progress_var.set(p))
                self.root.after(0, lambda idx=i, tot=total_rows: 
                               self.progress_text.set(f"分析中: {idx+1}/{tot} ({progress:.1f}%)"))
                
                if (i + 1) % 5 == 0 or i == 0 or i == total_rows - 1:
                    self.root.after(0, lambda idx=i, tot=total_rows, prog=progress: 
                                   self.log(f"\n进度: {idx+1}/{tot} ({prog:.1f}%)"))
                
                # 评估相似度，使用自定义日志记录函数
                log_func = lambda msg, idx=i: self.root.after(0, lambda: self.log(f"  样本 #{idx+1}: {msg}"))
                result = evaluate_similarity(row[expected_col], row[actual_col], log_func)
                results.append(result)
                
                # 每10个样本显示一次时间估计
                if (i + 1) % 10 == 0:
                    elapsed = time.time() - start_time
                    estimated_total = elapsed / (i + 1) * total_rows
                    remaining = estimated_total - elapsed
                    self.root.after(0, lambda e=elapsed, r=remaining: 
                                   self.log(f"已用时间: {e:.1f}秒, 预计剩余: {r:.1f}秒"))
            
            # 将结果添加到DataFrame
            df['效果评估'] = results
            
            # 计算分析统计
            if not self.analysis_stop_event.is_set():
                self._calculate_analysis_statistics(df, start_time, input_file)
            
        except Exception as e:
            self.root.after(0, lambda err=e: self.log(f"[错误] 分析过程中出错: {err}"))
        finally:
            # 恢复按钮状态
            self.root.after(0, self.reset_buttons)
            self.root.after(0, lambda: self.status_var.set("就绪"))
            self.root.after(0, self._close_progress_window)
    
    def _calculate_analysis_statistics(self, df, start_time, input_file):
        """
        计算分析统计数据并生成报告
        """
        try:
            total_time = time.time() - start_time
            total_rows = len(df)
            
            self.root.after(0, lambda: self.log(f"\n[完成] 相似度计算完成，总耗时: {total_time:.2f}秒，平均每个样本: {total_time/total_rows:.2f}秒"))
            
            # 计算整体评估统计
            self.root.after(0, lambda: self.log("\n[统计] 整体评估统计:"))
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
            
            self.root.after(0, lambda: self.log(f"总样本数: {total_rows}"))
            self.root.after(0, lambda: self.log(f"有效样本数: {valid_samples}"))
            self.root.after(0, lambda: self.log(f"优秀: {excellent} ({excellent/total_rows*100:.1f}% 总体, {valid_excellent:.1f}% 有效)"))
            self.root.after(0, lambda: self.log(f"良好: {good} ({good/total_rows*100:.1f}% 总体, {valid_good:.1f}% 有效)"))
            self.root.after(0, lambda: self.log(f"较好: {fair} ({fair/total_rows*100:.1f}% 总体, {valid_fair:.1f}% 有效)"))
            self.root.after(0, lambda: self.log(f"一般: {average} ({average/total_rows*100:.1f}% 总体)"))
            self.root.after(0, lambda: self.log(f"较差: {poor} ({poor/total_rows*100:.1f}% 总体)"))
            self.root.after(0, lambda: self.log(f"差距较大: {bad} ({bad/total_rows*100:.1f}% 总体)"))
            self.root.after(0, lambda: self.log(f"请求失败: {failed} ({failed/total_rows*100:.1f}% 总体)"))
            self.root.after(0, lambda: self.log(f"数据缺失或为空: {missing} ({missing/total_rows*100:.1f}% 总体)"))
            
            # 计算合格率（优秀+良好+较好）
            qualified_rate = (excellent + good + fair) / valid_samples * 100 if valid_samples > 0 else 0
            self.root.after(0, lambda: self.log(f"\n[结果] 合格率: {qualified_rate:.1f}%"))
            
            # 统计使用的方法
            bert_count = len(df[df['效果评估'].str.contains('BERT模型', na=False)])
            traditional_count = len(df[df['效果评估'].str.contains('传统方法', na=False)])
            self.root.after(0, lambda: self.log(f"\n[方法] 评估方法统计:"))
            self.root.after(0, lambda: self.log(f"BERT模型: {bert_count} 次 ({bert_count/valid_samples*100:.1f}% 有效样本)"))
            self.root.after(0, lambda: self.log(f"传统方法: {traditional_count} 次 ({traditional_count/valid_samples*100:.1f}% 有效样本)"))
            
            # 生成可视化图表和报告
            self._generate_analysis_reports(df, input_file, total_time, total_rows,
                                         excellent, good, fair, average, poor, bad, failed, missing,
                                         valid_samples, valid_excellent, valid_good, valid_fair, qualified_rate,
                                         bert_count, traditional_count)
            
        except Exception as e:
            self.root.after(0, lambda err=e: self.log(f"[错误] 计算统计时出错: {err}"))
    
    def _generate_analysis_reports(self, df, input_file, total_time, total_rows, 
                                excellent, good, fair, average, poor, bad, failed, missing,
                                valid_samples, valid_excellent, valid_good, valid_fair, qualified_rate,
                                bert_count, traditional_count):
        """
        生成分析报告和可视化图表
        """
        try:
            # 生成输出文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.analysis_dir, f"analysis_result_{timestamp}.xlsx")
            
            # 保存Excel结果
            self.root.after(0, lambda: self.log(f"[保存] 正在保存Excel结果: {output_path}"))
            df.to_excel(output_path, index=False)
            
            # 总结
            self.root.after(0, lambda: self.log("\n" + "=" * 80))
            self.root.after(0, lambda: self.log(f"[完成] 处理完成！总耗时: {total_time:.2f}秒"))
            self.root.after(0, lambda: self.log(f"[结果] 合格率: {qualified_rate:.1f}%"))
            self.root.after(0, lambda: self.log(f"[文件] 结果文件: {output_path}"))
            self.root.after(0, lambda: self.log("=" * 80))
            
            # 询问是否打开Excel文件
            self.root.after(0, lambda out=output_path: self._ask_open_result_file(out))
            
        except Exception as e:
            self.root.after(0, lambda err=e: self.log(f"[错误] 生成报告时出错: {err}"))
    
    def _ask_open_result_file(self, output_file):
        """
        询问是否打开结果文件
        """
        if messagebox.askyesno("评估完成", "智能评估已完成，是否打开结果文件?"):
            import os
            os.startfile(output_file)
    
    def _close_progress_window(self):
        """
        关闭进度窗口
        """
        if hasattr(self, 'progress_window') and self.progress_window:
            try:
                self.progress_window.destroy()
            except:
                pass
    
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
    
    def smart_evaluate_results(self, output_file):
        """
        使用智能评估系统评估测试结果
        """
        # 设置智能评估的API Token
        eval_token = "app-emvL5JV4kaHFZe4XnPRLfVFr"  # 默认评估智能体Token
        
        # 询问用户是否要使用自定义Token
        if messagebox.askyesno("智能评估设置", "是否使用自定义评估智能体Token？"):
            custom_token = simpledialog.askstring("输入Token", "请输入评估智能体Token:")
            if custom_token and custom_token.strip():
                eval_token = custom_token.strip()
        
        # 创建并启动评估线程
        self.log("开始智能评估测试结果...")
        self.status_var.set("正在进行智能评估...")
        
        # 禁用按钮，防止重复操作
        self.run_button.config(state=tk.DISABLED)
        
        # 创建进度条窗口
        self.progress_window = tk.Toplevel(self.root)
        self.progress_window.title("智能评估进度")
        self.progress_window.geometry("400x150")
        self.progress_window.resizable(False, False)
        self.progress_window.transient(self.root)
        self.progress_window.grab_set()
        
        # 添加进度条
        ttk.Label(self.progress_window, text="正在进行智能评估，请稍候...", font=("Arial", 10)).pack(pady=10)
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.progress_window, variable=self.progress_var, maximum=100, length=350)
        self.progress_bar.pack(pady=10)
        
        # 添加进度文本
        self.progress_text = tk.StringVar(value="准备中...")
        ttk.Label(self.progress_window, textvariable=self.progress_text).pack(pady=5)
        
        # 添加取消按钮
        self.eval_stop_event = threading.Event()
        ttk.Button(self.progress_window, text="取消评估", command=self._cancel_evaluation).pack(pady=10)
        
        # 创建并启动线程
        self.eval_thread = threading.Thread(
            target=self._run_smart_evaluation,
            args=(output_file, eval_token)
        )
        self.eval_thread.daemon = True
        self.eval_thread.start()
    
    def _cancel_evaluation(self):
        """
        取消正在进行的评估
        """
        if self.eval_thread and self.eval_thread.is_alive():
            self.log("正在取消智能评估...")
            self.eval_stop_event.set()
            self.progress_text.set("正在取消...")
    
    def _run_smart_evaluation(self, output_file, eval_token):
        """
        在单独的线程中运行智能评估过程
        """
        try:
            # 读取Excel文件
            df = pd.read_excel(output_file)
            
            # 确保有必要的列
            required_cols = ["测试问题", "测试答案", "真实测试结果"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                self.root.after(0, lambda: self.log(f"错误: 输入文件缺少必要的列: {', '.join(missing_cols)}"))
                self.root.after(0, self._close_progress_window)
                self.root.after(0, self.reset_buttons)
                return
            
            # 添加评估结果列
            df["智能评分"] = ""
            # 移除评估详情列
            # df["评估详情"] = ""
            
            # 遍历每个测试用例进行评估
            total = len(df)
            success_count = 0
            fail_count = 0
            
            self.root.after(0, lambda: self.log(f"共有 {total} 个测试用例需要评估"))
            
            for i, row in df.iterrows():
                # 检查是否需要停止
                if self.eval_stop_event.is_set():
                    self.root.after(0, lambda: self.log("智能评估已手动取消"))
                    break
                
                # 更新进度条
                progress = int((i+1) / total * 100)
                self.root.after(0, lambda p=progress: self.progress_var.set(p))
                self.root.after(0, lambda idx=i, tot=total: self.progress_text.set(f"正在评估 {idx+1}/{tot}"))
                self.root.after(0, lambda p=progress: self.status_var.set(f"智能评估进行中... {p}%"))
                
                # 构建评估请求
                context = row["测试答案"]
                query = row["真实测试结果"]
                
                if pd.isna(context) or pd.isna(query) or query == "请求失败":
                    df.at[i, "智能评分"] = "N/A"
                    # 不再设置评估详情
                    # df.at[i, "评估详情"] = "无法评估" if query != "请求失败" else "请求失败"
                    fail_count += 1
                    continue
                
                # 构建评估提示
                eval_prompt = f"测试答案：\n{context}\n\n真实测试结果：\n{query}"
                
                # 在主线程中更新日志
                self.root.after(0, lambda idx=i, tot=total: self.log(f"评估问题 {idx+1}/{tot}..."))
                
                # 发送评估请求
                response_text = self.send_eval_request(eval_prompt, eval_token)
                
                if response_text:
                    # 提取评估结果
                    try:
                        response_json = json.loads(response_text)
                        answer = response_json.get("answer", "")
                        
                        # 从answer中提取JSON部分
                        import re
                        json_match = re.search(r'```json\s*(.*?)\s*```', answer, re.DOTALL)
                        
                        if json_match:
                            result_json = json.loads(json_match.group(1))
                            score = result_json.get("score", "")
                            
                            df.at[i, "智能评分"] = score
                            # 不再设置评估详情
                            # df.at[i, "评估详情"] = answer
                            success_count += 1
                        else:
                            # 尝试直接解析整个answer
                            try:
                                result_json = json.loads(answer)
                                score = result_json.get("score", "")
                                
                                df.at[i, "智能评分"] = score
                                # 不再设置评估详情
                                # df.at[i, "评估详情"] = answer
                                success_count += 1
                            except:
                                df.at[i, "智能评分"] = "解析失败"
                                # 不再设置评估详情
                                # df.at[i, "评估详情"] = "无法从响应中提取评估结果"
                                fail_count += 1
                    except Exception as e:
                        self.root.after(0, lambda err=e: self.log(f"解析评估结果时出错: {err}"))
                        df.at[i, "智能评分"] = "解析失败"
                        # 不再设置评估详情
                        # df.at[i, "评估详情"] = f"解析错误: {str(e)}"
                        fail_count += 1
                else:
                    df.at[i, "智能评分"] = "请求失败"
                    # 不再设置评估详情
                    # df.at[i, "评估详情"] = "评估请求失败"
                    fail_count += 1
            
            # 生成输出文件名
            output_dir = os.path.dirname(output_file)
            base_name = os.path.basename(output_file)
            name, ext = os.path.splitext(base_name)
            smart_output = os.path.join(output_dir, f"{name}_智能评估_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}{ext}")
            
            # 保存结果
            df.to_excel(smart_output, index=False)
            
            # 在主线程中更新UI
            self.root.after(0, lambda: self.log(f"智能评估完成，结果已保存至: {smart_output}"))
            self.root.after(0, lambda sc=success_count, fc=fail_count, tot=total: 
                           self.log(f"评估结果: 总计 {tot} 个测试用例，成功 {sc} 个，失败 {fc} 个"))
            
            # 计算平均分
            try:
                scores = pd.to_numeric(df["智能评分"], errors="coerce")
                avg_score = scores.mean()
                if not pd.isna(avg_score):
                    self.root.after(0, lambda avg=avg_score: self.log(f"平均评分: {avg:.2f}"))
            except Exception as e:
                self.root.after(0, lambda err=e: self.log(f"计算平均分时出错: {err}"))
            
            # 关闭进度窗口
            self.root.after(0, self._close_progress_window)
            
            # 询问是否打开结果文件
            self.root.after(0, lambda out=smart_output: self._ask_open_result_file(out))
                
        except Exception as e:
            self.root.after(0, lambda err=e: self.log(f"智能评估过程中出错: {err}"))
        finally:
            # 恢复按钮状态
            self.root.after(0, self.reset_buttons)
            self.root.after(0, lambda: self.status_var.set("就绪"))
            self.root.after(0, self._close_progress_window)
    
    def send_eval_request(self, prompt, token):
        """
        发送评估请求到智能评估系统
        """
        conn = http.client.HTTPConnection("127.0.0.1")
        
        payload = json.dumps({
            "inputs": {},
            "query": prompt,
            "user": "testAPI"
        })
        
        headers = {
            'Authorization': f'Bearer {token}',
            'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
            'Content-Type': 'application/json',
            'Accept': '*/*',
            'Host': '127.0.0.1',
            'Connection': 'keep-alive'
        }
        
        try:
            logging.info(f"发送评估请求...")
            conn.request("POST", "/v1/chat-messages", payload, headers)
            res = conn.getresponse()
            
            status = res.status
            logging.info(f"评估响应状态码: {status}")
            
            if status != 200:
                logging.error(f"评估请求失败，状态码: {status}")
                return None
                
            data = res.read()
            response_text = data.decode("utf-8")
            logging.debug(f"收到评估响应: {response_text[:100]}...")
            return response_text
        except Exception as e:
            logging.error(f"发送评估请求时出错: {e}")
            return None
        finally:
            conn.close()
    
    def stop_test(self):
        if self.test_thread and self.test_thread.is_alive():
            self.log("正在停止测试...")
            self.stop_event.set()
            self.stop_button.config(state=tk.DISABLED)
    
    def reset_buttons(self):
        self.run_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.analyze_button.config(state=tk.NORMAL)
        self.status_var.set("就绪")


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