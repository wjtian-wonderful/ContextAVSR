import openai
import base64
import os
import json
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from threading import Semaphore
import itertools
import random


base_url =
api_version = 
ak = 
model_name = "gemini-2.5-pro-preview-06-05"

max_tokens = 8000
client = openai.AzureOpenAI(
    azure_endpoint=base_url,
    api_version=api_version,
    api_key=ak,
)



# ==============================================================================
# 辅助类：限速器 QPMLimiter（线程安全）
# ==============================================================================
class QPMLimiter:
    def __init__(self, qpm: int):
        self.interval = 60.0 / qpm
        self.lock = Semaphore(1)
        self.last_time = time.time()

    def wait(self):
        with self.lock:
            now = time.time()
            elapsed = now - self.last_time
            if elapsed < self.interval:
                time.sleep(self.interval - elapsed)
            self.last_time = time.time()

# 初始化一个全局限速器（稍后注入 QPM 值）
qpm_limiter = None

# ==============================================================================
# 辅助函数：检查是否是有效 JSON 文件 
# ==============================================================================
def is_valid_json_file(path: str) -> bool:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return bool(data)
    except:
        return False

# ==============================================================================
# 读取 Prompt
# ==============================================================================
def load_prompt_from_file(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

# ==============================================================================
# 【新增】辅助函数：从响应中稳健地提取和清理 JSON
# ==============================================================================
def extract_and_clean_json(response_content: str) -> (str | None, str | None):
    """
    从原始响应文本中提取并清理 JSON 字符串。
    
    Args:
        response_content (str): 大模型返回的完整文本。

    Returns:
        tuple[str | None, str | None]:
        - 第一个元素是清理后的 JSON 字符串，如果提取失败则为 None。
        - 第二个元素是原始提取但未清理的 JSON 字符串，用于调试，提取失败则为 None。
    """
    json_str = None
    
    # 策略 1: 优先匹配 Markdown 代码块 ```json ... ```
    match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', response_content, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        # 策略 2: 备用方案，匹配裸露的 JSON 对象 { ... }
        match = re.search(r'^\s*(\{.*\})', response_content, re.DOTALL)
        if match:
            json_str = match.group(1)

    if not json_str:
        return None, None # 无法从响应中提取任何 JSON 内容

    original_extracted_str = json_str # 保存一份原始提取的副本
    
    # 清理步骤：移除 } 或 ] 前的拖尾逗号，这是最常见的 LLM JSON 错误
    cleaned_json_str = re.sub(r',\s*([\}\]])', r'\1', json_str)
    
    return cleaned_json_str, original_extracted_str


def detect_language(text: str) -> str:
    """
    检测文本的语言类型（中文或英文）。

    Args:
        text (str): 输入的文本字符串。

    Returns:
        str: 检测到的语言类型，'zh' 表示中文，'en' 表示英文。
    """
    # 简单的基于中文字符的检测 如果包含中文字符则判断为中文
    if any('\u4e00' <= char <= '\u9fff' for char in text):
        return 'zh'
    else:
        return 'en'
# ==============================================================================
# 【优化】分析单个视频 (集成了修复逻辑)
# ==============================================================================
def analyze_video(item, output_json_path: str, key: str, prompt_text: str):

    try:

        try:
            if 'video_path' in item.keys():
                video_path = item['video_path']
            else :
                video_path = item['video']
            with open(video_path, "rb") as video_file:
                video_data = video_file.read()
            encoded_string = base64.b64encode(video_data).decode('utf-8')

        except Exception as e:
            return "ERROR", f"Key: {key} - 文件读取失败: {e}"
        if 'lang' not in item.keys():
            lang = detect_language(item['text'])
        else:
            lang = item['lang']
        lang_prompt = '使用中文回复。' if lang =='zh' else 'use english to answer.'

        qpm_limiter.wait()

        prompt_text  = f'''
# Role
你是一个高精度的**多模态语音转写专家 (Multimodal ASR)**。
你的核心能力是结合**宏观场景理解**与**微观视觉线索**，将语音流解码为精准的文本。

# Inputs
1. **[视频内容]**:
<视频场景描述>:{item['video_desc']}
<视频字幕>:{str(item['subtitles'])}
<视频场景内OCR>:{item['scene_text_description']}
<说话场景>:{item['speaking_scenario']}

2. **<音频文件>**

# Task
请对 音视频 进行多模态转写，输出 JSON 对象包含 "think_process"（推理链）和 "transcription"。

# Constraints & Strategy
请严格遵守 **[视觉建模 -> 声学感知 -> 多模态融合]** 的解码流程：

**第一步：视觉环境与实体构建 (Visual Context & Entity Anchoring)**
你需要从两个维度建立当前的“语言模型概率分布”：
1.  **宏观场景 (Macro Context)**：
    - 分析环境（如：技术发布会、厨房、游戏直播、嘈杂街道）。
    - 分析行为（如：正在写代码、正在炒菜、正在打怪）。
    - **作用**：确定**领域词汇表**（例如：编程场景下，"Java"的概率 >> "加哇"）。
2.  **微观实体 (Micro Entities)**：
    - 提取屏幕上的OCR文字、PPT标题、人名条、特定物体。
    - **作用**：建立**专有名词热词表**（Hotwords）。

**第二步：纯声学感知 (Pure Acoustic Perception)**
- **严格禁止直接输出语义文本**。
- **任务**：仅客观记录听觉特征。
  - 中文：记录**拼音**（含声调，如 /yún yuán shēng/）。
  - 英文/代码/术语：记录**音素**或**拟声拼写**（如 /bi-li-bi-li/）。
  - 噪声/模糊：标记为 `[noise]` 或 `[unclear]`。

**第三步：多模态解码与消歧 (Multimodal Decoding & Fusion)**
- **核心任务**：将“声学序列”映射为“文本”，并解释通过视觉解决了什么歧义。
- **推理逻辑**：
  1.  **场景对齐**：根据宏观场景，确定同音词的领域归属。
      - *例：听到 /qie-tu/ + 场景[前端开发] -> "切图"；场景[地理] -> "切土"。*
  2.  **实体锚定**：根据OCR/字幕，修正模糊发音或专有名词。
      - *例：听到 /da-mo/ + OCR[达摩院] -> "达摩"；OCR[打磨工艺] -> "打磨"。*
  3.  **结果生成**：输出最终文本。

# Output Format
请输出且仅输出一个合法的 JSON 对象：
{{
  "think_process": "1. [视觉线索] 当前处于...场景, 出现了... 2. [声学分析] 0-2秒听到 /wǒ-men-di-fain-yi-ge/; 2-4秒听到模糊的 /chuan-si-fao-mo/。 3. [多模态消歧] /di-fain/ 结合编程场景解码为 'define'; /chuan-si-fao-mo/ 结合OCR字幕....修正为 'Transformer'...",
  "transcription": "最终确定的转写文本"
}}

{lang_prompt}
'''
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": encoded_string}}
                ]}
            ],
            max_tokens=max_tokens,
        )

        response_content = completion.choices[0].message.content
        print('response_content', response_content)

        # --- 核心优化点：集成提取、清理和解析逻辑 ---
        item['gemini_construct_response'] = response_content
        
        with open(output_json_path, 'w', encoding='utf-8') as f:
            # json.dump(item, f, ensure_ascii=False, indent=4)
            json.dump(item, f, ensure_ascii=False)
            return "SUCCESS", key

    except Exception as e:
        error_msg = f"Key: {key} - API 调用失败: {e}"
        print(error_msg)
        return "ERROR", error_msg

# ==============================================================================
# 【修改】主任务函数（并行处理）
# ==============================================================================
def process_jsonl_file_parallel(jsonl_path: str, base_output_dir: str, prompt_text: str, qpm: int, max_lines: int = None, shuffle: bool = False):
    global qpm_limiter
    qpm_limiter = QPMLimiter(qpm)

    jsonl_filename = os.path.basename(jsonl_path)
    output_dir_name = os.path.splitext(jsonl_filename)[0]
    final_output_dir = os.path.join(base_output_dir, output_dir_name)

    print(f'final_output_dir: {final_output_dir}')
    os.makedirs(final_output_dir, exist_ok=True)

    tasks = []
    skipped_count = 0
    processed_count = 0

    print("正在读取文件...")
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    if shuffle:
        print("正在打乱文件行顺序...")
        random.shuffle(lines)
        print("打乱完成。")

    lines_iterator = lines
    if max_lines is not None and len(lines) > max_lines:
        print(f"将从打乱后的数据中处理前 {max_lines} 行。")
        lines_iterator = lines[:max_lines]

    print('get handing list')
    for line in tqdm(lines_iterator):
        line = line.strip()
        if not line: continue
        try:
                data = json.loads(line)

                if data['filter_result'] != 'Keep':
                    continue

                # 检查 'key' 是否存在，如果不存在则从 video_path 生成
                if "data_id" in data.keys():
                    key = data['data_id']
                elif 'id' in data.keys():
                    key = data['id']
                elif 'key' in data.keys():
                    key = data['key']
                else:
                    tqdm.write(f"跳过无 'data_id', 'id', 'key' 字段的行: {data}")
                    continue

                

                output_json_path = os.path.join(final_output_dir, f"{key}.json")
                if os.path.exists(output_json_path) and is_valid_json_file(output_json_path):
                    skipped_count += 1
                    continue

                tasks.append((data, output_json_path, key, prompt_text))

        except Exception as e:
            tqdm.write(f"跳过无效行: {e}")

    print(f"\n已跳过 {skipped_count} 个已处理任务。待处理任务数: {len(tasks)}")

    if not tasks:
        print("没有需要处理的任务。")
        return

    avg_video_time = 15
    max_workers = min(int((qpm / 60) * avg_video_time), 32)

    print(f"并行处理任务中（线程数: {max_workers}）...\n")

    success_count = 0
    error_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_key = {
            executor.submit(analyze_video, *task): task[2] for task in tasks
        }

        for future in tqdm(as_completed(future_to_key), total=len(tasks), desc="处理进度"):
            key = future_to_key[future]
            try:
                status, msg = future.result()
                if status == "SUCCESS":
                    success_count += 1
                else:
                    error_count += 1
                    tqdm.write(f"[失败] {msg}")
            except Exception as e:
                error_count += 1
                tqdm.write(f"[异常] {key} 执行失败: {e}")

    print(f"\n✅ 任务完成: 成功 {success_count}, 失败 {error_count}")

# ==============================================================================
# 主程序入口
# ==============================================================================

if __name__ == '__main__':
    # --- 参数配置 ---
    prompt_to_use = "1"
                                                                                         

    INPUT_JSONL_FILES = [

    ]
                                                                                                                                                                                                                            
    BASE_OUTPUT_DIR = 
    QPM =170

    # 【新增】最大处理行数 (设置为 None 则处理所有行)
    MAX_LINES = None

    # 【新增】是否在处理前打乱顺序
    SHUFFLE_BEFORE_PROCESSING = True

    print("\n🚀 开始处理视频任务...\n")
    start_time = time.time()

    for INPUT_JSONL_FILE in INPUT_JSONL_FILES:
        print(f"--- handling file: {INPUT_JSONL_FILE} ---")
        process_jsonl_file_parallel(
            jsonl_path=INPUT_JSONL_FILE,
            base_output_dir=BASE_OUTPUT_DIR,
            prompt_text=prompt_to_use,
            qpm=QPM,
            max_lines=MAX_LINES,
            shuffle=SHUFFLE_BEFORE_PROCESSING
        )

    print(f"\n⏱️ 总耗时: {(time.time() - start_time):.2f} 秒")
