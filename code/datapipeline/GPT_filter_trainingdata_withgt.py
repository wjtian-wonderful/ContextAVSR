import os
import json
import re
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai
from tqdm import tqdm

# Configuration: Use environment variables
MODEL_NAME = os.getenv("AZURE_OPENAI_MODEL_NAME", "gpt-4o-2024-08-06")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "")
API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")



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




PROMPT = '''
角色：你是一位资深的语音数据科学家，专门负责多模态 ASR（Video-conditioned ASR）的数据清洗与标注工作。

任务：根据两个 ASR 模型的粗略识别结果（Result A & B），和真实抄本(GroundTruth)，筛选出那些“单凭音频存在歧义，必须依赖视觉信息（如画面物体、环境、字幕 OCR）才能纠正”的高价值数据。

筛选逻辑：
1. Keep (保留)：
   - 同音/近音词冲突：结果存在同音歧义的情况，仅凭听觉无法断定语义。
   - 专有名词歧义：涉及人名、地名、品牌或垂直领域术语。

2. Discard (舍弃)：
   - 结果高度一致：内容基本相同，仅存在标点、空格或无意义语气词（嗯、啊、哦）的差异。
   - 质量极差：其中一方为乱码、重复无意义字符或完全不通顺。
   - 纯长度差异：一方多出了几个无关紧要的虚词，但不影响核心语义判断。

注意：可能两个结果区别不大，但它是有用的高价值数据。

输出要求：
请直接返回 JSON 格式，不要输出任何开场白。格式如下：
{
    "filter_result": "Keep" | "Discard",
    "reason": "简短说明。例如：'存在可能的同音歧义，需视频动作判定' 或 '文本一致，无信息差'。"
}

待分析数据：
- Result A: "{{ASR_A}}"
- Result B: "{{ASR_B}}"
- GroundTruth: "{{GroundTruth}}"
'''


def analyze_jsonl_entry(entry, client):

    whisper_text = entry.get("whisper_text", "")
    doubao_text = entry.get("doubao_text", "")
    GroundTruth = entry.get("text", "")

    if len(whisper_text) > 8192:
        whisper_text = whisper_text[:8192]
        return {"data": None}


    try:
        item_PROMPT = PROMPT.replace("{{ASR_A}}", whisper_text)
        item_PROMPT = item_PROMPT.replace("{{ASR_B}}", doubao_text)
        item_PROMPT = item_PROMPT.replace("{{GroundTruth}}", GroundTruth)



        api_response = client.chat.completions.create(
            model=MODEL_NAME,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": item_PROMPT
                            },
                        ]
                    },
                ],
            stream=False
        )
        answer = api_response.choices[0].message.content.strip()
        try:
            evaluation_dict = json.loads(answer)
        except:
            try:
                answer, raw = extract_and_clean_json(answer)
                evaluation_dict = json.loads(answer)
            except Exception as e:
                import traceback
                print(traceback.format_exc())


        print(f"Result: {answer}")
        entry.update(evaluation_dict)


        return {"answer": answer, "answer_dict": evaluation_dict, "data": entry}
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        print(f"API error: {e}")
        return {"data": None}

def main():
    parser = argparse.ArgumentParser(description="Batch process JSONL with GPT filter")
    parser.add_argument("--jsonl", required=True, nargs='+', help="Input jsonl file path(s)")
    parser.add_argument("--max-workers", type=int, default=8, help="Max threads")
    args = parser.parse_args()

    # Check for API configuration
    if not all([AZURE_ENDPOINT, API_VERSION, API_KEY]):
        print("Error: Please set AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION, and AZURE_OPENAI_API_KEY environment variables.")
        return

    back_fix = '-gpt4o-filter-output.jsonl'
    
    # Initialize API client
    client = openai.AzureOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        api_version=API_VERSION,
        api_key=API_KEY,
    )

    for jsonl_item in args.jsonl:
        if not os.path.exists(jsonl_item):
            print(f"File not found: {jsonl_item}")
            continue

        if jsonl_item.endswith(back_fix):
            print(f"Skipping output file: {jsonl_item}")
            continue

        print(f'Processing: {jsonl_item}')
        output_jsonl_path = os.path.splitext(jsonl_item)[0] + back_fix
        print('Output path:', output_jsonl_path)

        # Read jsonl
        entries = []
        with open(jsonl_item, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    entries.append(entry)
                except Exception as e:
                    print(f"Skipping invalid line: {e}")

        processed_ids = set()
        if os.path.exists(output_jsonl_path):
            print(f"Output file {output_jsonl_path} exists, checking for processed entries.")
            with open(output_jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        processed_entry = json.loads(line)
                        if 'data_id' in processed_entry:
                            processed_ids.add(processed_entry['data_id'])
                    except Exception as e:
                        print(f"Warning: Error parsing processed file line: {e}")

        unprocessed_entries = [entry for entry in entries if entry.get('data_id') not in processed_ids]
        
        print(f"Total entries: {len(entries)}, Processed: {len(processed_ids)}, To process: {len(unprocessed_entries)}")

        if unprocessed_entries:
            # Multi-threaded processing
            with ThreadPoolExecutor(max_workers=args.max_workers) as executor, open(output_jsonl_path, "a", encoding="utf-8") as f_out:
                future_to_entry = {executor.submit(analyze_jsonl_entry, entry, client): entry for entry in unprocessed_entries}
                for future in tqdm(as_completed(future_to_entry), total=len(future_to_entry), desc="Processing", ncols=80):
                    result = future.result()
                    if result.get("data", None) is None:
                        continue
                    
                    f_out.write(json.dumps(result.get("data"), ensure_ascii=False) + "\n")
                    f_out.flush()
            print(f"Done. Results appended to: {output_jsonl_path}")
        else:
            print(f"No new entries to process.")
                

                
if __name__ == "__main__":
    main()