# -*- coding: utf-8 -*-


# Efficient inference with FP8 checkpoint
# Requires NVIDIA H100+ and CUDA 12+

# vllm serve Qwen/Qwen3-VL-8B-Instruct \
#   --tensor-parallel-size 8 \
#   --gpu-memory-utilization 0.9  # 调小显存利用率
#   --mm-encoder-tp-mode data \
#   --enable-expert-parallel \
#   --async-scheduling \
#   --media-io-kwargs '{"video": {"num_frames": -1}}' \
#   --host 0.0.0.0 \
#   --port 22002

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

import os
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

def prepare_inputs_for_vllm(messages, processor):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # qwen_vl_utils 0.0.14+ reqired
    
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True
    )

    mm_data = {}
    if image_inputs is not None:
        mm_data['image'] = image_inputs
    if video_inputs is not None:
        mm_data['video'] = video_inputs

    return {
        'prompt': text,
        'multi_modal_data': mm_data,
        'mm_processor_kwargs': video_kwargs
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default='data/tagging-speech-understanding-lq2.jsonl', help='Path to the input jsonl file.')
    parser.add_argument('--output_file', type=str, default='True', help='Path to the output jsonl file.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for vLLM inference (adjust based on GPU memory)')
    parser.add_argument('--num_workers', type=int, default=16, help='Number of worker threads for data preparation.')

    args = parser.parse_args()

    INPUT_JSONL_FILES = [



    ]
    output_dir = 

    os.makedirs(output_dir, exist_ok=True)
    # TODO: change to your own checkpoint path
    checkpoint_path = "Qwen/Qwen3-VL-8B-Instruct"
    checkpoint_path = "Qwen/Qwen3-VL-8B-Instruct"
    checkpoint_path = "Qwen/Qwen3-VL-30B-A3B-Thinking"
    checkpoint_path = "Qwen/Qwen3-VL-30B-A3B-Instruct"


    processor = AutoProcessor.from_pretrained(checkpoint_path)

    llm = LLM(
        model=checkpoint_path,
        mm_encoder_tp_mode="data",
        gpu_memory_utilization=0.90,
        enable_expert_parallel=True,
        tensor_parallel_size=torch.cuda.device_count(),
        seed=0
    )

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=4096,
        top_k=-1,
        stop_token_ids=[],
    )
    from tqdm import tqdm

    # Calculate total items for progress bar
    prompt = '''
            你是一个视频内容分析专家。请提取视频中所有的文字信息，并将其分类为“字幕”和“场景文字”。
            请严格按照以下 JSON 格式返回结果，不要包含任何额外的解释：

            {
                "subtitles": ["字幕句子1", "字幕句子2", ...], 
                "scene_text_description": "一段描述性文字。例如：画面左侧的商店招牌上写着'龙湖天街'，主角手中的书封面写着'Python Code'。"
            }

            要求：
            1. subtitles: 提取视频底部出现的解说或对话字幕，按时间顺序组成字符串列表。若无，返回 []。
            2. scene_text_description: 提取视频画面环境中出现的文字（如路牌、屏幕、海报等），并结合其出现的位置或载体进行描述，汇总成一段通顺的文字。若无，返回空字符串。
            '''
    
    prompt = '''
    你是一个视频内容分析专家。请仔细分析视频的视觉画面、音频信息和文字信息，提取以下四项内容。
请严格按照 JSON 格式返回结果，不要包含 markdown 标记或额外的解释。

{
    "video_desc": "详细概括视频的视觉内容。包括场景环境（如地点、光线）、主要人物（外貌、衣着）、关键动作以及镜头语言（如特写、全景）。",
    "subtitles": ["字幕句子1", "字幕句子2", ...],
    "scene_text_description": "一段描述性文字。提取视频画面环境中出现的非字幕文字（如路牌、屏幕内容、物体标签等），并结合其位置或载体进行自然描述。",
    "speaking_scenario": "推测说话时的情境。例如办公室，小区，街道等。"
}

要求：
1. **video_desc**: 侧重于“看到了什么”，描述客观的画面流转和视觉元素。
2. **subtitles**: 提取水印或对话字幕，按时间顺序排列。若无，返回 []。
3. **scene_text_description**: 侧重于环境中的“OCR文字信息”，需融入语境描述（例如：“桌上的电脑屏幕显示着代码” 而非仅返回 “代码”）。
4. **speaking_scenario**: 侧重于“语音产生的背景”，帮助理解语音的情感和交互逻辑。
'''

    def item_message_generator(input_file, processed_count, prompt_text):
        with open(input_file, 'r', encoding='utf-8') as f_in:
            for _ in range(processed_count):
                try:
                    next(f_in)
                except StopIteration:
                    return
            for line in f_in:
                item = json.loads(line)
                message = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "video": item['video']},
                            {"type": "text", "text": prompt_text},
                        ],
                    }
                ]
                yield item, message

    def process_item(item_message_tuple):
        item, message = item_message_tuple
        try:
            input_data = prepare_inputs_for_vllm(message, processor)
        except Exception as e:
            print(f"Error processing item :{item} {e}")
            return None, None
        return item, input_data

    for input_file in INPUT_JSONL_FILES:
        args.input_file = input_file
        args.output_file = f"{output_dir}/{os.path.basename(args.input_file).split('.')[0]}.jsonl"
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        processed_count = 0
        if os.path.exists(args.output_file):
            with open(args.output_file, 'r', encoding='utf-8') as f:
                processed_count = sum(1 for _ in f)
            print(f"Found {processed_count} already processed items in {args.output_file}. Resuming.")

        with open(args.input_file, 'r', encoding='utf-8') as f:
            total_items = sum(1 for _ in f)

        with ThreadPoolExecutor(max_workers=args.num_workers) as executor, \
             open(args.output_file, 'a', encoding='utf-8') as f_out:

            generator = item_message_generator(args.input_file, processed_count, prompt)
            try:
                results_iterator = executor.map(process_item, generator)
            except Exception as e:
                print(f"Error processing batch: {e}")
                continue
            batch_items = []
            batch_inputs = []
            
            remaining_items = total_items - processed_count
            with tqdm(total=remaining_items, desc="Processing items") as pbar:
                for i, (item, input_data) in enumerate(results_iterator):
                    # Check for errors during item preparation
                    if input_data is None:
                        print(f"\nSkipping item that failed preparation: {item}")
                        pbar.update(1) # Mark as processed (skipped)
                    else:
                        batch_items.append(item)
                        batch_inputs.append(input_data)

                    # Process batch when reaching batch size or it's the last item
                    is_last_item = (i == remaining_items - 1)
                    if batch_inputs and (len(batch_inputs) >= args.batch_size or is_last_item):
                        print(f"\nProcessing batch with {len(batch_inputs)} items")
                        try:
                            outputs = llm.generate(batch_inputs, sampling_params=sampling_params)
                            # Process and write results for this batch
                            for out_item, output in zip(batch_items, outputs):
                                generated_text = output.outputs[0].text
                                output_item = out_item
                                output_item['ocr'] = generated_text
                                f_out.write(json.dumps(output_item, ensure_ascii=False) + '\n')
                        except Exception as e:
                            print(f"Error processing batch: {e} {batch_inputs}")
                        
                        pbar.update(len(batch_items))
                        # Reset batch
                        batch_items = []
                        batch_inputs = []
                        f_out.flush()  # Ensure data is written to disk