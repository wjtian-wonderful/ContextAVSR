import jsonlines
from transformers import pipeline
import jiwer
import torch
import torchaudio
from torchaudio.functional import resample
import multiprocessing
from tqdm import tqdm
import math
import librosa
import re
import traceback
import re
import os

def load_audio(speech_input, target_sr=16000):
    """
    使用librosa加载音频并统一采样率为16kHz，转为单声道，并转换为tensor
    :param speech_input: 音频文件路径
    :param target_sr: 目标采样率，固定为16000Hz
    :return: 处理后的音频tensor
    """
    # 使用librosa加载音频，强制转为单声道
    waveform, sample_rate = librosa.load(speech_input, sr=target_sr, mono=True)
    # 转换为PyTorch tensor，并添加批次维度
    waveform_tensor = torch.tensor(waveform)
    
    return waveform_tensor


def process_chunk(chunk, gpu_id, true_output_path, false_output_path):
    """
    A worker function to process a chunk of data on a specific GPU.
    """
    device = f"cuda:{gpu_id}"
    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3",
        tokenizer="openai/whisper-large-v3",
        device=device
    )
    wer_computer = jiwer.wer

    true_lines = []
    false_lines = []

    key_list = []


    for line in tqdm(chunk, desc=f"Processing GPU {gpu_id}"):
        try:

            text_key = "text"
            speech_key = "speech"
            if speech_key not in line or text_key not in line:
                print(f'没有 {speech_key} {text_key}')
                continue
            reference_text = line[text_key].strip()

            speech = line[speech_key].replace('.mp4','.wav')
            audio = load_audio(speech).to(device)
            asr_result = asr_pipeline(audio)
            predicted_text = asr_result["text"].strip()


            line["whisper_text"] = predicted_text
            line["doubao_text"] = reference_text
            true_lines.append(line)


        except Exception as e:
            print(f"Error processing line on GPU {gpu_id}「」」「」\n{line}: \n{e}")
            # 获取异常信息和行数
            # traceback.format_exc() 会返回包含文件名、行数和错误信息的字符串
            error_info = traceback.format_exc()
            print("完整错误信息：")
            print(error_info)

            continue

    with jsonlines.open(true_output_path.replace('.jsonl', f'_gpu{gpu_id}.jsonl'), "w") as true_out:
        true_out.write_all(true_lines)

    # with jsonlines.open(false_output_path.replace('.jsonl', f'_gpu{gpu_id}.jsonl'), "w") as false_out:
    #     false_out.write_all(false_lines)


def main(input_path, true_output_path, false_output_path):
    # if not torch.cuda.is_available() or torch.cuda.device_count() <= 1:
    #     print("Single GPU or CPU detected. Running in single-process mode.")
    #     # Fallback to original single-process implementation if no multi-GPU
    #     run_single_process(input_path, true_output_path, false_output_path)
    #     return

    # num_gpus = torch.cuda.device_count()
    gpu_list = [0,1,2,3,4,5,6,7]
    num_gpus = len(gpu_list)
    print(f"Found {num_gpus} GPUs. Starting multi-process execution.")

    with jsonlines.open(input_path, "r") as infile:
        lines = list(infile)

    print(f' length lines {len(lines)}')
    chunk_size = math.ceil(len(lines) / num_gpus)
    chunks = [lines[i:i + chunk_size] for i in range(0, len(lines), chunk_size)]
    print(f' chunks lines {len(chunks)}')

    processes = []
    for i in gpu_list:
        if i < len(chunks):
            p = multiprocessing.Process(target=process_chunk, args=(chunks[i], i, true_output_path, false_output_path))
            processes.append(p)
            p.start()
        else:
            raise ValueError(f"GPU {i} has no data to process.")

    for p in processes:
        p.join()

    # Merge the results from all GPUs
    merge_output_files(true_output_path, num_gpus)
    merge_output_files(false_output_path, num_gpus)

    print("Processing complete!")
    print(f"Data with accuracy >= 90% has been written to: {true_output_path}")
    print(f"Data with accuracy < 90% has been written to: {false_output_path}")


def merge_output_files(base_output_path, num_gpus):
    with jsonlines.open(base_output_path, "w") as outfile:
        for i in range(num_gpus):
            temp_file_path = base_output_path.replace('.jsonl', f'_gpu{i}.jsonl')
            try:
                with jsonlines.open(temp_file_path, "r") as infile:
                    for line in infile:
                        outfile.write(line)
                # os.remove(temp_file_path) # Optionally remove temp files
            except FileNotFoundError:
                continue

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)


    INPUT_JSONL = 
    output_dir =
    os.makedirs(output_dir, exist_ok=True)
    eee = os.path.basename(os.path.splitext(INPUT_JSONL)[0])
    TRUE_JSONL = f"{output_dir}/{eee}-wer.jsonl"         # 达标输出文件
    FALSE_JSONL = f"{output_dir}/tmp_gpu_false.jsonl"       # 未达标输出文件

    main(INPUT_JSONL, TRUE_JSONL, FALSE_JSONL)


# ca swift_sft