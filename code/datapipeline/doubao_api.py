import json
import time
import uuid
import requests
import base64
import os
import json
import librosa
from typing import List, Dict
import soundfile as sf
import base64
import io
import numpy as np
import librosa
import soundfile as sf

SUPPORTED_VIDEO_EXTENSIONS = ('.wav')


token = 
appid = 
resource = "volc.bigasr.auc"

query_url = "https://openspeech-direct.zijieapi.com/api/v3/auc/bigmodel/query"
submit_url = "https://openspeech.bytedance.com/api/v3/auc/bigmodel/submit"


def find_video_files(root_dir):
    """递归查找目录中的所有视频文件并排序"""
    video_paths = []
    
    if not os.path.isdir(root_dir):
        raise NotADirectoryError(f"目录不存在: {root_dir}")
    
    # 递归遍历目录
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(SUPPORTED_VIDEO_EXTENSIONS):
                video_path = os.path.join(dirpath, filename)
                if 'wav_split' not in video_path:
                    video_paths.append(video_path)
    
    # 按路径排序
    video_paths.sort()
    return video_paths




def librosa_resample_and_to_base64(audio_path, target_sample_rate=16000, target_channels=1):
    """
    用 librosa 重采样音频并转为 base64 编码
    :param audio_path: 本地音频文件路径
    :param target_sample_rate: 目标采样率（如 16000 Hz）
    :param target_channels: 目标声道数（1=单声道，2=立体声）
    :return: base64 编码字符串（失败返回 None）
    """
    # audio_path = "/mnt/bn/twj-data-multimodal2/common_voice_en_456493.mp3"
    try:
        # 1. 读取音频文件（返回：音频数据（numpy数组）、原采样率）
        # dtype='float32'：librosa 默认读取为 float32 格式（取值范围 [-1.0, 1.0]）
        y_resampled, sr = librosa.load(
            audio_path,
            sr=target_sample_rate,  # sr=None 保留原采样率，不自动重采样
            mono=True,  # mono=False 保留原声道数（后续手动调整）
            dtype='float32',
            res_type='soxr_vhq'
        )

        # 4. 格式转换：float32（[-1.0, 1.0]）→ 16位 PCM（int16，范围 [-32768, 32767]）
        # 多数 API 要求 16 位 PCM 格式，避免 float 格式不兼容
        y_pcm = (y_resampled * 32767).astype(np.int16)

        # 5. 内存中存储为 WAV 格式（无需生成临时文件）
        audio_io = io.BytesIO()
        # 用 soundfile 导出：指定采样率、格式、编码
        sf.write(
            audio_io,
            y_pcm,
            samplerate=target_sample_rate,
            format='WAV',  # 导出格式（API 支持的话可改为 'MP3'）
            subtype='PCM_16'  # 16 位 PCM 编码
        )

        # 6. 读取内存中的二进制数据，转 base64 编码
        audio_io.seek(0)  # 回到数据流开头
        audio_bytes = audio_io.read()
        base64_str = base64.b64encode(audio_bytes).decode('utf-8')

        print(f"✅ 重采样 + base64 编码成功！base64 长度：{len(base64_str)} 字符")
        return base64_str

    except FileNotFoundError:
        print(f"❌ 错误：音频文件不存在 → {audio_path}")
        return None
    except Exception as e:
        print(f"❌ 处理失败：{str(e)}")
        return None


def submit_task(audio_data):

    task_id = str(uuid.uuid4())

    headers = {
        "X-Api-App-Key": appid,
        "X-Api-Access-Key": token,
        "X-Api-Resource-Id": resource,
        # "X-Api-Resource-Id": "volc.bigasr.auc_turbo",
        "X-Api-Request-Id": task_id,
        "X-Api-Sequence": "-1"
    }

    request = {
        "user": {
            "uid": "fake_uid"
        },
        "audio": {
            "data": audio_data,
            "format": "wav",
            # "codec": "map3",
            # "rate": 48000,
            # "bits": 16,
            #"channel": 2
        },
        "request": {
            "model_name": "bigmodel",
            "model_version": "400",
            # "model_name": "bigmodel", 
            "enable_channel_split": False, 
            "enable_ddc": True, 
            "enable_speaker_info": True, 
            "enable_punc": True, 
            "enable_itn": True,
            # "enable_itn": True,
            # "enable_punc": True,
            # "enable_ddc": True,
            "show_utterances": True,
            # "enable_channel_split": True,
            # "vad_segment": True,
            # "enable_speaker_info": True,
            "show_speech_rate": True,
            "enable_gender_detection": True,
            "enable_emotion_detection": True,
            "enable_lid": True,
            # "corpus": {
            #     # "boosting_table_name": "test",
            #     "correct_table_name": "",
            #     "context": ""
            # }
        }
    }
    print(f'Submit task id: {task_id}')
    response = requests.post(submit_url, data=json.dumps(request), headers=headers)
    if 'X-Api-Status-Code' in response.headers and response.headers["X-Api-Status-Code"] == "20000000":
        print(f'Submit task response header X-Api-Status-Code: {response.headers["X-Api-Status-Code"]}')
        print(f'Submit task response header X-Api-Message: {response.headers["X-Api-Message"]}')
        x_tt_logid = response.headers.get("X-Tt-Logid", "")
        print(f'Submit task response header X-Tt-Logid: {response.headers["X-Tt-Logid"]}\n')
        return task_id, x_tt_logid
    # else:
    print(f'Submit task failed and the response are: {response}')

    print(f'Submit task failed and the response headers are: {response.headers}')
    return task_id


def query_task(task_id, x_tt_logid):

    headers = {
        "X-Api-App-Key": appid,
        "X-Api-Access-Key": token,
        "X-Api-Resource-Id": resource,
        "X-Api-Request-Id": task_id,
        "X-Tt-Logid": x_tt_logid  # 固定传递 x-tt-logid
    }

    response = requests.post(query_url, json.dumps({}), headers=headers)

    if 'X-Api-Status-Code' in response.headers:
        print(f'Query task response header X-Api-Status-Code: {response.headers["X-Api-Status-Code"]}')
        print(f'Query task response header X-Api-Message: {response.headers["X-Api-Message"]}')
        print(f'Query task response header X-Tt-Logid: {response.headers["X-Tt-Logid"]}\n')
    else:
        print(f'Query task failed and the response headers are: {response.headers}')
    return response


def call_api(audio_path, output_path):
    file_url = audio_path
    
    target_sample_rate = 16000  # 目标采样率（OpenSpeech 常用 16kHz）
    # 执行重采样 + base64 编码
    try:
        base64_audio = librosa_resample_and_to_base64(
            audio_path=audio_path,
            target_sample_rate=target_sample_rate,
        )
    except:
        print(f'librosa_resample_and_to_base64 failed and the audio path is: {audio_path}')
        return

    task_id, x_tt_logid = submit_task(base64_audio)
    while True:
        query_response = query_task(task_id, x_tt_logid)
        code = query_response.headers.get('X-Api-Status-Code', "")
        if code == '20000000':  # task finished
            # print(query_response.json())
            # 写入到文本文件中
            with open(output_path, 'w') as f:
                json.dump(query_response.json(), f, ensure_ascii=False, indent=4)
            print("SUCCESS!")
            return 
        elif code != '20000001' and code != '20000002':  # task failed
            print("FAILED!")
            return
        time.sleep(1)


def call_api_wrapper(args):
    """Wrapper for call_api to be used with multiprocessing pool."""
    audio_path, output_path = args
    try:
        call_api(audio_path, output_path)
    except Exception as e:
        print(f'call_api failed for {audio_path} with error: {e}')


if __name__ == '__main__':
    import multiprocessing as mp
    from tqdm import tqdm
    mp.set_start_method("spawn", force=True)

    # Set the number of concurrent processes to N
    N = 8  # You can change the number of concurrent processes here


    INPUT_JSONL_FILES = [

    ]

    base_output_dir =


    for jsonl_path in INPUT_JSONL_FILES:

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

        audio_paths = []
        for item in lines:
            item = json.loads(item)
            data_id = item['data_id']
            if 'speech' in item:
                audio_paths.append((data_id, item['speech']))
            elif 'video' in item:
                speech_path = item['video'].replace('.mp4', '.wav')
                audio_paths.append((data_id, speech_path))
            elif 'audios' in item:
                audio_paths.append((data_id, item['audios'][0]))
            
        audio_length = len(audio_paths)
        print(f'total paths {audio_length}')

        tasks = []
        # Prepare the list of tasks to be processed
        for data_id, audio_path in audio_paths:
            # print(f'handling: {audio_path}')

 
            output_path = os.path.join(final_output_dir, f'{data_id}.json')
            # output_path = audio_path.replace('.wav', '____doubao.json')
            # output_path = audio_path.replace('.wav', '.json')
            if os.path.exists(output_path):
                print(f'output path {output_path} already exists')
                continue

            tasks.append((audio_path, output_path))

        print(f"A total of {len(tasks)} tasks need to be processed.")

        # Use a process pool to process tasks
        with mp.Pool(processes=N) as pool:
            # Use tqdm to display the progress bar
            with tqdm(total=len(tasks)) as pbar:
                for _ in pool.imap_unordered(call_api_wrapper, tasks):
                    pbar.update()

        print("All tasks have been processed!")