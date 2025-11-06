
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目名称：denoise_final
文件名称：voice_restorer.py

功能描述：
1.人声分离，提供纯人声音频（./htdemucs/vocals.wav）、纯噪声音频（./htdemucs/no_vocals.wav）
2.修复分离后的纯人声音频

处理过程：
1. input.wav->FIR滤波器保留5.5kHz以下全部信号（去除易去除的高频音调噪音）->./htdemucs/input_fir.wav
2. ./htdemucs/input_fir.wav->demucs人声分离->./htdemucs/vocals.wav
3. ./htdemucs/vocals.wav->voicefixer语音修复->output_temp.wav
4. ./htdemucs/output_temp.wav->音频截断静音处理（voicefixer语音修复后的音频在最后总有噪音，原因不明）->output.wav

使用示例(终端管理员启动)：
    Python voice_restorer.py

作者：机械2507张溪麟
学号：2500473
联系方式：
    - 邮箱：zhangxilin@mails.neu.edu.cn
    - GitHub：https://github.com/yourusername

创建时间：2025-11-03
最后修改时间：2025-11-06
版本：1.9.0

依赖库：
    - voicefixer>=0.1.3
    - torch>=1.9.0
    - numpy>=1.21.0
    - scipy>=1.7.0
    - demucs>=4.0.1

"""

from voicefixer import VoiceFixer
import torch
import numpy as np
from scipy.io import wavfile
import scipy.signal as signal
from demucs.separate import main as demucs_main
from pathlib import Path
from typing import Optional, Dict, Any


def filter_audio_fir(input_file, output_file, cutoff_freq=5500, numtaps=101):
    """
    FIR滤波器
    
    参数:
        input_file: 输入音频文件路径
        output_file: 输出音频文件路径
        cutoff_freq: 截止频率(Hz)
        numtaps: 滤波器抽头数(影响滤波精度)
    """
    # 读取音频文件
    sample_rate, audio_data = wavfile.read(input_file)
    
    # 计算归一化截止频率
    nyquist_freq = sample_rate / 2
    normal_cutoff = cutoff_freq / nyquist_freq
    
    # 设计FIR低通滤波器
    taps = signal.firwin(numtaps, normal_cutoff, window='hamming')
    
    # 应用滤波器
    filtered_audio = signal.lfilter(taps, 1.0, audio_data, axis=0)
    
    # 保存滤波后的音频
    filtered_audio = np.int16(filtered_audio / np.max(np.abs(filtered_audio)) * 32767)
    wavfile.write(output_file, sample_rate, filtered_audio)
    print(f"FIR滤波完成！已滤除 {cutoff_freq}Hz 以上信号")

def trim_audio(input_file, output_file, duration=2):
    """
    将音频指定时长之后的部分静音（设为0）
    
    参数:
        input_file: 输入音频文件路径
        output_file: 输出音频文件路径
        duration: 保留音频的时长(秒)，之后的部分将被静音
    """
    # 读取音频文件
    sample_rate, audio_data = wavfile.read(input_file)
    
    # 计算要保留的样本数
    samples_to_keep = int(duration * sample_rate)
    
    # 如果音频长度小于等于要保留的时长，则无需处理
    if len(audio_data) <= samples_to_keep:
        print(f"音频长度只有 {len(audio_data)/sample_rate:.2f} 秒，小于等于 {duration} 秒，无需静音处理")
        # 直接复制原文件
        wavfile.write(output_file, sample_rate, audio_data)
        return
    
    # 将指定时长之后的音频数据设为0（静音）
    audio_data[samples_to_keep:] = 0
    
    # 保存处理后的音频
    wavfile.write(output_file, sample_rate, audio_data)
    print(f"音频处理完成！")

class AudioRestorer:
    def __init__(self):
        # 初始化 VoiceFixer
        self.voicefixer = VoiceFixer()
        
        # 检查是否有可用的GPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {self.device}")
    
    def restore_with_options(self, input_path, output_path, mode=1, use_cuda=None):
        """
        带选项的音频修复
        参数:

        mode: 0 - 普通模式 (针对严重损坏的音频)
              1 - 语音增强模式 (针对轻微损坏的音频)
              2 - 语音修复模式 (相比语音增强更为轻微)
        use_cuda: 是否使用GPU加速
        """
        if use_cuda is None:
            use_cuda = self.device == "cuda"
            
        try:
            self.voicefixer.restore(
                input=input_path,
                output=output_path, 
                cuda=use_cuda,
                mode=mode
            )
            print(f"成功修复音频: {input_path} -> {output_path}")
            return True
        except Exception as e:
            print(f"修复失败: {e}")
            return False


class DemucsSeparator:
    """Demucs音频分离工具类"""
    
    def __init__(
        self,
        default_model: str = "htdemucs",
        default_output_dir: str = "demucs_output",
        default_quality: int = 0
    ):
        """
        初始化分离工具（配置默认参数）
        default_model: 默认使用的模型（可选：htdemucs、mdx23c、htdemucs_ft等）
        default_output_dir: 默认输出文件夹
        default_quality: 默认质量等级（0=快速，1=高质量）
        """
        self.default_model = default_model
        self.default_output_dir = Path(default_output_dir)
        self.default_quality = default_quality
        
        # 确保输出文件夹存在
        self.default_output_dir.mkdir(exist_ok=True, parents=True)

    def _build_args(
        self,
        input_path: str,
        model: Optional[str] = None,
        output_dir: Optional[str] = None,
        two_stems: Optional[str] = None,
        quality: Optional[int] = None,
        filename_pattern: Optional[str] = None
    ) -> list:
        """构造Demucs运行参数（内部辅助方法）"""
        # 优先使用传入参数，无则用默认值
        model = model or self.default_model
        output_dir = output_dir or str(self.default_output_dir)
        quality = quality or self.default_quality

        # 组装Demucs所需的命令行格式参数
        args = [
            input_path,
            "-o", output_dir,
            "-n", model,
        ]
        
        # 如果指定了文件名模式，直接输出到根目录
        if filename_pattern:
            args.extend(["--filename", filename_pattern])
        else:
            # 默认模式：{track}/{stem}.{ext} 表示在输出目录下创建轨道文件夹
            args.extend(["--filename", "{track}/{stem}.{ext}"])
        
        # 如果指定了特定轨道分离
        if two_stems:
            args.extend(["--two-stems", two_stems])
        
        # 如果启用高质量模式
        if quality == 1:
            args.extend(["--shifts", "1"])  # 高质量参数
            
        return args

    def separate(
        self,
        input_path: str,
        model: Optional[str] = None,
        output_dir: Optional[str] = None,
        two_stems: Optional[str] = None,
        quality: Optional[int] = None,
        filename_pattern: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        执行音频分离（核心方法，供主程序调用）
        :param input_path: 输入音频文件路径（支持wav/mp3/flac等）
        :param model: 覆盖默认模型
        :param output_dir: 覆盖默认输出文件夹
        :param two_stems: 分离特定轨道（vocals/drums/bass/other）
        :param quality: 覆盖默认质量等级
        :param filename_pattern: 文件名模式（控制输出路径结构）
        :return: 分离结果字典（状态、信息、输出路径）
        """
        # 验证输入文件
        input_file = Path(input_path)
        if not input_file.exists():
            return {
                "success": False,
                "message": f"输入文件不存在：{input_path}",
                "output_dir": None
            }

        # 构造参数并执行分离
        try:
            demucs_args = self._build_args(input_path, model, output_dir, two_stems, quality, filename_pattern)
            print(f"执行命令: demucs {' '.join(demucs_args)}")
            demucs_main(demucs_args)
            
            final_output_dir = output_dir or str(self.default_output_dir)
            return {
                "success": True,
                "message": f"分离成功！文件保存至：{final_output_dir}",
                "output_dir": final_output_dir
            }
        except SystemExit:
            # demucs_main 会调用 sys.exit()，我们需要捕获它
            final_output_dir = output_dir or str(self.default_output_dir)
            return {
                "success": True,
                "message": f"分离成功！文件保存至：{final_output_dir}",
                "output_dir": final_output_dir
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"分离失败：{str(e)}",
                "output_dir": None
            }


if __name__ == "__main__":


    # 1.预处理：使用FIR滤波器滤除5.5kHz以上信号
    filter_audio_fir("input.wav", "./htdemucs/input_fir.wav", 5500)

    # 2.分离人声
    # 实例化分离工具
    separator = DemucsSeparator(
        default_model="htdemucs",  # 使用默认模型
        default_output_dir="./",   # 直接输出到程序根目录
    )
    # 调用分离方法
    result = separator.separate(
        input_path="./htdemucs/input_fir.wav",  # 你的音频文件路径
        two_stems="vocals",  # 只分离人声轨道
        quality=1,  # 启用高质量模式
        # 使用自定义文件名模式，直接在根目录生成文件
        filename_pattern="{stem}.{ext}"  # 包含文件扩展名
    )

    # 处理结果
    if result["success"]:
        print(f"分离成功： {result['message']}")
    else:
        print(f"分离失败： {result['message']}")


    # 3.修复
    restorer = AudioRestorer()
    audio_files = [
        # 选择文件，可以选择多个，这里只有一个
        ("./htdemucs/vocals.wav", "./htdemucs/output_temp.wav")
    ]
    
    for input_file, output_file in audio_files:
        success = restorer.restore_with_options(
            input_file, 
            output_file, 
            mode=2,  # 语音增强模式
            use_cuda=True
        )
        
        if success:
            print(f" {input_file} 音频修复完成")
        else:
            print(f" {input_file} 音频修复失败")
    trim_audio("./htdemucs/output_temp.wav", "output.wav", duration=2)