# Denoise Final - 音频降噪与语音修复工具

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Version](https://img.shields.io/badge/Version-1.9.0-orange)

## 项目简介

`denoise_final` 是一个音频降噪和语音修复工具，专门设计用于从嘈杂的音频中提取清晰的人声。该项目结合了先进的语音分离技术和深度学习语音修复算法，能够有效去除背景噪音并提升语音质量。

## 主要功能

 **人声分离**
- 使用 Demucs 算法分离人声和背景噪音
- 输出纯净人声音频 (`./htdemucs/vocals.wav`)
- 输出纯噪声音频 (`./htdemucs/no_vocals.wav`)

 **语音修复**
- 使用 VoiceFixer 技术修复分离后的人声
- 提升语音清晰度和质量
- 自动处理音频截断问题

## 处理流程

1. **FIR 滤波预处理**
   - 输入：`input.wav`
   - 处理：5.5kHz 低通滤波，去除高频噪音
   - 输出：`./htdemucs/input_fir.wav`

2. **人声分离**
   - 输入：`./htdemucs/input_fir.wav`
   - 处理：Demucs 人声分离算法
   - 输出：`./htdemucs/vocals.wav` (纯净人声)

3. **语音修复**
   - 输入：`./htdemucs/vocals.wav`
   - 处理：VoiceFixer 深度修复
   - 输出：`output_temp.wav`

4. **后处理**
   - 输入：`output_temp.wav`
   - 处理：音频截断静音处理
   - 输出：`output.wav` (最终结果)


### 依赖安装
```bash
pip install voicefixer>=0.1.3
pip install torch>=1.9.0
pip install numpy>=1.21.0
pip install scipy>=1.7.0
pip install demucs>=4.0.1
```

## 使用方法

### 基本使用
1. 将待处理的音频文件命名为 `input.wav` 并放置在项目根目录
2. 以管理员权限运行：
```bash
python voice_restorer.py
```

### 输出文件
处理完成后，你将获得：
- `output.wav` - 最终修复后的纯净人声
- `./htdemucs/` 目录下的中间处理文件

## 项目结构
```
denoise_final/
├── voice_restorer.py     # 主程序文件
├── input.wav            # 输入音频文件
├── output.wav           # 输出音频文件
└── htdemucs/            # 处理中间文件目录
    ├── input_fir.wav    # FIR滤波后音频
    ├── vocals.wav       # 分离的人声
    └── no_vocals.wav    # 分离的噪音
```

## 技术特点

-  **多级处理**：FIR滤波 + 人声分离 + 语音修复
-  **噪音控制**：有效去除高频噪音和背景杂音
-  **自动化流程**：一键式处理，无需复杂配置
-  **质量保证**：针对 VoiceFixer 输出的特殊处理

## 注意事项

- 需要以管理员权限运行程序
- 确保输入音频为 WAV 格式
- 处理时间取决于音频长度和硬件性能
- 建议音频采样率为 16kHz

## 致谢

感谢以下开源项目的支持：
- [Demucs](https://github.com/facebookresearch/demucs) - 音频源分离
- [VoiceFixer](https://github.com/haoheliu/voicefixer) - 语音修复
- 东北大学提供的学术支持

---

*如有问题或建议，请通过邮箱联系作者*
