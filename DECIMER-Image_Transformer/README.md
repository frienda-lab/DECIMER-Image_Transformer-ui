# DECIMER Desktop 化学结构识别软件

本项目在 [DECIMER Image Transformer](https://github.com/Kohulan/DECIMER-Image_Transformer) 的基础上增加了 Windows 桌面可视化界面，并整合了 [DECIMER Image Segmentation](https://github.com/Kohulan/DECIMER-Image-Segmentation)。

软件支持：

- 拖入单张或多张化学结构图片；
- 批量处理整个图片文件夹；
- 先从论文页面或专利页面分割多个化学结构，再逐个识别；
- 普通印刷结构和手绘结构识别；
- 每个结构保存为 `.smi` 文件；
- 生成可用 Excel 打开的 `smiles_results.csv` 汇总表；
- 保存分割后的结构图片；
- 完全离线运行。

## 普通用户：直接打开哪个文件

### 已有安装包

打开：

```text
release\DECIMER-Desktop-Setup.exe
```

按照提示安装。安装完成后，双击桌面上的 `DECIMER Desktop` 快捷方式即可使用。

安装版已经包含 Python、TensorFlow、三个模型和全部运行组件，不需要配置环境，也不需要首次联网下载模型。

如果 Windows SmartScreen 提示“未知发布者”，选择“更多信息 → 仍要运行”。当前安装包是本地构建版本，尚未使用商业代码签名证书。

### 在源码目录中运行

双击：

```text
run_ui.bat
```

脚本会创建 `.venv` 环境并安装所需组件，然后打开可视化界面。

## 界面使用方法

1. 将图片或图片文件夹拖到虚线区域，也可以点击“选择图片”或“选择文件夹”。
2. 如果需要处理下级文件夹，勾选“包含子文件夹”。
3. 一张图片只有一个独立化学结构时，不要勾选“先分割页面中的结构”。
4. 输入是论文页面、专利页面或一张图片包含多个结构时，勾选“先分割页面中的结构”。
5. 输入是手绘结构时，勾选“手绘结构模型”。
6. 点击“保存到…”选择输出目录。
7. 点击“开始识别”。

直接识别时，每张图片生成一个同名 `.smi` 文件。启用分割时，结果保存在：

```text
输出目录
├─ smiles_results.csv
└─ segments
   └─ 来源图片名称
      ├─ structure_001.png
      ├─ structure_001.smi
      ├─ structure_002.png
      └─ structure_002.smi
```

## 开发环境配置

### 系统要求

- Windows 10/11 64 位；
- Python 3.9 或更高版本；
- 建议至少 8 GB 内存，推荐 16 GB；
- 建议预留至少 6 GB 磁盘空间；
- NVIDIA GPU 不是必需，没有兼容 GPU 时使用 CPU。

当前开发和打包验证环境为：

```text
Python 3.13
TensorFlow 2.20
PySide6 6.x
OpenCV 5.x
```

### 创建虚拟环境

在 PowerShell 中进入项目目录：

```powershell
cd C:\Users\11588\Desktop\codex_test\DECIMER-Image_Transformer
```

创建环境：

```powershell
python -m venv .venv
```

激活环境：

```powershell
.\.venv\Scripts\Activate.ps1
```

如果 PowerShell 阻止脚本执行，可在当前终端临时允许：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
```

安装依赖：

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements-ui.txt
```

## 模型文件

为了完全离线运行，模型应放在以下位置：

```text
models
├─ mask_rcnn_molecule.h5
└─ DECIMER-V2
   ├─ DECIMER_model
   └─ DECIMER_HandDrawn_model
```

分割模型文件：

```text
models\mask_rcnn_molecule.h5
```

其官方 MD5 应为：

```text
edd1e6e469cfff7efa6bf8c38441a529
```

如果本地没有识别模型，DECIMER 原始代码会尝试从 Zenodo 下载。桌面安装版已经内置模型，不会执行下载。

## 运行代码

### 启动桌面可视化界面

虚拟环境激活后运行：

```powershell
python decimer_desktop.py
```

也可以直接双击：

```text
run_ui.bat
```

### 仅使用 DECIMER 识别单张图片

```powershell
python -c "from DECIMER import predict_SMILES; print(predict_SMILES(r'图片路径.png'))"
```

Python 示例：

```python
from DECIMER import predict_SMILES

smiles = predict_SMILES("chemical_structure.png")
print(smiles)
```

手绘结构：

```python
from DECIMER import predict_SMILES

smiles = predict_SMILES("hand_drawn_structure.png", hand_drawn=True)
print(smiles)
```

### 仅运行分割模型

```python
from pathlib import Path

import cv2
from decimer_segmentation import segment_chemical_structures_from_file

input_image = "paper_page.png"
output_dir = Path("segmented_structures")
output_dir.mkdir(parents=True, exist_ok=True)

segments = segment_chemical_structures_from_file(input_image, expand=True)

for index, segment in enumerate(segments, start=1):
    output_file = output_dir / f"structure_{index:03d}.png"
    cv2.imwrite(str(output_file), segment)

print(f"共分割出 {len(segments)} 个结构")
```

## 标准测试数据

单结构识别测试：

```text
validation_samples\inputs
```

分割前页面和分割结果：

```text
segmentation_validation\inputs
segmentation_validation\outputs
```

运行原项目测试：

```powershell
python -m pytest tests -v
```

咖啡因快速检查：

```powershell
python -c "from DECIMER import predict_SMILES; print(predict_SMILES(r'tests\caffeine.png'))"
```

预期输出：

```text
CN1C=NC2=C1C(=O)N(C)C(=O)N2C
```

## 构建 Windows EXE

项目使用 PyInstaller 构建目录式 EXE。模型和依赖体积较大，不建议制作每次启动都需要解压的单文件 EXE。

安装 PyInstaller：

```powershell
python -m pip install pyinstaller
```

运行构建脚本：

```powershell
powershell -ExecutionPolicy Bypass -File .\build_windows.ps1
```

构建完成后程序位于：

```text
dist\DECIMER Desktop\DECIMER Desktop.exe
```

安装包位于：

```text
release\DECIMER-Desktop-Setup.exe
```

详细打包说明见：

```text
PACKAGING_README_zh.md
```

## 常见问题

### 拖入图片后文字显示为方块

程序会主动加载 Windows 自带的微软雅黑字体。请关闭旧窗口并重新启动最新版本。

### `smiles_results.csv` PermissionError

该文件可能正被 Excel 占用。关闭 Excel 后重试。程序也会自动改用带时间戳的新文件名保存结果。

### 分割模型下载失败

确认以下文件已经存在：

```text
models\mask_rcnn_molecule.h5
```

并检查 MD5 是否正确。

### 为什么参考 SMILES 和结果字符串不完全一样

SMILES 字符串并不唯一。同一个分子可以因原子遍历顺序、芳环起点和闭环编号不同而产生不同的合法字符串。严格比较时应先进行分子结构规范化，不能只比较原始字符串。

## Git 版本

当前项目保存了以下标签：

```text
版本1
版本2
```

查看版本：

```powershell
git log --oneline --decorate
git tag
```

切换到版本2：

```powershell
git switch --detach "版本2"
```

## 许可证与来源

- DECIMER Image Transformer：MIT License；
- DECIMER Image Segmentation：MIT License；
- 分割模型：CC BY 4.0；
- 使用或再分发时请保留原作者、论文和许可证信息。

---

## Original DECIMER Image Transformer README

以下为上游 DECIMER Image Transformer 项目的原始说明。

<div align="center">

# 🧪 DECIMER Image Transformer 🖼️

### Deep Learning for Chemical Image Recognition using Efficient-Net V2 + Transformer

<p align="center">
  <img src="https://github.com/Kohulan/DECIMER-Image_Transformer/blob/master/DECIMER_V2.png?raw=true" alt="DECIMER Logo" width="600">
</p>

[![License](https://img.shields.io/badge/License-MIT%202.0-blue.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg?style=for-the-badge)](https://GitHub.com/Kohulan/DECIMER-Image_Transformer/graphs/commit-activity)
[![GitHub issues](https://img.shields.io/github/issues/Kohulan/DECIMER-Image_Transformer.svg?style=for-the-badge)](https://GitHub.com/Kohulan/DECIMER-Image_Transformer/issues/)
[![GitHub contributors](https://img.shields.io/github/contributors/Kohulan/DECIMER-Image_Transformer.svg?style=for-the-badge)](https://GitHub.com/Kohulan/DECIMER-Image_Transformer/graphs/contributors/)
[![tensorflow](https://img.shields.io/badge/TensorFlow-2.10.1-FF6F00.svg?style=for-the-badge&logo=tensorflow)](https://www.tensorflow.org)
[![Model Card](https://img.shields.io/badge/Model_Card-DECIMER-9cf.svg?style=for-the-badge)](https://zenodo.org/records/8300489)
[![DOI](https://zenodo.org/badge/293572361.svg)](https://zenodo.org/badge/latestdoi/293572361)
[![Documentation Status](https://readthedocs.org/projects/decimer-image-transformer/badge/?version=latest&style=for-the-badge)](https://decimer-image-transformer.readthedocs.io/en/latest/?badge=latest)
[![GitHub release](https://img.shields.io/github/release/Kohulan/DECIMER-Image_Transformer.svg?style=for-the-badge)](https://GitHub.com/Kohulan/DECIMER-Image_Transformer/releases/)
[![PyPI version fury.io](https://badge.fury.io/py/decimer.svg?style=for-the-badge)](https://pypi.python.org/pypi/decimer/)

</div>

---

## 📚 Table of Contents

- [📝 Abstract](#-abstract)
- [💡 Method and Model Changes](#-method-and-model-changes)
- [⚙️ Installation](#️-installation)
- [🚀 Usage](#-usage)
- [✍️ Hand-drawn Model](#️-decimer---hand-drawn-model)
- [📄 Citation](#-citation)
- [🙏 Acknowledgements](#-acknowledgements)
- [👨‍🔬 Author](#-author-kohulan)
- [🌐 Project Website](#-project-website)
- [🏛️ Research Group](#️-research-group)

---

## 📝 Abstract

<div align="center">
  <img src="https://github.com/Kohulan/DECIMER-Image-to-SMILES/raw/master/assets/DECIMER.gif" width="350" align="right">
</div>

> The DECIMER 2.2 project tackles the OCSR (Optical Chemical Structure Recognition) challenge using cutting-edge computational intelligence methods. Our goal? To provide an automated, open-source software solution for chemical image recognition.
> 
> We've supercharged DECIMER with Google's TPU (Tensor Processing Unit) to handle datasets of over 1 million images with lightning speed!

---

## 💡 Method and Model Changes

<table>
  <tr>
    <td width="50%" align="center">
      <h3>🖼️ Image Feature Extraction</h3>
      <p>Now utilizing EfficientNet-V2 for superior image analysis</p>
    </td>
    <td width="50%" align="center">
      <h3>🔮 SMILES Prediction</h3>
      <p>Employing a state-of-the-art transformer model</p>
    </td>
  </tr>
</table>

### 🚀 Training Enhancements

1. **📦 TFRecord Files** - Lightning-fast data reading
2. **☁️ Google Cloud Buckets** - Efficient cloud storage solution
3. **🔄 TensorFlow Data Pipeline** - Optimized data loading
4. **⚡ TPU Strategy** - Harnessing the power of Google's TPUs

---

## ⚙️ Installation

```bash
# Create a conda wonderland
conda create --name DECIMER python=3.10.0 -y
conda activate DECIMER

# Equip yourself with DECIMER
pip install decimer
```

---

## 🚀 Usage

```python
from DECIMER import predict_SMILES

# Unleash the power of DECIMER
image_path = "path/to/your/chemical/masterpiece.jpg"
SMILES = predict_SMILES(image_path)
print(f"🎉 Decoded SMILES: {SMILES}")
```

---

## ✍️ DECIMER - Hand-drawn Model

> 🌟 **New Feature Alert!** 🌟
> 
> Our latest model brings the magic of AI to hand-drawn chemical structures!
> 
> [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10781330.svg)](https://doi.org/10.5281/zenodo.10781330)

---

## 📄 Citation

If DECIMER helps your research, please cite:

1. Rajan K, et al. "DECIMER.ai - An open platform for automated optical chemical structure identification, segmentation and recognition in scientific publications." *Nat. Commun.* 14, 5045 (2023).
2. Rajan, K., et al. "DECIMER 1.0: deep learning for chemical image recognition using transformers." *J Cheminform* 13, 61 (2021).
3. Rajan, K., et al. "Advancements in hand-drawn chemical structure recognition through an enhanced DECIMER architecture," *J Cheminform* 16, 78 (2024).

---

## 🙏 Acknowledgements

- A big thank you to [Charles Tapley Hoyt](https://github.com/cthoyt) for his invaluable contributions!
- Powered by Google's TPU Research Cloud (TRC)

<p align="center">
  <img src="https://user-images.githubusercontent.com/30716951/220350828-913e6645-6a0a-403c-bcb8-160d061d4606.png" width="300">
</p>

---

## 👨‍🔬 Author: [Kohulan](https://kohulanr.com)

---

## 🌐 Project Website

Experience DECIMER in action at [decimer.ai](https://decimer.ai), brilliantly implemented by [Otto Brinkhaus](https://github.com/OBrink)!

---

<div align="center">

### 🎓 Maintained by the [Kohulan](https://www.kohulanr.com/#) @ Steinbeck Group

<a href="https://cheminf.uni-jena.de">
<img src="https://github.com/Kohulan/DECIMER-Image-to-SMILES/blob/master/assets/CheminfGit.png" width="400" alt="Cheminformatics Group"/>
</a>

**[Natural Products Cheminformatics Research Group](https://cheminf.uni-jena.de)**  
Institute for Inorganic and Analytical Chemistry  
Friedrich Schiller University Jena, Germany

---
## ⭐ Star History

<div align="center">

[![Star History Chart](https://api.star-history.com/svg?repos=Kohulan/DECIMER-Image_Transformer&type=Date)](https://star-history.com/#Kohulan/DECIMER-Image_Transformer&Date)

</div>

---

<div align="center">

### 📊 Project Analytics

![Repobeats](https://repobeats.axiom.co/api/embed/bf532b7ac0d34137bdea8fbb82986828f86de065.svg "Repobeats analytics image")

</div>

<div align="center">
  
**Made with ❤️ and ☕ for the global chemistry community**

**© 2025 Kohulan @ Steinbeck Lab, Friedrich Schiller University Jena**
---
