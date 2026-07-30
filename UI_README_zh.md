# 🧪 DECIMER-Image_Transformer-UI

DECIMER 桌面应用 — 化学结构图像识别可视化工具。

## 📖 关于本项目

本项目是 [DECIMER-Image_Transformer](https://github.com/Kohulan/DECIMER-Image_Transformer) 和 [DECIMER-Image-Segmentation](https://github.com/Kohulan/DECIMER-Image-Segmentation) 的 **可视化桌面包装**，提供图形化界面方便使用。

## 🔗 原始项目

| 项目 | 仓库地址 | 论文 |
|------|----------|------|
| **DECIMER-Image_Transformer** | [GitHub](https://github.com/Kohulan/DECIMER-Image_Transformer) | [Nat. Commun. 14, 5045 (2023)](https://doi.org/10.1038/s41467-023-40782-0) |
| **DECIMER-Image-Segmentation** | [GitHub](https://github.com/Kohulan/DECIMER-Image-Segmentation) | — |

### 引用

```bibtex
@article{Rajan2023DECIMER,
  title   = {DECIMER.ai - An open platform for automated optical chemical structure
             identification, segmentation and recognition in scientific publications},
  author  = {Rajan, Kohulan et al.},
  journal = {Nature Communications},
  volume  = {14},
  pages   = {5045},
  year    = {2023}
}

@article{Rajan2021DECIMER,
  title   = {DECIMER 1.0: deep learning for chemical image recognition using transformers},
  author  = {Rajan, Kohulan et al.},
  journal = {Journal of Cheminformatics},
  volume  = {13},
  pages   = {61},
  year    = {2021}
}
```

## 🤖 模型下载

DECIMER 首次运行时会自动下载模型，也可手动下载：

| 模型 | 大小 | 地址 |
|------|------|------|
| **DECIMER V2 (主模型)** | ~332 MB | [Zenodo](https://zenodo.org/records/8300489) |
| **DECIMER Hand-Drawn (手绘)** | ~332 MB | [Zenodo](https://zenodo.org/records/10781330) |

模型存放路径：`~/.data/DECIMER-V2/`

## 🚀 运行

```bash
# 安装依赖
pip install -r requirements.txt
pip install -r requirements-ui.txt

# 启动 GUI
python decimer_desktop.py

# 或双击
run_ui.bat
```

## 📦 打包

```bash
pip install pyinstaller
pyinstaller decimer_desktop.spec
```

## 📄 许可证

本项目沿用原始项目的 MIT 许可证。
