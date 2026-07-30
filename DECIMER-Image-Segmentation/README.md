<div align="center">

# 🔬 DECIMER Image Segmentation 📄

### Deep Learning for Chemical Image Recognition - Automated Structure Detection & Extraction

<p align="center">
  <img src="https://github.com/Kohulan/DECIMER-Image-Segmentation/blob/master/Validation/Abstract1.png?raw=true" alt="DECIMER Segmentation" width="700">
</p>

[![License](https://img.shields.io/badge/License-MIT%202.0-blue.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg?style=for-the-badge)](https://GitHub.com/Kohulan/DECIMER-Image-Segmentation/graphs/commit-activity)
[![GitHub issues](https://img.shields.io/github/issues/Kohulan/DECIMER-Image-Segmentation.svg?style=for-the-badge)](https://GitHub.com/Kohulan/DECIMER-Image-Segmentation/issues/)
[![GitHub contributors](https://img.shields.io/github/contributors/Kohulan/DECIMER-Image-Segmentation.svg?style=for-the-badge)](https://GitHub.com/Kohulan/DECIMER-Image-Segmentation/graphs/contributors/)
[![tensorflow](https://img.shields.io/badge/TensorFlow-2.10.1-FF6F00.svg?style=for-the-badge&logo=tensorflow)](https://www.tensorflow.org)
[![Model Card](https://img.shields.io/badge/Model_Card-Mask_RCNN-9cf.svg?style=for-the-badge)](https://zenodo.org/badge/latestdoi/268631290)
[![DOI](https://zenodo.org/badge/268631290.svg)](https://zenodo.org/badge/latestdoi/268631290)
[![GitHub release](https://img.shields.io/github/release/Kohulan/DECIMER-Image-Segmentation.svg?style=for-the-badge)](https://GitHub.com/Kohulan/DECIMER-Image-Segmentation/releases/)
[![PyPI version fury.io](https://badge.fury.io/py/decimer-segmentation.svg?style=for-the-badge)](https://pypi.python.org/pypi/decimer-segmentation/)

**🌐 Try it live at [decimer.ai](https://decimer.ai)**

</div>

---

## 📚 Table of Contents

- [📝 Overview](#-overview)
- [✨ Key Features](#-key-features)
- [🎯 How It Works](#-how-it-works)
- [⚙️ Installation](#️-installation)
- [🚀 Usage](#-usage)
  - [Command Line](#command-line-interface)
  - [Python API](#python-api)
  - [Windows Users](#-notes-for-windows-users)
- [📊 Model Information](#-model-information)
- [📄 Citation](#-citation)
- [🙏 Acknowledgements](#-acknowledgements)
- [👥 Authors](#-authors)
- [🌐 Project Website](#-project-website)
- [🏛️ Research Group](#️-research-group)

---

## 📝 Overview

<div align="center">
  <img src="https://github.com/Kohulan/DECIMER-Image-to-SMILES/raw/master/assets/DECIMER.gif" width="350" align="right">
</div>

> **Unlocking decades of chemical knowledge from scientific literature!**
>
> Chemistry has accumulated vast amounts of knowledge about chemical compounds, structures, and properties across countless scientific publications. DECIMER Segmentation is the first open-source, deep learning-based tool designed to automatically recognize and extract chemical structure depictions from scientific documents.

### 🎯 The Challenge

Converting images of chemical structures into machine-readable formats (OCSR - Optical Chemical Structure Recognition) is a crucial step in digitizing chemical knowledge. But before we can recognize structures, we need to find and extract them from complex document pages!

### 💡 The Solution

DECIMER Segmentation uses advanced deep learning to:
- 🔍 **Detect** chemical structure depictions in scientific publications
- ✂️ **Extract** individual structure images with precision
- 📚 **Process** both modern PDFs and scanned historical documents
- ⚡ **Automate** the entire workflow from document to segmented structures

---

## ✨ Key Features

<table>
  <tr>
    <td width="33%" align="center">
      <h3>🤖 Deep Learning Powered</h3>
      <p>Built on Mask R-CNN architecture for state-of-the-art detection accuracy</p>
    </td>
    <td width="33%" align="center">
      <h3>📖 Universal Compatibility</h3>
      <p>Works with PDFs, scanned pages, and bitmap images from any publisher</p>
    </td>
    <td width="33%" align="center">
      <h3>🆓 Open Source</h3>
      <p>Freely available code and pre-trained models for the community</p>
    </td>
  </tr>
  <tr>
    <td width="33%" align="center">
      <h3>⚡ High Performance</h3>
      <p>GPU acceleration support for rapid batch processing</p>
    </td>
    <td width="33%" align="center">
      <h3>🎨 Smart Post-Processing</h3>
      <p>Automatic mask expansion to capture complete structures</p>
    </td>
    <td width="33%" align="center">
      <h3>🌐 Web Application</h3>
      <p>User-friendly interface available at decimer.ai</p>
    </td>
  </tr>
</table>

---

## 🎯 How It Works

DECIMER Segmentation employs a sophisticated two-stage workflow:

### 1️⃣ **Detection Stage**
```
📄 Input Document → 🤖 Mask R-CNN Model → 🎭 Structure Masks
```
The deep learning model analyzes the page and creates precise masks indicating the location of each chemical structure.

### 2️⃣ **Post-Processing Stage**
```
🎭 Initial Masks → 🔧 Expansion Algorithm → ✅ Complete Structures
```
An intelligent post-processing workflow ensures that potentially incomplete masks are expanded to capture the full structure.

### 🎨 Visual Workflow

```
┌─────────────────┐
│  PDF/Image File │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Page Extraction│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Mask R-CNN      │
│ Detection       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Mask Expansion  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Segmented       │
│ Structures      │
└─────────────────┘
```

---

## ⚙️ Installation

### 🐍 Prerequisites

We strongly recommend using a Conda environment for seamless dependency management.

#### Install Miniconda (if not already installed)

```bash
# Linux
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# macOS
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
bash Miniconda3-latest-MacOSX-x86_64.sh
```

### 📦 Installation Options

<details>
<summary><b>Option 1: Install from GitHub (Development Version)</b></summary>

```bash
# Clone the repository
git clone https://github.com/Kohulan/DECIMER-Image-Segmentation.git
cd DECIMER-Image-Segmentation

# Create and activate conda environment
conda create --name DECIMER_IMGSEG python=3.10
conda activate DECIMER_IMGSEG

# Install dependencies
conda install pip
python -m pip install -U pip

# Install DECIMER-Segmentation
pip install .

# Install Poppler (required for PDF processing)
conda install -c conda-forge poppler
```

</details>

<details>
<summary><b>Option 2: Install from PyPI (Stable Release)</b></summary>

```bash
# Create and activate conda environment
conda create --name DECIMER_IMGSEG python=3.10
conda activate DECIMER_IMGSEG

# Install from PyPI
pip install decimer-segmentation

# Install Poppler (required for PDF processing)
conda install -c conda-forge poppler
```

</details>

### 🖥️ Hardware Requirements

- **CPU Mode**: Works on any modern CPU
- **GPU Mode** *(Recommended)*: CUDA-compatible GPU with appropriate drivers
  - Significantly faster processing
  - Essential for batch processing

---

## 🚀 Usage

### Command Line Interface

Process entire documents with a single command:

```bash
# Segment structures from a PDF or image file
python3 segment_structures_in_document.py your_document.pdf

# Output will be saved in a folder named after your input file
# e.g., your_document/ containing all segmented structures
```

### Python API

#### 🎨 **Example 1: Segment from Image Array**

```python
from decimer_segmentation import segment_chemical_structures
import cv2

# Load your scanned page
page_image = cv2.imread("path/to/scanned_page.png")

# Extract all chemical structures
segments = segment_chemical_structures(page_image, expand=True)

# segments is a list of numpy arrays, each containing a structure
for idx, structure in enumerate(segments):
    cv2.imwrite(f"structure_{idx}.png", structure)
    print(f"✅ Saved structure {idx}")
```

#### 📄 **Example 2: Segment from File (PDF or Image)**

```python
from decimer_segmentation import segment_chemical_structures_from_file

# Process a PDF file
segments = segment_chemical_structures_from_file(
    "path/to/document.pdf",
    expand=True
)

# Process an image file
segments = segment_chemical_structures_from_file(
    "path/to/page_image.jpg",
    expand=True
)

print(f"🎉 Extracted {len(segments)} chemical structures!")
```

#### 🔧 **Example 3: Batch Processing**

```python
from decimer_segmentation import segment_chemical_structures_from_file
import os
from pathlib import Path

def batch_segment(input_dir, output_dir):
    """Process multiple PDF files"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    for pdf_file in Path(input_dir).glob("*.pdf"):
        print(f"📄 Processing {pdf_file.name}...")
        
        segments = segment_chemical_structures_from_file(
            str(pdf_file),
            expand=True
        )
        
        # Save each segment
        file_output_dir = Path(output_dir) / pdf_file.stem
        file_output_dir.mkdir(exist_ok=True)
        
        for idx, segment in enumerate(segments):
            output_path = file_output_dir / f"structure_{idx:03d}.png"
            cv2.imwrite(str(output_path), segment)
        
        print(f"✅ Extracted {len(segments)} structures from {pdf_file.name}")

# Use it
batch_segment("input_pdfs/", "output_structures/")
```

#### 🎯 **Example 4: Advanced Usage with Custom Parameters**

```python
from decimer_segmentation import segment_chemical_structures
import cv2

# Load image
image = cv2.imread("complex_page.png")

# Segment with custom settings
segments = segment_chemical_structures(
    image,
    expand=True,          # Enable mask expansion
    visualization=True    # Generate visualization (if available)
)

# Process results
for idx, segment in enumerate(segments):
    # You can now pass this to DECIMER Image Transformer
    # for structure recognition
    print(f"Structure {idx}: {segment.shape}")
```

### 📓 Interactive Tutorial

For more comprehensive examples and interactive demonstrations, check out our **[Jupyter Notebook](https://github.com/Kohulan/DECIMER-Image-Segmentation/blob/master/DECIMER_Segmentation_notebook.ipynb)**!

---

### 🪟 Notes for Windows Users

<details>
<summary><b>Windows-Specific Instructions</b></summary>

#### 1️⃣ Use Anaconda PowerShell Prompt
Run all commands in the **Anaconda PowerShell Prompt** (not regular Command Prompt or PowerShell).

#### 2️⃣ Install Poppler for PDF Support

PDF processing on Windows requires Poppler. Follow these steps:

1. **Download Poppler**
   - Visit [Poppler for Windows](http://blog.alivate.com.au/poppler-windows/)
   - Download and extract to a location like `C:\Program Files\poppler`

2. **Specify Poppler Path in Code**
   ```python
   from decimer_segmentation import segment_chemical_structures_from_file
   
   segments = segment_chemical_structures_from_file(
       "document.pdf",
       expand=True,
       poppler_path=r"C:\Program Files\poppler\Library\bin"
   )
   ```

#### 3️⃣ GPU Support on Windows
Ensure you have:
- CUDA Toolkit installed
- cuDNN libraries configured
- Compatible GPU drivers

</details>

---

## 📊 Model Information

### 🤖 Pre-trained Model

The Mask R-CNN model is publicly available and ready to use:

**[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10142866.svg)](https://doi.org/10.5281/zenodo.10142866)**

### 🎓 Model Architecture

- **Base Network**: Mask R-CNN
- **Training Data**: Diverse chemical literature from multiple publishers
- **Task**: Instance segmentation of chemical structure depictions
- **Performance**: Manually validated on publications from various sources

### 🔍 Model Performance

The model has been rigorously evaluated on:
- ✅ Publications from multiple scientific publishers
- ✅ Documents spanning different time periods
- ✅ Both modern PDFs and scanned historical pages
- ✅ Various image qualities and layouts

---

## 📄 Citation

If DECIMER Segmentation contributes to your research, please cite:

```bibtex
@article{Rajan2021,
  author = {Rajan, Kohulan and Brinkhaus, Henning Otto and Sorokina, Maria and Zielesny, Achim and Steinbeck, Christoph},
  title = {DECIMER-Segmentation: Automated extraction of chemical structure depictions from scientific literature},
  journal = {Journal of Cheminformatics},
  year = {2021},
  volume = {13},
  number = {20},
  doi = {10.1186/s13321-021-00496-1}
}
```

**Full Citation:**  
Rajan, K., Brinkhaus, H.O., Sorokina, M. et al. *DECIMER-Segmentation: Automated extraction of chemical structure depictions from scientific literature.* J Cheminform **13**, 20 (2021). https://doi.org/10.1186/s13321-021-00496-1

---

## 🙏 Acknowledgements

<div align="center">

### 🌟 Special Thanks

This project wouldn't be possible without the support and contributions from the community and funding organizations.

<table>
  <tr>
    <td align="center">
      <strong>Contributors</strong><br>
      All our amazing contributors who helped improve the codebase
    </td>
    <td align="center">
      <strong>Community</strong><br>
      Users providing feedback and reporting issues
    </td>
    <td align="center">
      <strong>Open Source</strong><br>
      Projects we build upon: TensorFlow, Mask R-CNN
    </td>
  </tr>
</table>

</div>

## 🌐 Project Website

<div align="center">

### Experience DECIMER Live!

<a href="https://decimer.ai">
  <img src="https://github.com/Kohulan/DECIMER-Image_Transformer/blob/master/DECIMER_V2.png?raw=true" width="600" alt="DECIMER.ai">
</a>

**[🚀 Try DECIMER.ai](https://decimer.ai)** - Web application.

### 📦 Complete DECIMER Suite

DECIMER Segmentation is part of a comprehensive chemical structure recognition pipeline:

1. **🔍 [DECIMER Segmentation](https://github.com/Kohulan/DECIMER-Image-Segmentation)** *(You are here)*  
   Extract chemical structures from documents

2. **🧠 [DECIMER Image Transformer](https://github.com/Kohulan/DECIMER-Image_Transformer)**  
   Convert structure images to SMILES strings

3. **🗄️ [MARCUS](https://marcus.decimer.ai/)**  
   Molecular Annotation and Recognition for Curating Unravelled Structures

</div>

---

## 🏛️ Research Group

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

[![Star History Chart](https://api.star-history.com/svg?repos=Kohulan/DECIMER-Image-Segmentation&type=Date)](https://star-history.com/#Kohulan/DECIMER-Image-Segmentation&Date)

</div>

---

## 📊 Project Analytics

<div align="center">

![Repobeats](https://repobeats.axiom.co/api/embed/5a62f88de9624eca3a4bbfbdde6126fb8fb4c65d.svg "Repobeats analytics image")

</div>

---

<div align="center">

### 🤝 Contributing

We welcome contributions! Please feel free to submit a Pull Request.

**[📝 Report Bug](https://github.com/Kohulan/DECIMER-Image-Segmentation/issues)** · **[💡 Request Feature](https://github.com/Kohulan/DECIMER-Image-Segmentation/issues)** · **[⭐ Star this repo](https://github.com/Kohulan/DECIMER-Image-Segmentation)**

---

**Made with ❤️ and ☕ for the global chemistry community**

**© 2025 Kohulan @ Steinbeck Lab, Friedrich Schiller University Jena**

---

<sub>🔬 Advancing Open Science in Chemistry | 🌍 Digitizing Chemical Knowledge | 🤖 Powered by Deep Learning</sub>

</div>
