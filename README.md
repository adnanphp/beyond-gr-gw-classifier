# 🌌 Beyond-GR Gravitational-Wave Classifier

> Deep learning detection of Beyond-General-Relativity deviations in gravitational-wave signals using synthetic aLIGO-PSD noise and real LIGO H1 detector strain.

[![Paper](https://img.shields.io/badge/arXiv-2609.19416-b31b1b.svg)](https://arxiv.org/abs/2609.19416)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![PyCBC](https://img.shields.io/badge/PyCBC-Gravitational%20Waves-orange)](https://pycbc.org/)
[![GWpy](https://img.shields.io/badge/GWpy-LIGO%20Analysis-purple)](https://gwpy.github.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Paper:** [arXiv:2609.19416](https://arxiv.org/abs/2609.19416)

---

## 🔬 Overview

This repository contains the complete pipeline for generating, preprocessing, training, and evaluating a hybrid neural-network classifier designed to distinguish:

* **General Relativity (GR)** gravitational-wave waveforms
* **Beyond-General-Relativity (Beyond-GR)** waveforms containing controlled deviations

The classifier combines a **1D convolutional neural network (CNN)** with **10 hand-crafted waveform statistics**.

The pipeline supports both:

* Synthetic noise generated from the Advanced LIGO PSD
* Real LIGO H1 detector strain

The project investigates how detectability changes with **signal-to-noise ratio (SNR)**, deviation strength **β**, deviation type, and waveform parameters.

---

## ✨ Key Results

### Detection and Generalization

| Experiment                        |                            Result |
| --------------------------------- | --------------------------------: |
| Synthetic noise, β ∈ [0.3, 1.0]   |   Detection threshold at SNR ≈ 20 |
| Real LIGO noise, β ∈ [5, 20]      |         >96% accuracy at SNR 5–10 |
| Real LIGO noise, β ∈ [0.3, 0.5]   |              Threshold at β ≈ 0.4 |
| GW150914 template, β ∈ [0.3, 0.4] |        90.0% test-unseen accuracy |
| Unseen deviation type             | Accurate generalization to type C |
| Mass-ratio study                  |      89–93% accuracy for q = 1–10 |

These results are specific to the waveform modifications, noise conditions, SNR ranges, and evaluation protocol defined in this study.

---

# 🧠 Method

## Hybrid CNN + Statistical Features

The classifier uses two complementary branches.

### 1. 1D CNN

A three-block 1D convolutional network processes the **16,384-sample waveform**.

```text
Waveform
   │
   ▼
Conv Block 1 — stride 8
   │
   ▼
Conv Block 2 — stride 2
   │
   ▼
Conv Block 3 — stride 2
   │
   ▼
Learned waveform representation
```

The CNN is designed to learn local waveform structure directly from the time-domain signal.

### 2. Hand-Crafted Statistics

A second branch computes 10 waveform statistics:

```text
Mean absolute amplitude
Standard deviation
Maximum
P95
P99
P99.9
P99.99
RMS
Skewness
Kurtosis
```

These features are passed through a two-layer MLP.

### 3. Feature Fusion

```text
             Waveform
                 │
                 ▼
            1D CNN
                 │
                 │
                 ├──────────────┐
                 │              │
                 ▼              ▼
          CNN representation   Statistics
                                │
                                ▼
                               MLP
                 │              │
                 └──────┬───────┘
                        ▼
                 Feature Fusion
                        │
                        ▼
                 Classification Head
                        │
                        ▼
                  GR / Beyond-GR
```

**Total trainable parameters:** 31,697

---

# 🌀 Beyond-GR Deviations

Three controlled deviation families are implemented.

### Amplitude Modulation

$$
h(t) \rightarrow h(t)\left(1+\beta\tau^2\right)
$$

### Phase Modulation

$$
h(t) \rightarrow A(t)\cos\left(\phi(t)+\beta\tau^2\right)
$$

### Frequency Modulation

$$
\phi(t)\rightarrow
\phi(t)+2\pi\beta f_{\mathrm{ref}}\frac{\tau^2}{2}
$$

where:

* **β** is the deviation strength
* **τ ∈ [0,1]** is normalized time from inspiral to merger
* \(A(t)\) is the waveform amplitude
* \(\phi(t)\) is the waveform phase
* \(f_{\mathrm{ref}}\) is the reference frequency

The experiments vary β to study the transition from difficult-to-detect to readily detectable deviations.

---

# 📊 Results

## Main Experimental Results

| Experiment                         | Deviation | Noise     |   SNR | Test-Unseen Accuracy |
| ---------------------------------- | --------- | --------- | ----: | -------------------: |
| Synthetic noise, β ∈ [0.3, 1.0]    | Amplitude | Synthetic | 30–50 |            **1.000** |
| Synthetic noise, β ∈ [0.3, 1.0]    | Amplitude | Synthetic | 10–15 |            **0.997** |
| Real LIGO, β ∈ [5, 20]             | Amplitude | Real      |  5–10 |            **0.968** |
| Real LIGO, β ∈ [0.3, 0.5]          | Amplitude | Real      | 10–15 |            **0.891** |
| Real LIGO, β ∈ [0.5, 2.0]          | Phase     | Real      | 10–15 |            **0.930** |
| Real LIGO, β ∈ [5, 25]             | Frequency | Real      | 10–15 |            **0.885** |
| GW150914 template, β ∈ [0.3, 0.4]  | Amplitude | Real      |    20 |            **0.900** |
| GW150914 template, β ∈ [0.01, 0.2] | Amplitude | Real      |    20 |            **0.517** |

---

# 🎯 Detection Threshold

A central experiment measures test accuracy as a function of the deviation strength **β**.

For the **GW150914 template at SNR 20**:

| β Range  | Test-Unseen Accuracy |
| -------- | -------------------: |
| 0.01–0.2 |                0.517 |
| 0.2–0.3  |                0.750 |
| 0.3–0.4  |                0.900 |
| 0.4–0.5  |                0.967 |
| 0.5–1.0  |                1.000 |
| 5–20     |                1.000 |

The observed detection transition occurs around:

> **β ≈ 0.25**

This value is specific to the **quadratic-in-time modulation model adopted in this work**. It should not be interpreted as a generic observational constraint on Beyond-GR parameters.

---

# 🧪 Experimental Pipeline

```text
GR Waveforms ───────────────┐
                            │
                            ▼
                    Waveform Generation
                            │
                            ▼
                   Beyond-GR Modification
                            │
                            ▼
                  Noise Generation / Loading
                            │
                            ▼
                       SNR Injection
                            │
                            ▼
                       Preprocessing
                   ┌────────┼────────┐
                   │        │        │
              Whitening  Bandpass  Normalize
                   └────────┼────────┘
                            ▼
                     Dataset Splitting
                            │
                            ▼
                  Hybrid CNN + Statistics
                            │
                            ▼
                       Evaluation
                   ┌────────┼────────┐
                   │        │        │
                Accuracy   AUC      F1
```

---

# 🔊 Noise Sources

The project supports two noise modes.

### Synthetic Noise

Noise generated using an Advanced LIGO power spectral density (PSD).

### Real LIGO Noise

Real detector strain from **LIGO Hanford (H1)** is used for additional evaluation.

This allows the classifier to be tested under both controlled synthetic conditions and real detector noise.

---

# 📁 Repository Structure

```text
beyond-gr-classifier/
│
├── src/
│   ├── data/
│   │   ├── generate_waveforms.py
│   │   │   └── IMRPhenomD waveform generation
│   │   ├── inject_deviations.py
│   │   │   └── Amplitude, phase, frequency modulation
│   │   ├── generate_noise.py
│   │   │   └── Synthetic + real LIGO noise
│   │   ├── inject_signal.py
│   │   │   └── Signal-to-noise injection
│   │   ├── preprocessing.py
│   │   │   └── Whitening, bandpass, normalization
│   │   ├── build_dataset.py
│   │   │   └── Dataset-generation pipeline
│   │   └── split.py
│   │       └── Physics-based train/validation/test split
│   │
│   ├── models/
│   │   └── cnn1d.py
│   │       └── Hybrid CNN + statistics classifier
│   │
│   ├── training/
│   │   └── train.py
│   │       └── Training loop with early stopping
│   │
│   └── evaluation/
│       └── metrics.py
│           └── Accuracy, ROC-AUC, F1
│
├── scripts/
│   ├── step7_train_cnn.py
│   ├── step9_evaluate.py
│   └── build_gw150914_dataset.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── metadata/
│
├── models/
│
├── results/
│
├── paper/
│   ├── main.tex
│   ├── sections/
│   └── references.bib
│
├── config.yaml
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

The recommended environment uses **Python 3.11**, PyCBC, LALSuite, and GWpy.

### 1. Create Conda Environment

```bash
conda create -n beyondgr python=3.11 -y
conda activate beyondgr
```

### 2. Install Scientific Packages

```bash
conda install -c conda-forge \
  "numpy=1.26" \
  "scipy<1.14" \
  pycbc \
  lalsuite \
  gwpy \
  -y
```

### 3. Install ML and Utility Packages

```bash
pip install \
  torch \
  scikit-learn \
  pandas \
  tqdm \
  pyyaml \
  matplotlib \
  pytest
```

### ⚠️ NumPy Compatibility

PyCBC requires **NumPy < 2.0** for this environment.

Do not upgrade NumPy to version 2.x after installation.

---

# 🚀 Quick Start

## 1. Generate a Dataset

Generate a small dataset using synthetic noise:

```bash
python -m src.data.build_dataset \
  --n_per_class 200 \
  --snr_min 15 \
  --snr_max 20 \
  --noise_source synthetic
```

## 2. Train the Classifier

```bash
python scripts/step7_train_cnn.py \
  --epochs 200 \
  --lr 0.003
```

## 3. Evaluate

```bash
python scripts/step9_evaluate.py
```

---

# 🌌 GW150914 Study

The repository also includes an experiment using a **GW150914 template**.

### Download the GWOSC Data

```bash
mkdir -p data/raw/gw150914

cd data/raw/gw150914

curl -O -L \
"https://www.gw-openscience.org/eventapi/html/GWTC-1-confident/GW150914/v3/H-H1_GWOSC_4KHZ_R1-1126259447-32.hdf5"

cd ../../..
```

### Build the Dataset

```bash
python scripts/build_gw150914_dataset.py
```

### Train and Evaluate

```bash
python scripts/step7_train_cnn.py \
  --epochs 200 \
  --lr 0.003

python scripts/step9_evaluate.py
```

---

# ⚙️ Configuration

Experiments are controlled through `config.yaml`.

Important sections include:

| Section         | Purpose                                        |
| --------------- | ---------------------------------------------- |
| `waveform`      | Mass ranges, sample rate, waveform approximant |
| `deviation`     | Deviation type and β ranges                    |
| `noise`         | Synthetic/real noise and SNR range             |
| `preprocessing` | Whitening, bandpass, normalization             |
| `training`      | Learning rate, epochs, batch size              |

This allows experiments to be reproduced without modifying the core source code.

---

# 📝 Research Paper

The LaTeX manuscript is included in:

```text
paper/
```

Compile with:

```bash
cd paper

pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The resulting PDF will be:

```text
paper/main.pdf
```

**Paper:** [arXiv:2609.19416](https://arxiv.org/abs/2609.19416)

---

# 📦 Requirements

| Package      | Version |
| ------------ | ------- |
| Python       | 3.11+   |
| PyCBC        | ≥ 2.0   |
| GWpy         | ≥ 3.0   |
| PyTorch      | ≥ 2.0   |
| NumPy        | 1.26    |
| SciPy        | < 1.14  |
| scikit-learn | ≥ 1.3   |

---

# 🔭 Research Directions

Possible extensions include:

* [ ] Expand the training set with additional detector noise
* [ ] Evaluate multiple LIGO/Virgo/KAGRA detector configurations
* [ ] Investigate additional Beyond-GR deviation families
* [ ] Explore transformer-based waveform architectures
* [ ] Compare CNN-only and statistics-only baselines
* [ ] Perform broader mass-ratio generalization studies
* [ ] Investigate robustness across different SNR distributions
* [ ] Calibrate probabilistic detection thresholds
* [ ] Evaluate performance on larger real-detector datasets

---

# ⚠️ Scientific Scope

The β thresholds reported here characterize the **specific phenomenological waveform modifications and experimental setup used in this repository**.

They are **not generic constraints on modified-gravity theories or Beyond-GR parameters**.

The results should therefore be interpreted as a machine-learning detectability study rather than as an observational exclusion or discovery constraint.

---

# 📚 Acknowledgements

This project uses gravitational-wave data and open-source scientific software from:

* **Gravitational-Wave Open Science Center (GWOSC)**
* **LALSuite**
* **PyCBC**
* **GWpy**

---

# 📄 License

This project is released under the **MIT License**.

See [LICENSE](LICENSE) for details.

---

<div align="center">

### 🌌 Beyond General Relativity × Deep Learning

**Gravitational Waves • Physics-Informed ML • Computer Vision & Signal Processing • Scientific Machine Learning**

**Paper:** [arXiv:2609.19416](https://arxiv.org/abs/2609.19416)

</div>
