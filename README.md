# Beyond-GR Gravitational-Wave Classifier

Deep learning detection of Beyond-General-Relativity deviations in
gravitational-wave signals, using synthetic aLIGO-PSD noise and real
LIGO H1 detector strain.

## Overview

This repository contains the full pipeline for training and evaluating
a hybrid neural-network classifier (1D CNN + hand-crafted statistics)
that distinguishes General-Relativistic gravitational waveforms from
systematically modified (Beyond-GR) waveforms.

Three deviation families are supported:
- **Amplitude modulation:** `h(t) → h(t)·(1 + β·τ²)`
- **Phase modulation:** `h(t) → A(t)·cos(φ(t) + β·τ²)`
- **Frequency modulation:** `φ(t) → φ(t) + 2π·β·f_ref·τ²/2`

where `β` is the deviation strength and `τ ∈ [0,1]` is normalized
time from inspiral to merger.

## Key Results

- **Synthetic noise, β ∈ [0.3, 1.0]:** detection threshold at SNR ≈ 20
- **Real LIGO noise, β ∈ [5, 20]:** >96% accuracy down to SNR 5–10
- **Real LIGO noise, β ∈ [0.3, 0.5]:** threshold at β ≈ 0.4
- **GW150914 template, amplitude-controlled:** threshold at β ≈ 0.25
- **Generalization:** accurate on unseen deviation types (type C)
- **Mass ratios:** 89–93% accuracy across q = 1 to 10

## Repository Structure
src/
data/
generate_waveforms.py IMRPhenomD waveform generation
inject_deviations.py Amplitude, phase, frequency modulation
generate_noise.py Synthetic + real LIGO noise
inject_signal.py Signal-to-noise injection
preprocessing.py Whitening, bandpass, normalization
build_dataset.py Full data-generation pipeline
split.py Physics-based train/val/test split
models/
cnn1d.py Hybrid CNN + statistics classifier
training/
train.py Training loop with early stopping
evaluation/
metrics.py Accuracy, ROC-AUC, F1

scripts/
step7_train_cnn.py Train the classifier
step9_evaluate.py Evaluate on all splits
build_gw150914_dataset.py GW150914 template study

paper/
main.tex Full LaTeX manuscript
sections/ Paper sections
references.bib Bibliography

text

## Installation

Requires Python 3.11+ with PyCBC, GWpy, PyTorch.

```bash
# Create a fresh conda environment
conda create -n beyondgr python=3.11 -y
conda activate beyondgr

# Install scientific packages from conda-forge (keeps versions consistent)
conda install -c conda-forge "numpy=1.26" "scipy<1.14" pycbc lalsuite gwpy -y

# Install ML stack
pip install torch scikit-learn pandas tqdm pyyaml matplotlib pytest
Important: PyCBC requires NumPy < 2.0. Do not upgrade NumPy after installation.

Quick Start
1. Generate a small dataset
bash
python -m src.data.build_dataset --n_per_class 200 --snr_min 15 --snr_max 20 --noise_source synthetic
2. Train the classifier
bash
python scripts/step7_train_cnn.py --epochs 200 --lr 0.003
3. Evaluate
bash
python scripts/step9_evaluate.py
4. GW150914 template study
Download the GWOSC raw file for GW150914:

bash
mkdir -p data/raw/gw150914
cd data/raw/gw150914
curl -O -L "https://www.gw-openscience.org/eventapi/html/GWTC-1-confident/GW150914/v3/H-H1_GWOSC_4KHZ_R1-1126259447-32.hdf5"
cd ../../..
Then run:

bash
python scripts/build_gw150914_dataset.py
python scripts/step7_train_cnn.py --epochs 200 --lr 0.003
python scripts/step9_evaluate.py
Configuration
All experiments are controlled by config.yaml. Key sections:

waveform: mass ranges, sample rate, waveform approximant

deviation: deviation kind (amplitude, phase, frequency) and β ranges

noise: noise source (synthetic or real), SNR range

preprocessing: whitening, bandpass, normalization

training: learning rate, epochs, batch size

Method Summary
The classifier combines:

A 1D CNN with three convolutional blocks (strides 8, 2, 2) that
extracts local time-frequency structure from the 16384-sample waveform.

A statistics branch computing 10 hand-crafted features
(mean absolute amplitude, std, max, p95, p99, p99.9, p99.99,
RMS, skewness, kurtosis) passed through a two-layer MLP.

Both representations are concatenated and passed through a final
fully-connected head. Total parameters: 31,697.

Results Summary
Experiment	Deviation	Noise	SNR	test_unseen
Synthetic noise, β ∈ [0.3, 1.0]	amplitude	synthetic	30–50	1.000
Synthetic noise, β ∈ [0.3, 1.0]	amplitude	synthetic	10–15	0.997
Real LIGO, β ∈ [5, 20]	amplitude	real	5–10	0.968
Real LIGO, β ∈ [0.3, 0.5]	amplitude	real	10–15	0.891
Real LIGO, β ∈ [0.5, 2.0]	phase	real	10–15	0.930
Real LIGO, β ∈ [5, 25]	frequency	real	10–15	0.885
GW150914 template, β ∈ [0.3, 0.4]	amplitude	real	20	0.900
GW150914 template, β ∈ [0.01, 0.2]	amplitude	real	20	0.517
Detection Threshold
The central quantitative result is a detectability curve measuring
test accuracy versus deviation strength β. For the GW150914 template
at SNR 20:

β range	test_unseen
0.01–0.2	0.517 (chance)
0.2–0.3	0.750
0.3–0.4	0.900
0.4–0.5	0.967
0.5–1.0	1.000
5–20	1.000
The detection threshold is β ≈ 0.25. The value of β refers to the
quadratic-in-time modulation form adopted in this work; it should not
be interpreted as a generic constraint on Beyond-GR parameters.

Paper
The LaTeX manuscript is in paper/. To compile:

bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
The output is paper/main.pdf.

Requirements
Python 3.11

PyCBC ≥ 2.0

GWpy ≥ 3.0

PyTorch ≥ 2.0

NumPy 1.26 (not 2.x)

SciPy < 1.14

scikit-learn ≥ 1.3

License
MIT License.

Acknowledgements
This work uses gravitational-wave data from the Gravitational-Wave
Open Science Center (GWOSC) and waveform models from the LALSuite
and PyCBC software packages.
