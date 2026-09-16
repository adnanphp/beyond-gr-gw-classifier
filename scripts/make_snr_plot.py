"""Reproduce the SNR detection plot."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FILES = [
    ('results/metrics/final_results.json',  40,   '30-50'),
    ('results/metrics/snr_15_20.json',      17.5, '15-20'),
    ('results/metrics/snr_10_15.json',      12.5, '10-15'),
    ('results/metrics/snr_8_10.json',        9,   '8-10'),
    ('results/metrics/snr_5_8.json',         6.5, '5-8'),
]

snr, seen, unseen = [], [], []
for path, mid, label in FILES:
    if not Path(path).exists(): continue
    with open(path) as f: r = json.load(f)
    snr.append(mid)
    seen.append(r['test_seen']['accuracy'])
    unseen.append(r['test_unseen']['accuracy'])

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(snr, seen, 'o-', label='Seen types (A, B)', linewidth=2, markersize=10)
ax.plot(snr, unseen, 's-', label='Unseen type (C)', linewidth=2, markersize=10)
ax.axhline(0.9, color='red', linestyle='--', alpha=0.4, label='0.9 threshold')
ax.axhline(0.5, color='gray', linestyle=':', alpha=0.4, label='Chance')
ax.set_xlabel('SNR'); ax.set_ylabel('Accuracy')
ax.set_title('Detection accuracy vs SNR')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/figures/detection_vs_snr.png', dpi=150)
print('Saved results/figures/detection_vs_snr.png')
