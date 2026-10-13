from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np

fwd_path = Path(__file__).resolve().parent.parent / "sample-fwd.fif"
fwd = mne.read_forward_solution(fwd_path)
G = fwd["sol"]["data"]
picks = mne.pick_types(fwd["info"], meg="mag")
r = np.linalg.norm(fwd["source_rr"] - np.array([0, 0, 0.04]), axis=1)
gain = np.linalg.norm(G[picks].reshape(len(picks), len(r), -1), axis=(0, 2))
plt.scatter(r * 1000, gain, s=2)
plt.yscale("log")
plt.xlabel("distance from head center (mm)")
plt.ylabel("leadfield norm (mag)")
plt.show()