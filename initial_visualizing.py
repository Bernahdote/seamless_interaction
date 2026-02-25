import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

base = Path.home() / "datasets" / "seamless_interaction" / "improvised" / "dev" / "0000"
path_data1 = base / "0018" / "V00_S0696_I00000544_P0844A.npz"

path_data2 = base / "0021" / "V00_S0692_I00000535_P0844A.npz"
fps = 30.0

data = np.load(path_data2, allow_pickle=False)

gaze_key = "movement_v4:gaze_encodings"
head_key = "movement_v4:alignment_head_rotation"

if gaze_key not in data.files or head_key not in data.files:
    raise KeyError(
        f"Required movement_v4 keys not found. Missing: "
        f"{[k for k in [gaze_key, head_key] if k not in data.files]}"
    )

gaze = data[gaze_key]  # (N, 2): pitch, yaw
head = data[head_key]  # (N, 3): pitch, yaw, roll
gaze_deg = np.degrees(gaze)
head_deg = np.degrees(head)
combined_pitch_deg = head_deg[:, 0] + gaze_deg[:, 0]
combined_yaw_deg = head_deg[:, 1] + gaze_deg[:, 1]

t = np.arange(gaze_deg.shape[0]) / fps
pitch_min, pitch_max = gaze_deg[:, 0].min(), gaze_deg[:, 0].max()
yaw_min, yaw_max = gaze_deg[:, 1].min(), gaze_deg[:, 1].max()
head_pitch_min, head_pitch_max = head_deg[:, 0].min(), head_deg[:, 0].max()
head_yaw_min, head_yaw_max = head_deg[:, 1].min(), head_deg[:, 1].max()

print(f"Gaze key: {gaze_key}")
print(f"Gaze shape: {gaze_deg.shape}")
print(f"Pitch range (deg): {pitch_min:.2f} to {pitch_max:.2f}")
print(f"Yaw range (deg):   {yaw_min:.2f} to {yaw_max:.2f}")
print(f"Head key: {head_key}")
print(f"Head shape: {head_deg.shape}")
print(f"Head pitch range (deg): {head_pitch_min:.2f} to {head_pitch_max:.2f}")
print(f"Head yaw range (deg):   {head_yaw_min:.2f} to {head_yaw_max:.2f}")

# Correlation between gaze and aligned head rotation (using raw values).
print(f"corr(gaze_pitch, head_pitch): {np.corrcoef(gaze[:, 0], head[:, 0])[0, 1]:.3f}")
print(f"corr(gaze_yaw, head_yaw):     {np.corrcoef(gaze[:, 1], head[:, 1])[0, 1]:.3f}")
print(f"corr(gaze_pitch, head_yaw):   {np.corrcoef(gaze[:, 0], head[:, 1])[0, 1]:.3f}")
print(f"corr(gaze_yaw, head_pitch):   {np.corrcoef(gaze[:, 1], head[:, 0])[0, 1]:.3f}")



fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

ax1.plot(t, gaze_deg[:, 0], label="pitch (deg)")
ax1.plot(t, gaze_deg[:, 1], label="yaw (deg)")
ax1.set_title("Eye Gaze (Degrees)")
ax1.set_ylabel("Degrees")
ax1.legend()

ax2.plot(t, head_deg[:, 0], label="pitch (deg)")
ax2.plot(t, head_deg[:, 1], label="yaw (deg)")
ax2.set_title("Head Rotation (Degrees)")
ax2.set_xlabel("Seconds")
ax2.set_ylabel("Degrees")
ax2.legend()

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 4))
plt.plot(t, combined_pitch_deg, label="pitch (head + gaze)")
plt.plot(t, combined_yaw_deg, label="yaw (head + gaze)")
plt.axhline(
    combined_pitch_deg.mean(),
    linestyle="--",
    linewidth=1,
    alpha=0.8,
    label=f"pitch mean ({combined_pitch_deg.mean():.2f})",
)
plt.axhline(
    combined_yaw_deg.mean(),
    linestyle="--",
    linewidth=1,
    alpha=0.8,
    label=f"yaw mean ({combined_yaw_deg.mean():.2f})",
)
plt.title("Combined Rotation Approximation (Degrees)")
plt.xlabel("Seconds")
plt.ylabel("Degrees")
plt.legend()
plt.tight_layout()
plt.show()
