import numpy as np
import json 
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import soundfile as sf


base = Path.home()/"datasets"/"seamless_interaction"

path_data = base /"by_interaction"/"V00_S2022_I00001199/V00_S2022_I00001199_P1275A.npz"

path_json1 = base /"by_interaction"/"V00_S2022_I00001199/V00_S2022_I00001199_P1275A.json"
path_wav1 = base /"by_interaction"/"V00_S2022_I00001199/V00_S2022_I00001199_P1275A.wav"

path_json2 = base /"by_interaction"/"V00_S2022_I00001199/V00_S2022_I00001199_P1277A.json"
path_wav2 = base /"by_interaction"/"V00_S2022_I00001199/V00_S2022_I00001199_P1277A.wav"



# data = np.load(path_data)
# print(data.files)

# --- PART 1: EXPLORING VAD ---

with open(path_json1, 'r') as f:
    metadata1 = json.load(f)

vad_intervals1 = metadata1["metadata:vad"]

with open(path_json2, 'r') as f:
    metadata2 = json.load(f)

vad_intervals2 = metadata2["metadata:vad"]



# Load audio file
wave1, sr1 = sf.read(path_wav1)
wave2, sr2 = sf.read(path_wav2)


vad1 = np.zeros(len(wave1), dtype=np.float32)
vad2 = np.zeros(len(wave2), dtype=np.float32)

for seg in vad_intervals1:
    s = int(seg["start"]*sr1)
    e  = int(seg["end"] *sr1)
    vad1[s:e] = 1

for seg in vad_intervals2:
    s = int(seg["start"]*sr2)
    e  = int(seg["end"] * sr2)
    vad2[s:e] = 1


plt.figure() 

plt.subplot(2,1,1)
t_full1 = np.arange(len(wave1)) / sr1
idx1 = np.flatnonzero(vad1 == 1)
t1 = idx1 / sr1
plt.scatter(t1, np.ones_like(t1), s=2, color="orange")
plt.plot(t_full1, wave1, color="blue", alpha=0.5)

plt.subplot(2,1,2)
t_full2 = np.arange(len(wave2)) / sr2
idx2 = np.flatnonzero(vad2 == 1)
t2 = idx2 / sr2
plt.scatter(t2, np.ones_like(t2), s=2, color="green")
plt.plot(t_full2, wave2, color="blue", alpha=0.5)

plt.show()




# --- PART 2: EXPLORING OTHER DATA ---


#data = np.load(path_data)
# Primary interest 

# gaze_enc = data["movement_v4:gaze_encodings"] #Neural encodings of gaze direction from computed blendshapes
# head_enc = data["movement:head_encodings"] #Neural encodings of head position and rotation
# body_pose = data["smplh:body_pose"] #Body pose parameters
# lh_pose = data["smplh:left_hand_pose"] #Left hand pose parameters
# rh_pose = data["smplh:right_hand_pose"] #Right hand pose parameters
# mv_is_valid = data["movement_v4:is_valid"] #Indicates valid movement frames
# smplh_is_valid = data["smplh:is_valid"] #Indicates valid SMPL-H frames

# print(np.count_nonzero(mv_is_valid == 0))
# print(np.count_nonzero(smplh_is_valid == 0)) 

# print(np.max(gaze_enc[:, 0]), np.min(gaze_enc[:, 0]))  # Unnormalized 
# print(np.max(gaze_enc[:, 1]), np.min(gaze_enc[:, 1])) 


# print(gaze_enc.shape)
# valid = (mv_is_valid != 0) & (smplh_is_valid != 0)
# gaze_enc = gaze_enc[valid]

# print(gaze_enc.shape)

# mean = gaze_enc.mean(axis=0)   #Normalizing 
# std  = gaze_enc.std(axis=0)
# gaze_norm = (gaze_enc - mean) / std



