# alg/pref_learn.py

import os
import numpy as np
import pandas as pd
from features import extract_features_from_state_trajectory
import aprel
from aprel.basics.trajectory import Trajectory, TrajectorySet

# 🛠️ Full patch to fix "can't set attribute" in TrajectorySet
def dynamic_features_matrix(self):
    return np.vstack([traj.features() for traj in self])

TrajectorySet.features_matrix = property(dynamic_features_matrix)

# Monkey patch TrajectorySet so it stops trying to set features_matrix directly
original_init = TrajectorySet.__init__

def patched_init(self, trajectories):
    self.trajectories = list(trajectories)
    # do NOT set self.features_matrix — we override it dynamically
    # Any other initialization logic from the original class can go here if needed

TrajectorySet.__init__ = patched_init

# Now apply your dynamic property
TrajectorySet.features_matrix = property(
    lambda self: np.vstack([traj.features() for traj in self])
)




class FeatureTrajectory(Trajectory):
    def __init__(self, features):
        self.trajectory = []  # unused
        self._features = np.array(features).reshape(1, -1)

    def features(self, _=None):
        return self._features

    @property
    def features_matrix(self):
        return self._features








# Loading the demo trajectories and computing features

TRAJ_FOLDER = "../demo_data/PickAndPlace/"

# Assume filenames: state_traj_0.csv, ..., state_traj_29.csv
trajectories = []
features = []

for i in range(30):
    traj_path = os.path.join(TRAJ_FOLDER, f"state_traj_{i}.csv")
    if not os.path.exists(traj_path):
        print(f"Missing trajectory: {traj_path}")
        continue

    # Load and parse
    df = pd.read_csv(traj_path, header=None)
    parsed = df[0].apply(lambda row: np.array([float(x) for x in row.strip().split()]))
    #state_array = np.stack(parsed.values)
    state_array = np.loadtxt(traj_path)


    # Extract feature vector
    phi = extract_features_from_state_trajectory(state_array)
    trajectories.append(state_array)
    features.append(phi)

#  Defining the reward model 

# Create a linear reward model with your extracted features
reward_model = aprel.LinearRewardBelief()

# Wrap each feature vector into a Trajectory object (APReL format)
aprel_trajectories = [FeatureTrajectory(feat.reshape(1, -1)) for feat in features]



#print("Sample APReL features shape:", aprel_trajectories[0].features.shape)

print("Sample APReL features shape:", aprel_trajectories[0].features().shape)








# Setting up a query generator ===


#asking for human comparison for 10 clips

from itertools import combinations #######
from moviepy.editor import VideoFileClip ####

query_candidates = [
    aprel.PreferenceQuery([left, right])

    for left, right in combinations(aprel_trajectories, 2)
]




# Human feedback loop 

# Ask 10 preference queries choose which clip is better
queries = [np.random.choice(query_candidates) for _ in range(10)]
feedback = []


from moviepy.editor import VideoFileClip

def find_index_by_features(target_feat, all_feats):
        for i, feat in enumerate(all_feats):
            if np.allclose(feat, target_feat):  # tolerant float comparison
                return i
        return -1  # Not found


for i, query in enumerate(queries):
    print(f"\n🟩 Query {i+1} of {len(queries)}:")
    
    left_traj, right_traj = query._slate[0], query._slate[1]

    left_idx = find_index_by_features(left_traj.features(), features)
    right_idx = find_index_by_features(right_traj.features(), features)

    left_clip_path = f"../demo_data/PickAndPlace/clip_{left_idx}.mp4"
    right_clip_path = f"../demo_data/PickAndPlace/clip_{right_idx}.mp4"

    '''
    print(f"🎥 Playing first video: clip_{left_idx}.mp4")
    VideoFileClip(left_clip_path).preview()

    print(f"🎥 Playing second video: clip_{right_idx}.mp4")
    VideoFileClip(right_clip_path).preview()

    answer = input("Which do you prefer? (0 = left, 1 = right): ")
    '''
    answer = 0
    feedback.append(aprel.Preference(query, int(answer)))

num_features = aprel_trajectories[0].features().shape[1]
params = {
    "beta": 1.0,
    "weights": np.zeros(num_features)
}
user_model = aprel.SoftmaxUser(params)


reward_model = aprel.SamplingBasedBelief(
    dataset = feedback,
    user_model = user_model,
    initial_point = {
    "beta": 1.0,
    "weights": np.zeros(num_features)
    }
)


# save the learned weights 
weights = reward_model.mean["weights"]
np.savetxt("final_feature_weights.csv", weights, delimiter=",")
print("Saved learned reward weights to alg/final_feature_weights.csv")
