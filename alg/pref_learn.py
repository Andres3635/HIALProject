# alg/pref_learn.py

import os
import numpy as np
import pandas as pd
from alg.features import extract_features_from_state_trajectory
import aprel

# Loading the demo trajectories and computing features

TRAJ_FOLDER = "demo_data/PickAndPlace/"

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
    state_array = np.stack(parsed.values)

    # Extract feature vector
    phi = extract_features_from_state_trajectory(state_array)
    trajectories.append(state_array)
    features.append(phi)

#  Defining the reward model 

# Create a linear reward model with your extracted features
reward_model = aprel.LinearRewardModel()

# Wrap each feature vector into a Trajectory object (APReL format)
aprel_trajectories = [aprel.Trajectory(feat.reshape(1, -1)) for feat in features]

# Setting up a query generator ===


#asking for human comparison for 10 clips
query_generator = aprel.PreferenceQueryGenerator(aprel_trajectories, reward_model)

# Human feedback loop 

# Ask 10 preference queries choose which clip is better
queries = [query_generator.make_query() for _ in range(10)]
feedback = []

for query in queries:
    query.display()  
    answer = input("Which do you prefer? (0 = left, 1 = right): ")
    feedback.append(aprel.Preference(query, int(answer)))
    reward_model = reward_model.fit(feedback)  # update reward model with new info here <<

# save the learned weights 
weights = reward_model.get_weights()
np.savetxt("alg/final_feature_weights.csv", weights, delimiter=",")
print("Saved learned reward weights to alg/final_feature_weights.csv")
