import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from time import sleep

st.set_page_config(layout="wide")
st.title("PSO Swarm Movement Visualisation")

# Import the necessary functions from main script
from f21bcijjz import preprocessing, data, target, ANNArchitecture, PSO
from f21bcijjz import get_pca_trajectory

# Preprocess and initialize
X_train, X_test, y_train, y_test = preprocessing(data, target)
ann = ANNArchitecture(X_train, '8', '4, 4', 'ReLU, tanh, ReLU')
pso = PSO(ann, swarmsize=20)
pso.init_swarm_pos_vel()
best_position, best_fitness = pso.update_pos_and_vel(20, y_train)

# Get position history
positions_2d = get_pca_trajectory(pso.position_history)

# Animate
st.write("### Swarm Trajectory (2D PCA Projection)")

fig, ax = plt.subplots()
scatter = ax.scatter([], [], s=50)
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")
plot_placeholder = st.empty()

for step in range(len(positions_2d)):
    pos = positions_2d[step]
    scatter.set_offsets(pos)
    ax.set_title(f"Iteration {step+1}")
    plot_placeholder.pyplot(fig)
    sleep(0.2)