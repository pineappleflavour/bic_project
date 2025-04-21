import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from time import sleep
from sklearn.decomposition import PCA

st.set_page_config(layout="wide")
st.title("PSO Swarm Movement Visualisation")

# Import the necessary functions from your main script
from f21bcijjz import preprocessing, data, target, ANNArchitecture, PSO, get_pca_trajectory

# --- Streamlit UI for configuration ---
st.sidebar.header("Model Configuration")

# Choose configuration method
config_method = st.sidebar.radio("Choose Configuration Method", ("Bypass", "UI"))

# Default values
iterations = 100
user_feature_count = '8'
user_hidden_layers_nodes = '4, 4'
activation_function_list = 'ReLU, tanh, ReLU'

if config_method == "Bypass":
    st.sidebar.write("Using Bypass Configuration")
    # These values could also be editable with st.sidebar.text_input/selectbox if you want

else:
    st.sidebar.write("Custom UI Configuration")
    iterations = st.sidebar.number_input("Number of Iterations", min_value=10, max_value=500, value=100, step=10)
    user_feature_count = st.sidebar.text_input("Feature Count", value='8')

    num_hidden_layers = st.sidebar.number_input("Number of Hidden Layers", min_value=1, max_value=10, value=2)

    user_hidden_layers_nodes = []
    activation_function_list = []

    for i in range(num_hidden_layers):
        user_hidden_layers_nodes.append(st.sidebar.number_input(f"Nodes in Hidden Layer {i+1}", min_value=1, max_value=100, value=4))
        activation = st.sidebar.selectbox(f"Activation for Layer {i+1}", ["ReLU", "tanh", "logistic"], index=0, key=f"activ_{i}")
        activation_function_list.append(activation)

    activation_function_list.append(st.sidebar.selectbox("Activation for Output Layer", ["ReLU", "tanh", "logistic"], index=0))
    user_hidden_layers_nodes = ', '.join(str(i) for i in user_hidden_layers_nodes)
    activation_function_list = ', '.join(activation_function_list)

# Preprocess and initialize
X_train, X_test, y_train, y_test = preprocessing(data, target)
ann = ANNArchitecture(X_train, user_feature_count, user_hidden_layers_nodes, activation_function_list)
pso = PSO(ann, swarmsize=20)
pso.init_swarm_pos_vel()
best_position, best_fitness = pso.update_pos_and_vel(iterations, y_train)

# Get position history
positions_2d = get_pca_trajectory(pso.position_history)

# Animate
st.write("### Swarm Trajectory (2D PCA Projection)")

fig, ax = plt.subplots()
scatter = ax.scatter([], [], s=50)
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")
plot_placeholder = st.empty()

for step in range(len(positions_2d)):
    pos = positions_2d[step]
    scatter.set_offsets(pos)
    ax.set_title(f"Iteration {step+1}")
    plot_placeholder.pyplot(fig)
    sleep(0.2)
