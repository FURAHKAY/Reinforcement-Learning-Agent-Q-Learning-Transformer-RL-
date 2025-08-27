# app.py
import os
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from environments.gridworld import GridWorld
from agents.q_learning import QLearningAgent

# Try PyTorch + your transformer agent (optional)
HAS_TORCH = True
try:
    import torch
    import torch.nn as nn
    from agents.transformer_rl import TransformerRLAgent
except Exception:
    HAS_TORCH = False

st.set_page_config(page_title="GridWorld RL", page_icon="🧭", layout="wide")
st.title("GridWorld Reinforcement Learning")

tab_q, tab_tx = st.tabs(["Q-Learning", "Transformer RL"])

# ------------------------ Q-LEARNING TAB ------------------------
with tab_q:
    colL, colR = st.columns([1.2, 1])

    with st.sidebar:
        st.header("Environment")
        size = st.number_input("Grid size", 4, 12, 5, 1)
        start_r = st.number_input("Start row", 0, size-1, 0, 1)
        start_c = st.number_input("Start col", 0, size-1, 0, 1)
        goal_r  = st.number_input("Goal row", 0, size-1, size-1, 1)
        goal_c  = st.number_input("Goal col", 0, size-1, size-1, 1)

        st.header("Q-learning params")
        episodes = st.number_input("Episodes", 10, 5000, 500, 10)
        alpha    = st.slider("alpha (lr)", 0.01, 1.0, 0.1, 0.01)
        gamma    = st.slider("gamma (discount)", 0.50, 0.999, 0.99, 0.01)
        epsilon  = st.slider("epsilon (explore)", 0.0, 1.0, 0.1, 0.05)

        run_q = st.button("Train Q-learning")

    def draw_grid(ax, size, start, goal, agent_pos=None, title=""):
        ax.clear()
        ax.imshow(np.zeros((size, size)), cmap="Greys", vmin=0, vmax=1)
        ax.scatter(goal[1], goal[0], marker="*", s=300, color="gold", label="Goal")
        ax.scatter(start[1], start[0], marker="s", s=120, color="tab:blue", label="Start")
        if agent_pos is not None:
            ax.scatter(agent_pos[1], agent_pos[0], marker="o", s=120, color="tab:red", label="Agent")
        ax.set_xticks(range(size)); ax.set_yticks(range(size))
        ax.set_xticklabels([]); ax.set_yticklabels([])
        ax.grid(True, color="lightgray", linewidth=0.7)
        ax.set_title(title)
        ax.legend(loc="upper left", fontsize=8)

    if run_q:
        os.makedirs("plots", exist_ok=True)
        env = GridWorld(size=size, start=(start_r, start_c), goal=(goal_r, goal_c))
        agent = QLearningAgent(n_states=size*size, n_actions=4, alpha=alpha, gamma=gamma, epsilon=epsilon)
        rewards = []

        live = colL.empty()
        for ep in range(int(episodes)):
            s = env.reset()
            total, done, steps = 0.0, False, 0
            while not done and steps < size*size*4:
                a = agent.act(s)
                s_next, r, done = env.step(a)
                agent.update(s, a, r, s_next)
                s = s_next
                total += r
                steps += 1
            rewards.append(total)

            if ep % max(1, int(episodes)//20) == 0 or ep == episodes-1:
                fig, ax = plt.subplots(figsize=(4,4))
                draw_grid(ax, size, env.start, env.goal, (s//size, s%size), f"Episode {ep+1}/{int(episodes)}")
                live.pyplot(fig, clear_figure=True); plt.close(fig)

        # reward curve
        fig2, ax2 = plt.subplots(figsize=(6,3))
        ax2.plot(rewards); ax2.set_xlabel("Episode"); ax2.set_ylabel("Total Reward"); ax2.set_title("Q-learning Reward")
        colL.pyplot(fig2, clear_figure=True); plt.close(fig2)

        # arrows policy
        arrow = {0:"↑",1:"↓",2:"←",3:"→"}
        policy = np.array([arrow[int(np.argmax(agent.Q[s]))] for s in range(size*size)]).reshape(size, size)
        colR.subheader("Greedy policy")
        colR.table(policy)

        # max-Q heatmap
        bestQ = agent.Q.max(axis=1).reshape(size, size)
        fig3, ax3 = plt.subplots(figsize=(4.5,4))
        im = ax3.imshow(bestQ, cmap="viridis"); ax3.set_title("Max Q per cell"); ax3.set_xticks([]); ax3.set_yticks([])
        fig3.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
        colR.pyplot(fig3, clear_figure=True); plt.close(fig3)

        # save
        np.save("plots/Q_table.npy", agent.Q)
        with open("plots/rewards_q.txt","w") as f: f.write("\n".join(map(str, rewards)))
        st.success("Saved: plots/Q_table.npy, plots/rewards_q.txt")

    else:
        fig, ax = plt.subplots(figsize=(4,4))
        draw_grid(ax, size, (start_r,start_c), (goal_r,goal_c), None, "Environment")
        colL.pyplot(fig); plt.close(fig)
        colR.info("Set parameters and click **Train Q-learning**.")

# --------------------- TRANSFORMER TAB ---------------------
with tab_tx:
    if not HAS_TORCH:
        st.warning("PyTorch not detected. Install it first:\n\n"
                   "`pip install torch --index-url https://download.pytorch.org/whl/cpu`")
    else:
        st.caption("This runs a short on-policy TD-style loop using your Transformer encoder.")
        t_size   = st.number_input("Grid size", 4, 12, 5, 1, key="t_size")
        t_eps    = st.number_input("Episodes", 10, 2000, 200, 10, key="t_eps")
        t_lr     = st.slider("LR", 1e-4, 5e-3, 1e-3, 1e-4, key="t_lr")
        t_gamma  = st.slider("Gamma", 0.50, 0.999, 0.99, 0.01, key="t_gamma")
        run_tx   = st.button("Train Transformer RL")

        if run_tx:
            os.makedirs("plots", exist_ok=True)
            env = GridWorld(size=t_size)
            model = TransformerRLAgent(state_dim=t_size*t_size, n_actions=4, d_model=64, n_heads=4, n_layers=2)
            opt = torch.optim.Adam(model.parameters(), lr=float(t_lr))
            loss_fn = nn.MSELoss()
            rewards = []

            for ep in range(int(t_eps)):
                s = env.reset()
                seq = [s]
                total, done, steps = 0.0, False, 0

                while not done and steps < t_size*t_size*4:
                    seq_tensor = torch.tensor([seq[-10:]], dtype=torch.long)      # (1, seq_len)
                    q_vals = model(seq_tensor)                                     # (1, n_actions)
                    a = int(torch.argmax(q_vals, dim=1).item())

                    s_next, r, done = env.step(a)

                    with torch.no_grad():
                        seq_next = torch.tensor([[*seq[-9:], s_next]], dtype=torch.long)
                        q_next = model(seq_next).max().item()

                    target = torch.tensor(r + float(t_gamma) * q_next, dtype=torch.float32)
                    pred_q = q_vals[0, a]
                    loss = loss_fn(pred_q, target)
                    opt.zero_grad(); loss.backward(); opt.step()

                    seq.append(s_next)
                    total += r
                    s = s_next
                    steps += 1

                rewards.append(total)

            fig, ax = plt.subplots(figsize=(6,3))
            ax.plot(rewards)
            ax.set_xlabel("Episode"); ax.set_ylabel("Total Reward"); ax.set_title("Transformer RL Reward")
            st.pyplot(fig, clear_figure=True); plt.close(fig)

            with open("plots/rewards_transformer.txt","w") as f: f.write("\n".join(map(str, rewards)))
            st.success("Saved: plots/rewards_transformer.txt")
