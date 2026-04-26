import pandas as pd
import matplotlib.pyplot as plt

# remove duplicate rows
df = pd.read_csv("training_log.csv")
df = df.drop_duplicates(subset="rally").sort_values("rally")

rallies = df["rally"].values
avg_len = df["avg_rally_length"].values
epsilon = df["epsilon"].values
left_states = df["left_states"].values
right_states = df["right_states"].values

# average
smooth = pd.Series(avg_len).rolling(window=5, center=True, min_periods=1).mean().values

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7))
fig.suptitle("training progression of Q-Learning agents", fontweight="bold")
fig.subplots_adjust(hspace=0.45)

# --- a graph rally length and epsilon ---
ax1.plot(rallies, avg_len, color="lightblue", linewidth=0.8, alpha=0.6)
ax1.plot(rallies, smooth, color="blue", linewidth=2, label="avg rally length")
ax1.set_xlabel("rallies")
ax1.set_ylabel("avg rally length")
ax1.set_title("(a)  average rally length and epsilon decay over training", fontweight="bold")

ax1r = ax1.twinx()
ax1r.plot(rallies, epsilon, color="orange", linewidth=1.8, linestyle="--", label="epsilon (ε)")
ax1r.set_ylabel("epsilon (ε)", color="orange")
ax1r.tick_params(axis="y", labelcolor="orange")
ax1r.set_ylim(0, 1.05)

for checkpoint in [15000, 30000]:
    ax1.axvline(checkpoint, color="grey", linewidth=1, linestyle=":")

ax1.text(15200, 4.5, "15k\n(ε → 0.05)", fontsize=8, color="grey")
ax1.text(30200, 4.5, "30k", fontsize=8, color="grey")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1r.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

# --- b table q-table growth ---
ax2.plot(rallies, left_states, color="blue", linewidth=2, label="left agent")
ax2.plot(rallies, right_states, color="green", linewidth=2, linestyle="--", label="right agent")
ax2.set_xlabel("rallies")
ax2.set_ylabel("unique Q-table states")
ax2.set_title("(b)  Q-table state space growth over training", fontweight="bold")
ax2.legend(fontsize=8)

for checkpoint in [15000, 30000]:
    ax2.axvline(checkpoint, color="grey", linewidth=1, linestyle=":")

plt.savefig("training_progress.png", dpi=200, bbox_inches="tight")
print("saved training_progress.png")