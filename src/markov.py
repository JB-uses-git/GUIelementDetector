"""Markov Chain Analysis for GUI Element Sequences.

Models the sequential layout of UI elements (e.g., Text -> EditText -> TextButton)
using a First-Order Discrete Markov Chain.
"""
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class UIMarkovModel:
    def __init__(self):
        # transitions[current_state][next_state] = count
        self.transitions = defaultdict(lambda: defaultdict(int))
        self.state_counts = defaultdict(int)

    def extract_sequence(self, dets):
        """Sort detections by vertical position (y, then x) to represent layout reading order."""
        if not dets:
            return []
        sorted_dets = sorted(dets, key=lambda d: (d["box"]["y"], d["box"]["x"]))
        return [d["class"] for d in sorted_dets]

    def add_sequence(self, sequence):
        """Update Markov counts with a sequence of element classes."""
        if len(sequence) < 2:
            if len(sequence) == 1:
                self.state_counts[sequence[0]] += 1
            return

        for a, b in zip(sequence[:-1], sequence[1:]):
            self.transitions[a][b] += 1
            self.state_counts[a] += 1
        self.state_counts[sequence[-1]] += 1

    def get_transition_matrix(self, sequence_only=None):
        """Return transition probability matrix DataFrame.
        
        If sequence_only is provided, calculates the transition matrix strictly for that sequence.
        Otherwise, uses the model's accumulated transitions.
        """
        trans = defaultdict(lambda: defaultdict(int))
        if sequence_only is not None:
            if len(sequence_only) >= 2:
                for a, b in zip(sequence_only[:-1], sequence_only[1:]):
                    trans[a][b] += 1
        else:
            trans = self.transitions

        states = sorted(list(set(trans.keys()) | {b for row in trans.values() for b in row.keys()}))
        if not states:
            return pd.DataFrame()

        mat = np.zeros((len(states), len(states)), dtype=float)
        for i, s1 in enumerate(states):
            total = sum(trans[s1].values())
            if total > 0:
                for j, s2 in enumerate(states):
                    mat[i, j] = trans[s1][s2] / total

        return pd.DataFrame(mat, index=states, columns=states)

    def plot_heatmap(self, df_probs, title="Markov Chain: UI Element Transition Probabilities"):
        """Plot and return RGB numpy image of the transition probability matrix."""
        if df_probs.empty or df_probs.shape[0] == 0:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, "Not enough elements for sequence transitions\n(need at least 2 detected elements)",
                    ha="center", va="center", fontsize=12, color="gray")
            ax.axis("off")
        else:
            n = len(df_probs)
            fig_w = max(6, min(11, n * 0.8 + 2.5))
            fig_h = max(5, min(9, n * 0.65 + 2.2))
            fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=100)

            cax = ax.matshow(df_probs.values, cmap="Blues", vmin=0, vmax=1)
            fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04, label="Transition Probability P(S_{t+1} | S_t)")

            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(df_probs.columns, rotation=45, ha="left", fontsize=9, fontweight="bold")
            ax.set_yticklabels(df_probs.index, fontsize=9, fontweight="bold")

            # Annotate cells with non-zero probability values
            for i in range(n):
                for j in range(n):
                    val = df_probs.values[i, j]
                    if val > 0:
                        color = "white" if val > 0.55 else "black"
                        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                                color=color, fontsize=8, fontweight="bold")

            ax.set_xlabel("Next Element (State t+1)", labelpad=10, fontsize=10, fontweight="bold")
            ax.set_ylabel("Current Element (State t)", labelpad=10, fontsize=10, fontweight="bold")
            ax.set_title(title, pad=25, fontsize=11, fontweight="bold")
            plt.tight_layout()

        # Render to numpy RGB array
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        plt.close(fig)
        return rgba[:, :, :3]

    def generate_report(self, sequence, df_probs):
        """Generate markdown summary of sequence transitions and predictions."""
        if not sequence:
            return "No UI elements detected to analyze."

        seq_str = " -> ".join([f"**{s}**" for s in sequence])
        md = [
            "### UI Element Reading Sequence (Top to Bottom)",
            f"{seq_str}\n"
        ]

        if len(sequence) < 2 or df_probs.empty:
            md.append("> *At least 2 UI elements are needed to compute transitions.*")
            return "\n".join(md)

        # Top transitions observed
        transitions_found = []
        for i in range(len(df_probs.index)):
            curr = df_probs.index[i]
            for j in range(len(df_probs.columns)):
                nxt = df_probs.columns[j]
                prob = df_probs.values[i, j]
                if prob > 0:
                    transitions_found.append((curr, nxt, prob))

        transitions_found.sort(key=lambda x: x[2], reverse=True)

        md.append("### Key Transition Probabilities: P(Next | Current)")
        md.append("| Current Element (State t) | Next Element (State t+1) | Probability |")
        md.append("| :--- | :--- | :--- |")
        for curr, nxt, p in transitions_found[:7]:
            md.append(f"| **{curr}** | **{nxt}** | **{p*100:.1f}%** ({p:.2f}) |")

        # Next element prediction for viva discussion
        last_elem = sequence[-1]
        if last_elem in df_probs.index:
            row = df_probs.loc[last_elem]
            best_next = row.idxmax()
            best_prob = row.max()
            if best_prob > 0:
                md.append(f"\n> **Viva Insight / Next Prediction:** Given the bottom element was **`{last_elem}`**, "
                          f"the Markov Model predicts the most probable next element is **`{best_next}`** ({best_prob*100:.1f}% probability).")

        return "\n".join(md)
