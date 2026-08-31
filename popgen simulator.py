"""
Population Genetics Simulator
A simple interactive app simulating genetic drift, selection, and mutation
using the Wright-Fisher model.

Run with:  streamlit run popgen_simulator.py
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Population Genetics Simulator", layout="wide")

st.title("🧬 Population Genetics Simulator")
st.markdown(
    "Simulates allele frequency change over generations using the "
    "**Wright-Fisher model** with genetic drift, natural selection, and mutation."
)

# ---------------- Sidebar controls ----------------
st.sidebar.header("Simulation Parameters")

N = st.sidebar.slider("Population size (N)", 10, 2000, 100, step=10)
generations = st.sidebar.slider("Number of generations", 10, 1000, 200, step=10)
p0 = st.sidebar.slider("Initial allele frequency of A (p₀)", 0.0, 1.0, 0.5, step=0.01)
s = st.sidebar.slider("Selection coefficient (s) favoring A", -0.2, 0.2, 0.0, step=0.005)
mu = st.sidebar.slider("Mutation rate A→a (μ)", 0.0, 0.01, 0.0, step=0.0001, format="%.4f")
nu = st.sidebar.slider("Mutation rate a→A (ν)", 0.0, 0.01, 0.0, step=0.0001, format="%.4f")
replicates = st.sidebar.slider("Number of replicate populations", 1, 20, 5)
seed = st.sidebar.number_input("Random seed (0 = random)", 0, 99999, 0)

if seed != 0:
    np.random.seed(seed)

# ---------------- Simulation ----------------
def simulate(N, generations, p0, s, mu, nu):
    """Wright-Fisher simulation for one population. Returns allele freq trajectory."""
    freqs = np.zeros(generations + 1)
    p = p0
    freqs[0] = p
    for gen in range(1, generations + 1):
        # Selection: relative fitness of A = 1+s, a = 1
        w_bar = p * (1 + s) + (1 - p)
        p_sel = p * (1 + s) / w_bar if w_bar > 0 else p
        # Mutation
        p_mut = p_sel * (1 - mu) + (1 - p_sel) * nu
        # Drift: binomial sampling of 2N gene copies
        p = np.random.binomial(2 * N, p_mut) / (2 * N)
        freqs[gen] = p
        if p == 0.0 or p == 1.0:
            freqs[gen:] = p  # allele fixed or lost
            break
    return freqs

trajectories = [simulate(N, generations, p0, s, mu, nu) for _ in range(replicates)]

# ---------------- Plot ----------------
fig, ax = plt.subplots(figsize=(10, 5))
for i, traj in enumerate(trajectories):
    ax.plot(traj, lw=1.5, alpha=0.8, label=f"Population {i+1}" if replicates <= 8 else None)
ax.axhline(0.5, color="gray", ls="--", lw=0.8, alpha=0.5)
ax.set_xlabel("Generation")
ax.set_ylabel("Frequency of allele A (p)")
ax.set_ylim(-0.02, 1.02)
ax.set_title(f"Allele frequency over time  (N={N}, s={s}, μ={mu}, ν={nu})")
if replicates <= 8:
    ax.legend(loc="best", fontsize=8)
st.pyplot(fig)

# ---------------- Summary stats ----------------
final = np.array([t[-1] for t in trajectories])
col1, col2, col3, col4 = st.columns(4)
col1.metric("Mean final frequency", f"{final.mean():.3f}")
col2.metric("Populations where A fixed", int((final == 1.0).sum()))
col3.metric("Populations where A lost", int((final == 0.0).sum()))
col4.metric("Still polymorphic", int(((final > 0) & (final < 1)).sum()))

# ---------------- Hardy-Weinberg panel ----------------
st.subheader("Hardy–Weinberg Genotype Frequencies (final generation, mean)")
p_mean = final.mean()
q_mean = 1 - p_mean
hw = {
    "AA (p²)": p_mean**2,
    "Aa (2pq)": 2 * p_mean * q_mean,
    "aa (q²)": q_mean**2,
}
fig2, ax2 = plt.subplots(figsize=(6, 3))
ax2.bar(hw.keys(), hw.values(), color=["#4c72b0", "#dd8452", "#55a868"])
ax2.set_ylabel("Expected frequency")
ax2.set_ylim(0, 1)
st.pyplot(fig2)

st.markdown("---")
st.caption(
    "Key ideas: small populations show stronger **genetic drift** (random fixation/loss); "
    "positive **selection (s>0)** pushes allele A toward fixation; "
    "**mutation** maintains variation and can balance selection."
)
