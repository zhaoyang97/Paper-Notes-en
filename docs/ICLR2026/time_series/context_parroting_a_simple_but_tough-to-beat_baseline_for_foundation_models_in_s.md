---
title: >-
  [Paper Note] Context parroting: A simple but tough-to-beat baseline for foundation models in scientific machine learning
description: >-
  [ICLR2026][Time Series][Context Parroting] The authors propose a minimalist baseline called "context parroting"—which simply identifies the most similar segment in historical trajectories and copies the subsequent evolution as the prediction. On zero-shot forecasting of low-dimensional chaos, turbulence, coupled oscillators, and ECG signals, this method **outperforms leading foundation models like Chronos, TimesFM, Time-MoE, Moirai, and DynaMix in both accuracy and long-term…
tags:
  - "ICLR2026"
  - "Time Series"
  - "Context Parroting"
  - "Zero-shot Forecasting"
  - "Chaotic Dynamical Systems"
  - "Time-series Foundation Models"
  - "In-context Scaling Law"
date: 2026-05-08
content_hash: 3b87ad983845e101
---

# Context parroting: A simple but tough-to-beat baseline for foundation models in scientific machine learning

**Conference**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=EUAXc9Hlvm](https://openreview.net/forum?id=EUAXc9Hlvm)  
**Code**: https://github.com/y-z-zhang/parroting  
**Area**: Time-series Foundation Models / Chaotic Systems / Scientific Machine Learning  
**Keywords**: Context Parroting, Zero-shot Forecasting, Chaotic Dynamical Systems, Time-series Foundation Models, In-context Scaling Law

## TL;DR
The authors propose a minimalist baseline called "context parroting"—which simply identifies the most similar segment in historical trajectories and copies the subsequent evolution as the prediction. On zero-shot forecasting of low-dimensional chaos, turbulence, coupled oscillators, and ECG signals, this method **outperforms leading foundation models like Chronos, TimesFM, Time-MoE, Moirai, and DynaMix in both accuracy and long-term attractor reconstruction, while being six orders of magnitude cheaper in inference**, thereby exposing that current foundation models have not truly "learned physics."

## Background & Motivation
**Background**: A core task for testing generalization in Scientific ML is "zero-shot forecasting"—predicting the future state of an entirely new physical system given only a short context trajectory, without knowing the underlying equations. Earlier approaches trained specialized models for each system, limited by specific data availability. Recent shifts favor "Time-series Foundation Models" (Chronos, TimesFM, Time-MoE, Moirai, etc.), pre-trained on massive real and simulated data, claiming zero-shot capabilities for unseen dynamical systems. Prior work even suggested these models predict chaotic systems better than classical deep models when historical data is scarce.

**Limitations of Prior Work**: It remains unclear "what mechanism" these foundation models utilize for zero-shot forecasting or why they work on unseen systems. They act as black boxes without interpretable strategies. Furthermore, the default assumption that they have "learned universal temporal laws" has never been rigorously challenged.

**Key Challenge**: The authors previously observed that Chronos often employs an extremely simple strategy when predicting chaos: scanning the context for nearly identical "motifs" and copying the segment following the best match. If a trivial "copy-paste" approach can match or exceed these expensive pre-trained models, it suggests that current foundation models do not fully exploit context information and are far from truly understanding dynamical systems.

**Goal**: (1) Distill this "parroting" strategy into a clean, reproducible baseline algorithm; (2) Systematically evaluate it against leading foundation models to expose common failure modes; (3) Use this interpretable model to explain the "in-context neural scaling laws" observed in literature.

**Key Insight**: From the perspective of dynamical systems geometry, a sufficiently long chaotic trajectory is a random sampling of an attractor. Takens' Embedding Theorem guarantees that delay coordinates can recover key geometric properties. Thus, "finding the most similar historical segment and copying it" is essentially a nearest-neighbor prediction in the delay embedding space. This perspective explains foundation model efficacy and links "context length vs. accuracy" to the fractal dimension of the attractor.

**Core Idea**: Utilize "nearest-neighbor motif matching + copying in delay embedding space" as a strong baseline for zero-shot forecasting, revealing the capacity limits and scaling sources of foundation models.

## Method

### Overall Architecture
The logic of context parroting is simple: **Take the last $D$ points of the context as a "query motif," find the most similar motif in the preceding history, and copy the evolution following that motif as the prediction; if the desired length is not met, repeat the process until the prediction length $H$ is reached.** The input is a context trajectory $x_{1:L}$ of length $L$ and two hyperparameters (embedding dimension $D$, prediction length $H$), outputting $x_{L+1:L+H}$.

Here, $D$ serves as the "matching motif length" and represents the "embedding dimension" in Takens’ theorem. In $D$-dimensional delay coordinates, this process is a nearest-neighbor algorithm. The last $D$ points are excluded from the search to avoid copying segments too close to the prediction starting point. This process corresponds to three actions of an induction head: query lookup (copy head), nearest-neighbor matching (selector), and point-by-point copying (aggregation). As a pure retrieval-and-copy method, it has no parameters, requires no training, and costs only $O(DL)$ for a nearest-neighbor search.

### Key Designs

**1. Nearest-Neighbor Motif Matching in Delay Embedding Space**

This is formalized as the primary algorithm. Given context $x_{1:L}$, for every motif $x_{s-D+1:s}$ in history, the Euclidean distance $d_s$ to the "terminal query motif" $x_{L-D+1:L}$ is calculated. The evolution following the best match $s_{\text{opt}}$ is copied. This addresses the "black box" nature of foundation models by providing a non-parametric, analytical $O(DL)$ algorithm as a benchmark. Notably, although parroting is periodic by definition, it can capture **finite-time Lyapunov exponents** on finite trajectories. As context length increases, the copied cycles lengthen, and the estimated Lyapunov and power spectra approach ground truth values.

**2. Equivalence to Induction Heads and Classical Nonlinear Prediction**

Context parroting is fundamentally similar to the "induction heads" that emerge in LLMs—both are "copy-and-paste" mechanisms. While a simple induction head matches one token (seeing `[A][B]...[A]` and outputting `[B]`), parroting matches continuous tokens. This explains why LLMs trained on text can predict time series without fine-tuning: text-trained induction heads are reused as parroting strategies. Furthermore, the authors prove parroting is equivalent to "simplex projection" and "S-maps" from nonlinear dynamics under different limits. Unlike simplex projection, which is sensitive to $D$ due to motif averaging, parroting uses a single best match and remains robust to $D$.

**3. Parsing In-context Neural Scaling Laws via Parroting**

Prior research observed that prediction error in LLMs follows a power law relative to context length. Using parroting, the authors replicate and explain this: longer contexts allow for closer nearest neighbors in the embedding space, enabling the model to "shadow" the ground truth longer. Assuming the mean error $\langle e\rangle$ is linearly related to the motif distance $\langle \ell\rangle$, both follow the same power law $L^{-\alpha}$. The authors link the exponent $\alpha$ to the attractor’s correlation dimension $d_{\text{cor}}$, predicting:

$$\alpha = \frac{1}{d_{\text{cor}}}.$$

Experiments show a Spearman correlation of ~0.85 between $d_{\text{cor}}$ and $1/\alpha$, suggesting that neural scaling laws are fundamentally determined by the invariants of the data-generating process.

## Key Experimental Results

### Main Results: Low-dimensional Chaos (dysts, 135 systems)
Evaluated on 135 systems (30 points/Lyapunov time sampling, normalized), context length 512, prediction length 300. Baselines: Chronos, Chronos-Bolt, TimesFM-2.0, Time-MoE, Moirai-2.0, and DynaMix.

| Metric | Parrot (Ours) | Best Foundation Model | Conclusion |
|------|------|------|------|
| Prediction Error (sMAPE) | Lowest | Chronos (Best among Transformers) | Parrot consistently lowest |
| Attractor KL Divergence | Lowest | DynaMix / Chronos | Parrot is best; DynaMix (RNN) preserves "climate" |
| Power Spectrum (Hellinger) | Best | — | Parrot is most accurate despite periodic output |
| Inference Cost | Baseline | Chronos is ~$10^6$ higher | Six orders of magnitude difference |

Key Observation: **Chronos performs best among Transformers** because it frequently employs a parroting strategy. Trained with cross-entropy as a language model over quantized tokens, it preserves $k$-gram frequencies. In contrast, **TimesFM/Time-MoE, trained with MSE, lose diversity in long-term predictions, regressing to the mean and suppressing oscillations.**

### Role of Context Length
Longer context benefits Parroting, DynaMix, and Chronos. However, Chronos saturates at its architectural limit (512), while Parroting (and SSMs) improves indefinitely with length. Interestingly, **Chronos outperforms Parroting with very short contexts**, suggesting it possesses non-copying strategies (e.g., local trend continuation) for non-stationary segments.

### Real-world Tasks Beyond Chaos
Validated on 4 high-dimensional systems (Turbulence, ECG, Circuits, Kuramoto oscillators):

| Task | MAE@50 Parrot | Rank | Remarks |
|------|------|------|------|
| Turbulence | 0.403 | Top 3 | TimeMoE/Moirai slightly better |
| ECG | 0.624 | 1st | Significantly better than foundation models |
| Circuit | 0.083 | 1st | Far exceeds DynaMix(0.425)/Chronos(0.111) |
| Kuramoto | 0.004 | Joint 1st | Tied with Moirai, far exceeds others |

**Parrot is the only model to rank in the top 3 across all tasks and metrics.**

### Key Findings
- **Foundation models' greatest weakness is "regression to the mean"**: MSE-trained models (TimesFM, Time-MoE) underestimate oscillations in long-term forecasts.
- **Training loss dictates behavior**: Cross-entropy (Chronos) preserves $k$-gram frequency $\rightarrow$ favors parroting $\rightarrow$ closer to truth. MSE $\rightarrow$ loss of diversity $\rightarrow$ regression to mean.
- **Parrot is robust to the embedding dimension $D$**, making it a practical and powerful baseline.

## Highlights & Insights
- **The "Mirror" for Foundation Models**: If a foundation model cannot beat "parroting," it hasn't learned the underlying physics. This follows the "simple but tough-to-beat baseline" philosophy of Arora et al. (2017).
- **Unifying Framework**: Connects modern foundation models, LLM induction heads, and classical nonlinear dynamics (simplex/S-maps) as variations of nearest-neighbor copying in delay embeddings.
- **Geometric Origin of Scaling**: Links $\alpha=1/d_{\text{cor}}$, suggesting neural scaling laws are determined by data-generating invariants. This invites the question: "Can we infer the fractal dimension of language from scaling laws?"
- **Actionable Insight**: Any claim of "zero-shot understanding" should be compared against "retrieve-and-copy." Evaluations should focus on abilities orthogonal to parroting (e.g., inferring unobserved parameters or generalization across bifurcations).

## Limitations & Future Work
- **Stationarity Assumption**: Parroting assumes a stationary underlying measure, which may not apply to non-stationary series (weather/finance trends). "Non-stationary parroting" is a future direction to replace Seasonal Naive baselines.
- **Stochastic Systems**: The power-law explanation currently covers deterministic ODEs and discrete maps; purely stochastic scaling (e.g., Markov chains) remains unexplained.
- **Baseline, not Replacement**: Parroting is a baseline intended to expose gaps and guide architecture. It performs periodic extrapolation and lacks true physical extrapolation (e.g., generalizing across bifurcation regimes).

## Related Work & Insights
- **vs. Foundation Models**: Highlights the computational inefficiency and "mean reversion" failure of expensive models compared to non-parametric retrieval.
- **vs. Induction Heads**: Bridges ICL mechanism research with time-series forecasting, explaining the "unreasonable effectiveness" of text LLMs on numerical data.
- **vs. Classical Methods**: Proves parroting is a more robust version of simplex projection by avoiding the sensitivity inherent in multi-neighbor averaging.
- **vs. Neural Scaling Laws**: Shifts focus from model/data size to the geometric properties of the data-generating process.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Uses a naive baseline to subvert assumptions about "learned physics" and unifies three disparate fields.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extensive comparison across 135+ systems with theoretical and empirical scaling law support.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear, powerful argumentation combining physical intuition with analytical theory.
- Value: ⭐⭐⭐⭐⭐ Establishes a mandatory baseline for the field, impacting future evaluation and model design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] In-Context Learning of Stochastic Differential Equations with Foundation Inference Models](../../NeurIPS2025/time_series/in-context_learning_of_stochastic_differential_equations_with_foundation_inferen.md)
- [\[ICLR 2026\] SciTS: Scientific Time Series Understanding and Generation with LLMs](scits_scientific_time_series_understanding_and_generation_with_llms.md)
- [\[ICLR 2026\] FeDaL: Federated Dataset Learning for General Time Series Foundation Models](fedal_federated_dataset_learning_for_general_time_series_foundation_models.md)
- [\[ICLR 2026\] CauKer: Classification Time Series Foundation Models Can Be Pretrained on Synthetic Data](cauker_classification_time_series_foundation_models_can_be_pretrained_on_synthet.md)
- [\[ICLR 2026\] Brain-Semantoks: Learning Semantic Tokens of Brain Dynamics with a Self-Distilled Foundation Model](brain-semantoks_learning_semantic_tokens_of_brain_dynamics_with_a_self-distilled.md)

</div>

<!-- RELATED:END -->
