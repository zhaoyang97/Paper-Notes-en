---
title: >-
  [Paper Note] Inverse Depth Scaling From Most Layers Being Similar
description: >-
  [ICML2026][LLM Pretraining][Depth scaling] By measuring LLM hidden state dynamics and performing control experiments with a teacher-student toy model…
tags:
  - "ICML2026"
  - "LLM Pretraining"
  - "Depth scaling"
  - "ensemble averaging"
  - "residual networks"
  - "Chinchilla"
  - "width-depth tradeoff"
date: 2026-05-08
content_hash: af7460457f0ebea1
---

# Inverse Depth Scaling From Most Layers Being Similar

**Conference**: ICML2026  
**arXiv**: [2602.05970](https://arxiv.org/abs/2602.05970)  
**Code**: https://github.com/liuyz0/DepthScaling  
**Area**: LLM Pre-training / Neural Scaling Laws  
**Keywords**: Depth scaling, ensemble averaging, residual networks, Chinchilla, width-depth tradeoff

## TL;DR
By measuring LLM hidden state dynamics and performing control experiments with a teacher-student toy model, this paper demonstrates that LLM loss is approximately inversely proportional to depth ($\alpha_\ell \approx 1$). This is attributed to a robust but inefficient usage pattern where the vast majority of layers perform functionally similar incremental updates, neutralizing errors through ensemble averaging.

## Background & Motivation
**Background**: Neural scaling laws characterize loss as a power law of the number of parameters $N$ and data volume $D$: $L = c_N/N^{\alpha_N} + c_D/D^{\alpha_D} + L_0$ (Kaplan 2020, Chinchilla 2022). However, most works treat $N$ as a black-box integer, failing to distinguish the individual contributions of width $m$ and depth $\ell$.

**Limitations of Prior Work**: Another line of research (Levine 2020, Liu 2025a, Bordelon 2025b) has begun to disentangle width and depth. However, three conflicting theoretical candidates remain for the specific functional form of how depth impacts loss: (i) **compositional assembly**—each layer learns an abstract hierarchy, and loss depends on the hierarchical structure of the data; (ii) **procedural assembly**—residual networks approximate neural ODEs, and loss follows a power law of discretization error; (iii) **ensemble averaging**—layers act like an ensemble of shallow sub-networks, where loss is driven by the Central Limit Theorem at the $1/\ell$ scale. Empirical studies (Gromov, Sanyal, Men, etc.) repeatedly find that LLMs have many redundant layers that can be deleted or swapped, but a quantitative framework connecting "why redundancy exists" to "how loss scales with depth" is missing.

**Key Challenge**: Theoretically, three candidate mechanisms can produce power laws. Empirically, only qualitative descriptions like "layer redundancy" exist. No prior work has measured the actual $\alpha_\ell$ of LLMs and mapped it to a specific mechanism.

**Goal**: To proceed in two steps: first, measure the depth-specific loss term and its exponent $\alpha_\ell$ in real LLMs; second, design a toy model with controllable mechanisms to map the measured exponent and hidden state signatures back to one of the three candidate theories.

**Key Insight**: The authors observe that the three mechanisms predict different signatures for **hidden state trajectories**. Compositional assembly would manifest as "early stopping" (different inputs stop updating at different depths). Procedural assembly requires neighboring updates to be **correlated** (existence of a first-order derivative for smooth dynamics). Ensemble averaging expects neighboring updates to be **uncorrelated** with step sizes proportional to $1/\ell$. This provides a metric to distinguish mechanisms directly from hidden states.

**Core Idea**: Use the angle between adjacent hidden states $\theta(h_l, h_{l+1})$ and incremental correlation $\theta(\Delta h_l, \Delta h_{l+1})$ as probes. Combine this with a teacher-student toy model where "weight tying vs. independence" toggles between procedural and ensemble ground truths to match LLM empirical signals back to the mechanism. The conclusion is that LLMs primarily utilize ensemble averaging, leading to $L_\ell \propto 1/\ell$.

## Method

### Overall Architecture
The method is divided into two parallel pipelines—"LLM measurement" and "toy model training"—followed by signal comparison:

- **LLM Side**: Run FineWeb on the Pythia model series (e.g., Pythia-410m), calculating $\theta(h_l, h_{l+1})$ per token and per layer. Use PCA clustering to distinguish between "uniform middle updates" and "early stopping" trajectories. Simultaneously, fit a decomposed functional form including a depth term (Eq. 3) on ~200 public Chinchilla model points to extract $\alpha_\ell$.
- **Toy Side**: Construct a "teacher" residual network with depth $\ell^* = 128$ to generate KL targets for a "student" with depth $\ell \in [6, 48]$. Teacher weights can be **tied** (shared across layers) or **independent** (i.i.d. sampling), corresponding to smooth dynamics (procedural ground truth) and random-walk dynamics (ensemble ground truth), respectively. Softmax temperature controls the sharpness of the target distribution.

Finally, the $\alpha_\ell$ values, hidden state step-size curves, and incremental correlations from the toy experiments are matched against the corresponding signatures measured from LLMs. The mechanism whose signatures align with LLMs is identified as the actual mechanism.

### Key Designs

1.  **Depth-Width Decomposed Loss Fitting**:
    - Function: This decomposes the $c_N/N^{\alpha_N}$ term from the traditional Chinchilla form into separate width and depth terms: $L = c_m/m^{\alpha_m} + c_\ell/\ell^{\alpha_\ell} + c_D/D^{\alpha_D} + L_0$, allowing the depth exponent $\alpha_\ell$ to be independently identified.
    - Mechanism: The width term captures error from "limited representation capacity," while the depth term captures error from "limited transformation capacity," assuming they are essentially independent. Cross-terms are assumed to be negligible at higher orders. Using ~200 Chinchilla reconstruction points to minimize the MSE of $\log L$, 7 free parameters are fitted simultaneously, yielding $\alpha_m = 0.98 \pm 0.08$, $\alpha_\ell = 1.2 \pm 0.3$, and $\alpha_D = 0.30 \pm 0.01$ with a mean relative error of 0.4%.
    - Design Motivation: Previous theories either only provided $\log \ell$ corrections or pure power laws that failed to fit real data. This decomposed form with minimal assumptions allows $\alpha_\ell$ to be read directly and verifies the optimal width-depth relationship $m \propto \ell$, which aligns $N^{-1/3}$ with the Chinchilla empirical measurement of 0.34.

2.  **Dual Probes for Hidden State Trajectories**:
    - Function: Simultaneously distinguish the three mechanisms using two metrics: $\theta(h_l, h_{l+1})$ (step size, distinguishing "early exit vs. uniform") and $\theta(\Delta h_l, \Delta h_{l+1})$ (incremental direction correlation, distinguishing smooth dynamics vs. random walk).
    - Mechanism: PCA on $\ell$-dimensional angle vectors per token reveals that 99.6% of tokens cluster into the "uniform middle update" category (aligning with "evenly in the middle" ideal trajectories), while only 0.4% (first tokens of documents) belong to "early exit"—effectively ruling out compositional assembly dominance. Plotting average step size $\langle \theta \rangle_{\mathcal{D}, l}$ against depth shows $\langle \theta \rangle \propto 1/\ell$, consistent with procedural/ensemble expectations. However, the angle between adjacent increments $\theta(\Delta h_l, \Delta h_{l+1})$ is nearly $\pi/2$, indicating updates are almost orthogonal and lack a first-order derivative, contradicting smooth (procedural) dynamics.
    - Design Motivation: A single signature like step size cannot distinguish between procedural and ensemble mechanisms as both yield $1/\ell$. Adding "neighboring correlation" as a second-order signature closes the loop for identification. Both metrics can be read directly from a forward pass without additional training costs.

3.  **Two-Knob Calibration for Teacher-Student Toy**:
    - Function: In a minimal, analytical residual network, knobs for "teacher weight tying vs. independence" and "target distribution temperature" are used to push the student into procedural or ensemble regions, establishing a ground-truth signature library for the three mechanisms.
    - Mechanism: The architecture uses standard Resid + RMSNorm + ReLU² MLP, with teacher depth $\ell^* = 128$ much larger than student $\ell$. Tied weights make the cumulative transformation $h_0^* \to h_{\ell^*}^*$ converge to smooth dynamics; independent weights make it a random walk. Theoretical derivation (Eq. 10-12) shows: under tied weights, discretization error dominates after convergence, yielding loss $\propto 1/\ell^3$ (i.e., $\alpha_\ell = 3$). Under independent weights, layers can only fit the **integral** $\int_0^1 f^*(s)\,\mathrm{d}s$ with $f^\circ(l/\ell)$, with per-layer error $O(1/\ell)$; the sum is governed by the CLT, giving $\|\cdot\| \sim 1/\sqrt{\ell}$, and thus loss $\propto 1/\ell$. Experiments show tied weight $\alpha_\ell$ rises from 1 to 3 during training, while independent weight $\alpha_\ell$ remains stable near 1—matching LLM signatures in step curves, $1/\ell$ scaling, and neighboring orthogonality.
    - Design Motivation: Controlled experiments on LLMs are too expensive and suffer from confounding factors. The toy model strictly maps mechanisms to signatures via closed-form experiments, acting as a bridge to translate "theoretical candidates" into "falsifiable empirical fingerprints."

### Loss & Training
The toy student is trained using Adam for 40,000 steps (extended to 80,000 in Fig. 4). The loss is the KL divergence between student and teacher output distributions (equivalent to cross-entropy minus a constant, preserving scaling behavior). Teacher MLP weights are initialized and scaled by $1/\sqrt{\ell}$ to ensure the cumulative transformation $h_0^* \to h_{\ell^*}^*$ is $O(1)$. Temperature is used to scale logits before softmax. No training is performed on the LLM side; measurements are taken via forward passes on Pythia checkpoints and curve fitting on Chinchilla data points.

## Key Experimental Results

### Main Results: LLM Empirical Decomposed Scaling

| Fitted Term | Exponent | Meaning |
| :--- | :--- | :--- |
| Width $\alpha_m$ | $0.98 \pm 0.08$ | Consistent with Liu 2025a theory ($\approx 1$) |
| Depth $\alpha_\ell$ | $\mathbf{1.2 \pm 0.3}$ | Core finding: $L_\ell \approx 1/\ell$ in LLMs |
| Data $\alpha_D$ | $0.30 \pm 0.01$ | Consistent with original Chinchilla $0.30$ |
| Mean Relative Error of $\log L$ | 0.4% | Quality of fit on 200 Chinchilla points |

### Toy Model Mechanism Comparison

| Teacher Weights | Temperature | Training Steps | Fitted $\alpha_\ell$ | Corresponding Mechanism |
| :--- | :--- | :--- | :--- | :--- |
| Independent ($\rho = 0$) | Any | 40k | $\approx 1$ | Ensemble averaging |
| Tied ($\rho = 1$) | High | 40k | $\to 3$ (post-conv) | Procedural assembly |
| Tied ($\rho = 1$) | Low | 40k | $\approx 1$ (unconv) | Artifact—rises to 3 with more training |
| Tied + Higher-order | High | 80k | $> 3$ | Validates Procedural Mechanism |

### Key Findings
- **PCA splits tokens cleanly**: In Pythia-410m, 99.6% of tokens belong to the "uniform middle update" cluster, while only 0.4% (mostly document starts) are in the "early exit" cluster—effectively rejecting compositional assembly as the dominant mechanism.
- **Heterogeneity of first/last vs. middle layers**: Step sizes $\theta$ in the first and last layers are $\approx \pi/2$ and independent of depth, behaving like "composition." Middle layer step sizes, however, scale strictly by $1/\ell$, indicating the bulk of the network is ensemble-driven.
- **Neighboring increments are nearly orthogonal**: $\theta(\Delta h_l, \Delta h_{l+1})$ is near $\pi/2$, meaning the first-order derivatives required for smooth dynamics (procedural) do not exist. This matches the orthogonality signature of the independent-weight toy student.
- **Insufficient training mimics ensemble**: Under low temperature, students of tied-weight teachers appear to have $\alpha_\ell \approx 1$, but this increases to 3 with extended training. This warns future researchers against concluding mechanisms based on a single training snapshot.
- **Width-depth coupling**: $\alpha_m \approx \alpha_\ell \approx 1$ naturally implies an optimal ratio $m \propto \ell$, giving a total parameter scaling of $N^{-1/3}$, which matches the Chinchilla empirical $0.34$—providing a mechanistic explanation for the Chinchilla exponent.

## Highlights & Insights
- **Quantifying qualitative "redundancy"**: Previous works likely ShortGPT or layer pruning noted "layers are deletable"; this work provides a precise exponent $1/\ell$ for the resulting loss and identifies the CLT as the driver—a key step from descriptive observation to mechanistic explanation.
- **Transferable diagnostic paradigm**: The use of "neighboring angle + neighboring increment correlation" as a mechanism fingerprint can be transferred to diagnose other architectures (e.g., Mamba, MoE) to identify their mechanical regime without expensive retraining.
- **Functional group perspective**: Using causal tracing from ROME, the authors infer a "weak compositionality" where layers cluster into functional groups—ensemble averaging occurs within groups, while division of labor occurs between groups. This provides experimental grounding for the critique that identity-friendly residual architectures do not naturally encourage compositionality.
- **Architectural implications**: Since the issue stems from "residual connections + non-smooth targets," solutions like "recursive depth" (e.g., Geiping 2025) which force multiple uses of the same weights might be the key to bypassing the slow $1/\ell$ scaling.

## Limitations & Future Work
- The decomposition in Eq. (3) is an empirical/theoretical working hypothesis, not derived from first principles; cross-terms like $L_{m\ell}$ are assumed negligible, which might not hold for smaller models.
- Possible mechanisms outside the three candidates cannot be strictly excluded; the paper shows ensemble averaging is the most consistent among existing candidates.
- Hidden state analysis on LLMs only captures statistical average behavior and cannot identify exactly what a specific layer is computing; the "functional group" concept needs better quantification of scale and quantity.
- The toy model excludes attention and embedding training; while authors argue scaling exponents remain invariant under PDE generalization, cross-token coupling could introduce cross-terms.
- Experiments only cover Pythia and Chinchilla families; whether $1/\ell$ dominates in MoE, state-space models, or highly structured data (code/math) remains an open question.
- **Future Directions**: Testing recursive depth, depth-wise weight tying, or introducing explicit hierarchical targets to see if $\alpha_\ell$ can be pushed toward 2 or 3.

## Related Work & Insights
- **vs. Gromov 2024 / Men 2025 (ShortGPT) / Sanyal 2024**: While they empirically found "layers are redundant/deletable," Ours assigns a quantitative scaling $\alpha_\ell \approx 1$ and a mechanistic explanation (ensemble averaging), turning observations into predictions.
- **vs. Liu 2025a (Superposition scaling) / Bordelon 2025b**: They theoretically proposed that width and depth should scale separately; Ours provides direct empirical measurement and numerical matching ($\alpha_m \approx 1$ consistent with Liu 2025a).
- **vs. Csordás 2025 ("Do LLMs use depth efficiently?")**: They found LLMs do not fully exploit compositional data structures; Ours explains "why"—architectural bias and non-smooth targets force the network into the inefficient ensemble region.
- **vs. Sander 2022 / Chizat 2025 (residual ↔ ODE)**: Previous works used worst-case error bounds to analyze residual networks as ODE discretizations; Ours shows real LLMs are not in the worst-case but in a typical behavior region dominated by the CLT.
- **vs. Lad 2024 ("Stages of inference")**: Their stages can be integrated with the "functional group" picture in this paper, providing a starting point for quantifying "intra-group ensemble + inter-group division of labor."

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Ties together layer redundancy, $1/\ell$ scaling, and ensemble averaging into a quantitative framework, explaining the Chinchilla exponent.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three-way evidence loop (Chinchilla fitting, Pythia hidden states, 4-knob toy model), though lacks validation on modern MoE or non-transformer dense models.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear argumentation chain from mechanisms to probes to matching.
- Value: ⭐⭐⭐⭐⭐ Provides diagnostic tools and architectural directions (recursive depth, hierarchical targets) to make depth truly efficient.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Scaling Depth Capacity via Zero/One-Layer Model Expansion](scaling_depth_capacity_via_zeroone-layer_model_expansion.md)
- [\[NeurIPS 2025\] Scaling Embedding Layers in Language Models](../../NeurIPS2025/llm_pretraining/scaling_embedding_layers_in_language_models.md)
- [\[ICML 2026\] Model Merging Scaling Laws in Large Language Models](model_merging_scaling_laws_in_large_language_models.md)
- [\[ICML 2026\] Dropout Universality: Scaling Laws and Optimal Scheduling at the Edge-of-Chaos](dropout_universality_scaling_laws_and_optimal_scheduling_at_the_edge-of-chaos.md)
- [\[ICML 2026\] InfoLaw: Information Scaling Laws for Large Language Models with Quality-Weighted Mixture Data and Repetition](infolaw_information_scaling_laws_for_large_language_models_with_quality-weighted.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ICML 2026\] Scaling Depth Capacity via Zero/One-Layer Model Expansion](scaling_depth_capacity_via_zeroone-layer_model_expansion.md)
- [\[NeurIPS 2025\] Scaling Embedding Layers in Language Models](../../NeurIPS2025/llm_pretraining/scaling_embedding_layers_in_language_models.md)
- [\[ICML 2026\] Dropout Universality: Scaling Laws and Optimal Scheduling at the Edge-of-Chaos](dropout_universality_scaling_laws_and_optimal_scheduling_at_the_edge-of-chaos.md)
- [\[ICML 2026\] InfoLaw: Information Scaling Laws for Large Language Models with Quality-Weighted Mixture Data and Repetition](infolaw_information_scaling_laws_for_large_language_models_with_quality-weighted.md)
- [\[ICLR 2026\] Implicit Bias and Loss of Plasticity in Matrix Completion: Depth Promotes Low-Rank](../../ICLR2026/llm_pretraining/implicit_bias_and_loss_of_plasticity_in_matrix_completion_depth_promotes_low-ran.md)

</div>

<!-- RELATED:END -->
