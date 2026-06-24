---
title: >-
  [Paper Note] HD-NDEs: Neural Differential Equations for Hallucination Detection in LLMs
description: >-
  [ACL 2025][Hallucination Detection][Neural ODE] This paper represents the first attempt to apply Neural Differential Equations (Neural DEs) to LLM hallucination detection. By modeling the continuous trajectories of token activations in the hidden space, the proposed method systematically evaluates the truthfulness of statements, outperforming the state-of-the-art (SOTA) on the True-False dataset by over 14% in AUC-ROC.
tags:
  - "ACL 2025"
  - "Hallucination Detection"
  - "Neural ODE"
  - "Neural CDE"
  - "Neural SDE"
  - "Hidden State Trajectories"
  - "Classifiers"
date: 2026-05-08
content_hash: cc32c7e50e22bfc1
---

# HD-NDEs: Neural Differential Equations for Hallucination Detection in LLMs

**Conference**: ACL 2025  
**arXiv**: [2506.00088](https://arxiv.org/abs/2506.00088)  
**Area**: Hallucination Detection  
**Keywords**: Hallucination Detection, Neural ODE, Neural CDE, Neural SDE, Hidden State Trajectories, Classifiers  

## TL;DR

This paper represents the first attempt to apply Neural Differential Equations (Neural DEs) to LLM hallucination detection. By modeling the continuous trajectories of token activations in the hidden space, the proposed method systematically evaluates the truthfulness of statements, outperforming the state-of-the-art (SOTA) on the True-False dataset by over 14% in AUC-ROC.

## Background & Motivation

**Hallucination as a Core Challenge in LLM Deployment**: The generation of inaccurate or non-factual statements by LLMs remains a major obstacle to their practical application, potentially leading to user churn or legal risks.

**Limitations of Prior Work**:
   - **Evidence-based methods** (retrieving external knowledge for verification): Computationally heavy and time-consuming, making them unsuitable for high-throughput scenarios.
   - **Logit-based methods** (e.g., AvgProb, AvgEnt): Estimating sentence-level uncertainty through token-level uncertainty estimation, which yields coarse granularity.
   - **Consistency-based methods** (e.g., SelfCheckGPT): Judging consistency through multiple generations, resulting in low efficiency.
   - **Classification-based methods** (e.g., SAPLMA): Efficient but rely solely on the hidden state of the final token, leading to degraded performance when non-factual information appears earlier or in the middle of a sequence.

**Insufficiency of the Final Token**: PCA analysis reveals that for correct and incorrect answers to the same question, the hidden state activations of the final few tokens are almost identical (as they share the same ending tokens), whereas the divergence primarily manifests in the middle of the sequence. This indicates a necessity to utilize the hidden state information across the **entire sequence**.

**Theoretical Alignment with Neural DEs**:
   - Transformers can be mathematically interpreted as numerical solvers for differential equations (Lu et al., 2019).
   - Neural DEs excel in time-series modeling, making them naturally suited for modeling the dynamic evolution of token-level hidden states.
   - The token generation process can be viewed as a continuous trajectory in the hidden space.

## Method

### Overall Architecture

The workflow of HD-NDEs is as follows:

1. **Feature Extraction**: Feed the statement into the LLM and extract the embedding for each token at a specified hidden layer: $\boldsymbol{x} = (x_0, x_1, ..., x_n) \in \mathbb{R}^{d_x}$.
2. **Dimension Reduction**: Project high-dimensional embeddings into a lower-dimensional space using PCA: $\boldsymbol{y} = (y_0, y_1, ..., y_n) \in \mathbb{R}^{d_y}$.
3. **Neural DE Solving**: Model the hidden space trajectory $\boldsymbol{z} = (z_0, z_1, ..., z_n)$ using Neural ODEs, CDEs, or SDEs.
4. **Classification**: Extract $z^*$ from the hidden states and output the hallucination probability $P(\xi=1|\boldsymbol{x})$ through a linear classifier.

### Three Neural DE Variants

**Neural ODEs**: Model smooth, continuous-time dynamics using deterministic differential equations:

$$z(t) = z(0) + \int_0^t f(s, z(s); \theta_f) ds$$

with initial condition $z(0) = h(\boldsymbol{y}; \theta_h)$, where $f$ and $h$ are learnable neural networks. Solved using the fourth-order Runge-Kutta (RK4) method.

**Neural CDEs**: Introduce a control signal to guide system evolution, addressing the limitation where Neural ODEs are solely determined by initial conditions:

$$z(t) = z(0) + \int_0^t f(s, z(s); \theta_f) dY(s)$$

The control path $Y(t)$ is constructed by interpolating the time-series data using natural cubic splines or Hermite splines.

**Neural SDEs**: Incorporate a stochastic noise term to capture uncertainty in the system:

$$z(t) = z(0) + \int_0^t f(s, z(s); \theta_f) ds + \int_0^t g(s, z(s); \theta_g) dW(s)$$

where $\{W_t\}_{t \geq 0}$ represents Brownian motion, solved using the Euler-Maruyama method.

### Key Designs

Features $z^*$ are extracted from the hidden state sequence $\boldsymbol{z}$ via a function $k(\theta_k)$, and are then passed through a simple linear layer and a sigmoid function to output the hallucination probability. The parameter size of the entire classifier is extremely small.

### Loss & Training

The adjoint method is used for gradient computation, enabling parameter updates from the final state to the initial state with constant memory cost.

## Key Experimental Results

### Experimental Setup

- **5 Datasets**: Company*, Fact*, City*, Invention*, True-False
- **6 LLMs**: LLama-2-7B, LLama-2-13B, Alpaca-13B, Vicuna-13B, Mistral-7B-v0.3, Gemma-2-9B
- **Baseline Methods**: P(True), AvgProb, AvgEnt, EUBHD, SAPLMA, MIND, Probe@Exact
- **Evaluation Metrics**: AUC-ROC

### Main Results

**Company Dataset**:

| Method | LLama-2-7B | LLama-2-13B | Vicuna-13B | Gemma-2-9B |
|------|------------|-------------|------------|------------|
| SAPLMA | 54.0 | 58.2 | 68.2 | 64.8 |
| MIND | 56.4 | 60.3 | 69.8 | 65.9 |
| Neural CDEs | **65.9**| **72.8** | **79.8** | **73.6** |
| Neural SDEs | 73.8 | 78.4 | 72.3 | 72.8 |

**City Dataset**:

| Method | LLama-2-7B | LLama-2-13B | Vicuna-13B | Gemma-2-9B |
|------|------------|-------------|------------|------------|
| SAPLMA | 60.0 | 69.3 | 64.5 | 64.7 |
| Neural ODEs | 73.0 | 82.3 | 73.2 | 72.4 |
| Neural CDEs | **75.7**| 80.6 | **80.1** | **77.2** |

### Key Data Highlights

- **True-False Dataset**: HD-NDEs (specifically the Neural CDEs variant) outperforms SOTA methods like SAPLMA by **over 14%** in AUC-ROC.
- Neural CDEs generally achieve the best performance because the control signal mechanism leverages temporal information in the sequence more effectively.
- Neural SDEs outperform Neural CDEs on certain datasets because the stochastic term helps capture the intrinsic uncertainty within the generation process.
- Even the simplest Neural ODEs consistently outperform all classification-based baselines.

### Cross-Model Consistency

- HD-NDEs consistently outperform baseline methods across all 6 LLMs, demonstrating excellent cross-model generalization ability.
- As the model scale increases (7B $\rightarrow$ 13B), the improvements gained by HD-NDEs typically become more pronounced.

## Highlights & Insights

1. **Clear Theoretical Motivation**: Directly demonstrates failure cases of using only the final token for hallucination detection via PCA visualization, making the motivation compelling.
2. **Deep Connection between Neural DEs and Transformers**: Leverages the theory that Transformers can be mathematically structuralized as ODE solvers, providing a solid foundation for utilizing Neural DEs in LLM analysis.
3. **Simple and Efficient Design**: The classifier is merely a simple linear layer, primarily learning to make decisions within the hidden space modeled by the Neural DE, eliminating the need to train massive models.
4. **Complementarity of the Three DE Variants**: ODEs capture deterministic dynamics, CDEs introduce external control, and SDEs model stochasticity, successfully covering the requirements of diverse scenarios.

## Limitations & Future Work

1. **Requires White-Box Access**: Intermediate-layer hidden states of the LLM must be accessible, which makes it inapplicable to black-box API models (such as GPT-4, Claude, etc.).
2. **Information Loss from PCA Dimension Reduction**: Projecting high-dimensional embeddings to lower-dimensional spaces via PCA may discard crucial information.
3. **Sentence-Level Detection Granularity**: Can only determine whether the entire statement contains a hallucination, failing to localize exactly which tokens are inaccurate.
4. **Training Data Requirements**: Labeled dataset collection is required for each LLM to train the Neural DE parameters separately, and cross-model transferability has not yet been validated.
5. **Computational Overhead**: Solving Neural DEs (especially with RK4 and the adjoint method) incurs a higher computational cost compared to simple classifiers.

## Related Work & Insights

- **Hallucination Detection**: SAPLMA (Azaria and Mitchell, 2023) trains a classifier utilizing final token hidden states; MIND and Probe@Exact improve feature extraction.
- **Neural DEs**: Chen et al. (2018) proposed Neural ODEs for continuous-depth networks; Kidger et al. (2020) introduced Neural CDEs to handle time-series data.
- **LLMs and Dynamical Systems**: Lu et al. (2019) first analogized Transformers to ODEs.

## Rating

⭐⭐⭐⭐ — Highly novel, representing the first attempt to apply Neural DEs to hallucination detection, backed by a clear theoretical motivation and significant experimental improvements (14%+). While the restriction to white-box access is a major bottleneck, it remains highly valuable for open-source LLM scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] ICR Probe: Tracking Hidden State Dynamics for Reliable Hallucination Detection in LLMs](icr_probe_tracking_hidden_state_dynamics_for_reliable_hallucination_detection_in.md)
- [\[ICLR 2026\] Neural Message-Passing on Attention Graphs for Hallucination Detection](../../ICLR2026/hallucination/neural_message-passing_on_attention_graphs_for_hallucination_detection.md)
- [\[NeurIPS 2025\] Robust Hallucination Detection in LLMs via Adaptive Token Selection](../../NeurIPS2025/hallucination/robust_hallucination_detection_in_llms_via_adaptive_token_selection.md)
- [\[ACL 2025\] DRAG: Distilling RAG for SLMs from LLMs to Transfer Knowledge and Mitigate Hallucination](drag_distilling_rag_slm.md)
- [\[ACL 2025\] Automated Explanation Generation and Hallucination Detection for Heritage Image Retrieval](automated_explanation_generation_and_hallucination_detection_for_heritage_image_.md)

</div>

<!-- RELATED:END -->
