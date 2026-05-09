---
title: >-
  [Paper Note] Toward Consistent World Models with Multi-Token Prediction and Latent Semantic Enhancement
description: >-
  [ACL 2026][LLM Safety][multi-token prediction] This paper provides a theoretical analysis of how Multi-Token Prediction (MTP) induces representational contractiveness through gradient coupling mechanisms to promote the emergence of belief states. It simultaneously reveals a "structural hallucination" problem in MTP—namely, illegal shortcuts in the latent space—and proposes the LSE-MTP framework, which anchors predictions to true latent state trajectories via latent consistency loss and semantic anchoring loss. The approach significantly improves path legality and robustness on synthetic graphs and real-world Manhattan taxi navigation tasks.
tags:
  - ACL 2026
  - LLM Safety
  - multi-token prediction
  - world models
  - belief states
  - structural hallucination
  - latent semantic enhancement
date: 2026-05-08
content_hash: 3862d3fe180b091b
---

# Toward Consistent World Models with Multi-Token Prediction and Latent Semantic Enhancement

**Conference**: ACL 2026
**arXiv**: [2604.06155](https://arxiv.org/abs/2604.06155)
**Code**: [GitHub](https://github.com/QiminZhong/LSE-MTP)
**Area**: World Models / LLM Representation Learning
**Keywords**: multi-token prediction, world models, belief states, structural hallucination, latent semantic enhancement

## TL;DR
This paper provides a theoretical analysis of how Multi-Token Prediction (MTP) induces representational contractiveness through gradient coupling mechanisms to promote the emergence of belief states. It simultaneously reveals a "structural hallucination" problem in MTP—namely, illegal shortcuts in the latent space—and proposes the LSE-MTP framework, which anchors predictions to true latent state trajectories via latent consistency loss and semantic anchoring loss. The approach significantly improves path legality and robustness on synthetic graphs and real-world Manhattan taxi navigation tasks.

## Background & Motivation

**Background**: World models—the ability to simulate state evolution within an environment—are a hallmark of intelligent behavior. Whether LLMs can develop consistent internal world models through next-token prediction (NTP) is a central debate. The optimization objective of NTP is local, focusing solely on the conditional probability of the next token, which enables models to capture surface-level patterns but hinders the sustained internalization of deep global structure.

**Limitations of Prior Work**: MTP provides a more structured training signal by simultaneously predicting multiple future tokens, promoting better representation learning. However, empirical observations reveal that even when MTP-trained models achieve accurate token-level predictions, their latent state evolution may violate fundamental environmental constraints—intermediate steps are implicitly skipped, producing shortcuts that are invalid under true dynamics.

**Key Challenge**: The contractiveness induced by MTP is "outcome-driven"—it constrains representational alignment of future outcomes but neglects the physical legality of intermediate states. Optimizing for long-range prediction without explicitly constraining trajectory legality incentivizes models to prioritize outcomes over process. This is the root cause of "structural hallucination."

**Goal**: (1) Theoretically characterize the gradient coupling mechanism and contractiveness of MTP; (2) expose the structural hallucination problem; (3) propose a solution that bridges the gap between discrete token supervision and continuous state representations.

**Key Insight**: The Neural Tangent Kernel (NTK) framework is employed to analyze the gradient flow dynamics of MTP in the linearized regime, deriving a contractiveness theorem and cross-path gradient coupling effects.

**Core Idea**: Two auxiliary losses are added on top of MTP—a latent consistency loss (aligning predicted representations to true future latent states) and a semantic anchoring loss (aligning to target token embeddings)—transforming MTP's contractive force from "blind outcome alignment" into "trajectory-aware alignment."

## Method

### Overall Architecture
LSE-MTP augments the standard MTP architecture with two auxiliary losses. Given backbone hidden state $h_n$, $K$ step-specific transformation layers $\mathcal{T}_\phi^{(k-1)}$ generate multi-step prediction representations $\hat{h}_{n,k}$. The training objective comprises: (1) multi-step cross-entropy loss (standard MTP); (2) latent consistency loss (aligning predicted representations to future backbone states); (3) semantic anchoring loss (aligning predicted representations to target token embeddings). At inference time, all transformation layers and auxiliary losses are discarded, and decoding proceeds with standard autoregressive NTP.

### Key Designs

1. **Gradient Coupling Analysis**:

    - *Function*: Theoretically explains why and how MTP promotes the emergence of belief states.
    - *Mechanism*: Through NTK analysis, it is proven that under MTP, two hidden states $h_1 \sim_k h_2$ that are $k$-step future equivalent (sharing the $k$-th step target but with different next-step targets) exhibit a "prediction coupling" effect—training on one trajectory increases prediction confidence for the corresponding tokens of another trajectory (Theorem 2). It is further proven that, under the full-rank transformation Jacobian condition, MTP induces stable contractiveness $\dot{\mathcal{D}} \leq 0$, causing hidden states sharing a future to locally converge to a unified belief state (Lemma 1).
    - *Design Motivation*: Existing MTP research lacks theoretical explanations for its representation learning effects. This analysis fills that gap and theoretically predicts the risk of structural hallucination.

2. **Latent Consistency Loss**:

    - *Function*: Enforces predicted representations to follow the true latent state trajectory, preventing illegal shortcuts.
    - *Mechanism*: $\mathcal{L}_{latent} = \sum_{k=2}^{K} \mathbb{E}_n \|\hat{h}_{n,k} - h_{n+k-1}\|_2^2$, aligning the $k$-step prediction representation $\hat{h}_{n,k}$ to the actual backbone hidden state at step $n+k-1$. This ensures that the evolution path of predicted representations is consistent with true encodings.
    - *Design Motivation*: Standard MTP transformation layers can learn arbitrary latent space mappings as long as final token prediction is correct, which permits illegal shortcuts. The latent consistency loss enforces the transformation layers to simulate true state transitions.

3. **Semantic Anchoring Loss**:

    - *Function*: Anchors predicted representations to the semantic space of target tokens, enhancing the semantic interpretability of representations.
    - *Mechanism*: $\mathcal{L}_{semantic} = \sum_{k=2}^{K} \mathbb{E}_n \|\hat{h}_{n,k} - \text{sg}(\mathbf{E}(u_{n+k}))\|_2^2$, where $\text{sg}(\cdot)$ denotes stop-gradient and $\mathbf{E}(\cdot)$ is the model's embedding layer. Predicted representations are aligned to the embedding vectors of target tokens.
    - *Design Motivation*: The embedding layer encodes the semantic information of tokens; anchoring to the embedding space prevents predicted representations from drifting into semantically meaningless regions.

### Loss & Training
The total loss is $\mathcal{L}_{total} = \mathcal{L}_{ce} + \lambda_l \mathcal{L}_{latent} + \lambda_s \mathcal{L}_{semantic}$, with default values $\lambda_l = \lambda_s = 0.1$. A 6-layer Transformer (6 heads, hidden dimension 120) is trained for 20,000 iterations on 100-node graphs. All auxiliary components are discarded at inference, preserving standard autoregressive decoding.

## Key Experimental Results

### Main Results
Representational alignment on ER graphs and USG (urban street graphs) (Structure Gain = Sim(F) − random baseline):

| Model | ER k=2 Gain | ER k=3 Gain | USG k=2 Gain | USG k=3 Gain |
|-------|-------------|-------------|--------------|--------------|
| 1TP (NTP) | 0.027 | 0.022 | -0.005 | 0.018 |
| 2TP (MTP) | 0.210 | 0.074 | 0.214 | 0.066 |
| 4TP (MTP) | 0.176 | 0.162 | 0.178 | 0.180 |
| 4TP (LSE-MTP) | - | - | - | - |

Belief compression (cosine similarity of hidden states sharing the same goal and position):

| Model | K | ER G=,P= | USG G=,P= |
|-------|---|----------|-----------|
| NTP | 1 | 0.29 | 0.22 |
| MTP | 4 | 0.44 | 0.32 |
| LSE-MTP | 4 | **0.46** | **0.38** |

### Ablation Study

| Configuration | ER ISP ↓ | ER Legal Prob ↑ | USG ISP ↓ | USG Legal Prob ↑ |
|---------------|----------|-----------------|-----------|------------------|
| NTP (1TP) | 2.7e-5 | 0.995 | 2.2e-5 | 0.998 |
| MTP (3TP) | 7.8e-5 | 0.992 | 7.3e-5 | 0.994 |
| LSE-MTP (3TP) | **2.1e-5** | **0.996** | **1.8e-5** | **0.998** |

### Key Findings
- MTP reliably induces representational alignment (Structure Gain improvement of 8–21×), corroborating the theoretically predicted contractiveness.
- MTP simultaneously increases the probability of illegal shortcut paths (ISP increases from 2.7e-5 to 7.8e-5), empirically validating the existence of structural hallucination.
- LSE-MTP reduces ISP to below the NTP baseline while preserving the belief compression advantage, successfully resolving structural hallucination.
- On real-world Manhattan taxi navigation data, LSE-MTP substantially improves path legality and robustness to perturbations.

## Highlights & Insights
- The theoretical analysis of MTP is particularly rigorous—gradient coupling across paths and contractiveness are formally derived via the NTK framework, providing stronger justification than purely empirical observations. The intuition behind Theorem 2's "prediction coupling" (training one path indirectly benefits another path sharing a future) is especially elegant.
- The concept of "structural hallucination" is highly valuable—it exposes a fundamental limitation of MTP: token-level accuracy does not imply trajectory-level consistency. This serves as an important warning for all large model training pipelines employing MTP.
- The linear model experimental design is notably ingenious—five-dimensional orthogonal basis vectors are used to construct a minimal case that transparently demonstrates how gradient coupling causes unobserved transitions to be reinforced.

## Limitations & Future Work
- The theoretical analysis is grounded in the linearized regime (lazy training); practical deep networks may deviate from this assumption.
- Experiments are conducted solely on graph navigation tasks; validation on natural language tasks is absent.
- LSE-MTP requires true latent states as training signals; alternative approaches are needed for settings where ground-truth states are unavailable.
- Future work may explore applying LSE-MTP to actual LLM pretraining.

## Related Work & Insights
- **vs. Standard MTP (Gloeckle et al., 2024)**: Standard MTP performs multi-step token prediction without constraining the latent space, leading to structural hallucination; LSE-MTP introduces trajectory-level constraints.
- **vs. DreamerV3**: DreamerV3 learns world models for RL via latent state prediction; LSE-MTP brings analogous ideas into the MTP framework for language models.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — Both the theoretical analysis of MTP and the structural hallucination concept are original contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers synthetic graphs and real-world data with sufficient theoretical validation, but lacks NLP task experiments.
- Writing Quality: ⭐⭐⭐⭐⭐ — Theoretical derivations are rigorous, experimental design is elegant, and the narrative is clear.
- Value: ⭐⭐⭐⭐⭐ — Provides both theoretical foundations and practical methods for understanding and improving MTP.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] DUET: Dual Execution for Test Output Prediction with Generated Code and Pseudocode](duet_dual_execution_for_test_output_prediction_with_generated_code_and_pseudocod.md)
- [\[CVPR 2026\] Multi-Paradigm Collaborative Adversarial Attack Against Multi-Modal Large Language Models](../../CVPR2026/llm_safety/multi-paradigm_collaborative_adversarial_attack_against_multi-modal_large_langua.md)
- [\[ICLR 2026\] Measuring Physical-World Privacy Awareness of Large Language Models: An Evaluation Benchmark](../../ICLR2026/llm_safety/measuring_physical-world_privacy_awareness_of_large_language_models_an_evaluatio.md)
- [\[ACL 2026\] AGSC: Adaptive Granularity and Semantic Clustering for Uncertainty Quantification in Long-text Generation](agsc_adaptive_granularity_and_semantic_clustering_for_uncertainty_quantification.md)
- [\[ACL 2026\] XMark: Reliable Multi-Bit Watermarking for LLM-Generated Texts](xmark_reliable_multi-bit_watermarking_for_llm-generated_texts.md)

</div>

<!-- RELATED:END -->
