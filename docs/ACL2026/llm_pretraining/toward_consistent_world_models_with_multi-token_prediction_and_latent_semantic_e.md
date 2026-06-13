---
title: >-
  [Paper Note] Toward Consistent World Models with Multi-Token Prediction and Latent Semantic Enhancement
description: >-
  [ACL 2026][LLM Pretraining][Multi-token prediction] This paper theoretically analyzes how Multi-Token Prediction (MTP) induces representation contractivity and facilitates the emergence of belief states through a gradien…
tags:
  - "ACL 2026"
  - "LLM Pretraining"
  - "Multi-token prediction"
  - "World models"
  - "Belief states"
  - "Structural hallucination"
  - "Latent semantic enhancement"
date: 2026-05-08
content_hash: aa437ff917251f42
---

# Toward Consistent World Models with Multi-Token Prediction and Latent Semantic Enhancement

**Conference**: ACL 2026  
**arXiv**: [2604.06155](https://arxiv.org/abs/2604.06155)  
**Code**: [GitHub](https://github.com/QiminZhong/LSE-MTP)  
**Area**: World Models / LLM Representation Learning  
**Keywords**: Multi-token prediction, World models, Belief states, Structural hallucination, Latent semantic enhancement

## TL;DR
This paper theoretically analyzes how Multi-Token Prediction (MTP) induces representation contractivity and facilitates the emergence of belief states through a gradient coupling mechanism. Simultaneously, it reveals the "structural hallucination" problem in MTP (illegal shortcuts in the latent space) and proposes the LSE-MTP framework. By employing latent consistency and semantic anchoring losses, the framework anchors predictions to the true latent state trajectories, significantly improving path legality and robustness in synthetic graphs and real Manhattan taxi navigation tasks.

## Background & Motivation

**Background**: World models (the ability to simulate state evolution within an environment) are hallmarks of intelligent behavior. Whether Large Language Models (LLMs) can develop consistent internal world models via next-token prediction (NTP) is a central debate. The optimization objective of NTP is local—focusing only on the conditional probability of the next token—which makes models adept at capturing surface patterns but struggling to persistently internalize deep global structures.

**Limitations of Prior Work**: Multi-token prediction (MTP) provides more structured training signals by predicting multiple future tokens simultaneously, promoting better representation learning. However, practical observations indicate that even when MTP-trained models are accurate at the token level, their latent state evolution may violate essential environmental constraints—intermediate steps are implicitly skipped, creating shortcuts that are invalid under real dynamics.

**Key Challenge**: The contractivity of MTP is "outcome-driven"—it constrains the alignment of representations for future outcomes but ignores the physical legality of intermediate states. Optimizing long-range predictions without explicitly constraining trajectory legality motivates models to prioritize results over processes. This is the root cause of "structural hallucinations."

**Goal**: (1) Theoretically characterize the gradient coupling mechanism and its contractivity in MTP; (2) Reveal the structural hallucination problem; (3) Propose solutions to bridge the gap between discrete token supervision and continuous state representations.

**Key Insight**: The authors analyze the gradient flow dynamics of MTP under the Neural Tangent Kernel (NTK) framework in a linearized regime, deriving contractivity theorems and cross-path gradient coupling effects.

**Core Idea**: Based on MTP, the authors add a latent consistency loss (aligning predicted representations with ground-truth future hidden states) and a semantic anchoring loss (aligning with target token embeddings). This transforms the contractive force of MTP from "blind outcome alignment" into "trajectory-aware alignment."

## Method

### Overall Architecture
LSE-MTP adds two auxiliary losses to the standard MTP architecture. Given a backbone latent state $h_n$, $K$ step-specific transition layers $\mathcal{T}_\phi^{(k-1)}$ generate multi-step predicted representations $\hat{h}_{n,k}$. The training objective includes: (1) multi-step cross-entropy loss (standard MTP); (2) latent consistency loss (aligning predicted representations with future backbone states); (3) semantic anchoring loss (aligning predicted representations with target token embeddings). At inference time, all transition layers and auxiliary losses are discarded, and decoding maintains standard autoregressive NTP.

### Key Designs

1. **Gradient Coupling Analysis**:

    - **Function**: Theoretically explains why and how MTP promotes the emergence of belief states.
    - **Mechanism**: Through NTK analysis, the authors prove that under MTP, two latent states $h_1 \sim_k h_2$ that are $k$-step future equivalent (sharing the same $k$-th step target but having different next-step targets) exhibit a "prediction coupling" effect—training one trajectory enhances the prediction confidence for the corresponding token in the other trajectory (Theorem 2). Furthermore, it is proven that under full-rank transition Jacobian conditions, MTP induces a stable contractive force $\dot{\mathcal{D}} \leq 0$, causing latent states sharing the same future to locally converge into a unified belief state (Lemma 1).
    - **Design Motivation**: Existing MTP research lacks theoretical explanations for its representation learning effects. This analysis fills the gap while theoretically predicting the risk of structural hallucinations.

2. **Latent Consistency Loss**:

    - **Function**: Forces predicted representations to follow the ground-truth latent state trajectory, preventing illegal shortcuts.
    - **Mechanism**: $\mathcal{L}_{latent} = \sum_{k=2}^{K} \mathbb{E}_n \|\hat{h}_{n,k} - h_{n+k-1}\|_2^2$, which aligns the $k$-step predicted representation $\hat{h}_{n,k}$ with the actual backbone latent state at step $n+k-1$. This ensures that the evolution path of the predicted representation is consistent with the true encoding.
    - **Design Motivation**: Standard MTP transition layers can learn arbitrary latent space mappings as long as the final token prediction is correct, allowing illegal shortcuts. Latent consistency loss forces the transition layers to simulate real state transitions.

3. **Semantic Anchoring Loss**:

    - **Function**: Anchors the predicted representation to the semantic space of the target token, enhancing the semantic interpretability of the representation.
    - **Mechanism**: $\mathcal{L}_{semantic} = \sum_{k=2}^{K} \mathbb{E}_n \|\hat{h}_{n,k} - \text{sg}(\mathbf{E}(u_{n+k}))\|_2^2$, where $\text{sg}(\cdot)$ is stop-gradient and $\mathbf{E}(\cdot)$ is the model's embedding layer. This aligns the predicted representation with the embedding vector of the target token.
    - **Design Motivation**: The embedding layer encodes semantic information of tokens. Anchoring to the embedding space prevents the predicted representation from drifting into semantically meaningless regions.

### Loss & Training
The total loss is $\mathcal{L}_{total} = \mathcal{L}_{ce} + \lambda_l \mathcal{L}_{latent} + \lambda_s \mathcal{L}_{semantic}$, with defaults $\lambda_l = \lambda_s = 0.1$. A 6-layer Transformer (6 heads, hidden dimension 120) is used on a 100-node graph, trained for 20,000 iterations. During inference, all auxiliary components are discarded to maintain standard autoregressive decoding.

## Key Experimental Results

### Main Results
Representation alignment on ER graphs and USG (Urban Street Graphs) (Structure Gain = Sim(F) - random baseline):

| Model | ER k=2 Gain | ER k=3 Gain | USG k=2 Gain | USG k=3 Gain |
|-------|-------------|-------------|--------------|--------------|
| 1TP (NTP) | 0.027 | 0.022 | -0.005 | 0.018 |
| 2TP (MTP) | 0.210 | 0.074 | 0.214 | 0.066 |
| 4TP (MTP) | 0.176 | 0.162 | 0.178 | 0.180 |
| 4TP (LSE-MTP) | - | - | - | - |

Belief compression (cosine similarity of latent states with same target and position):

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
- MTP indeed induces representation alignment (Structure Gain increases by 8-21×), supporting the contractivity predicted by the theory.
- However, MTP simultaneously increases the probability of illegal shortcuts (ISP increases from 2.7e-5 to 7.8e-5), validating the existence of structural hallucinations.
- LSE-MTP successfully resolves structural hallucinations, reducing ISP below NTP levels while maintaining the advantages of belief compression.
- On real Manhattan taxi navigation data, LSE-MTP significantly improves path legality and robustness to perturbations.

## Highlights & Insights
- The theoretical analysis of MTP is profound—using the NTK framework to rigorously derive cross-path gradient coupling and contractivity is more convincing than empirical observation alone. The intuition of "prediction coupling" in Theorem 2 (training one path indirectly helps another sharing the same future) is particularly insightful.
- The concept of "structural hallucination" is highly valuable—it reveals a fundamental limitation of MTP: token-level accuracy does not equate to trajectory-level consistency. This is an important warning for training large models using MTP.
- The design of the linear model experiment is clever—using 5-dimensional orthogonal basis vectors to construct a minimal case transparently demonstrates how gradient coupling lead to the reinforcement of unobserved transitions.

## Limitations & Future Work
- The theoretical analysis is based on a linearized regime (lazy training); actual deep networks might deviate from this assumption.
- Experiments were only validated on graph navigation tasks and were not tested on natural language tasks.
- LSE-MTP requires ground-truth latent states as training signals, requiring alternatives for scenarios where true states are unavailable.
- Future work could explore methods to apply LSE-MTP in large-scale LLM pre-training.

## Related Work & Insights
- **vs. Standard MTP (Gloeckle et al., 2024)**: Standard MTP only performs multi-step token prediction without latent space constraints, leading to structural hallucinations; LSE-MTP adds trajectory-level constraints.
- **vs. DreamerV3**: DreamerV3 learns world models in RL using latent state prediction; LSE-MTP introduces similar ideas into the MTP framework for language models.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The theoretical analysis of MTP and the concept of structural hallucination are original contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐ Sufficient validation with synthetic graphs and real data; however, experiments on NLP tasks are missing.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous theoretical derivation, clever experimental design, and clear narrative.
- Value: ⭐⭐⭐⭐⭐ Provides a theoretical foundation and practical methods for understanding and improving MTP.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Next Semantic Scale Prediction via Hierarchical Diffusion Language Models](../../NeurIPS2025/llm_pretraining/next_semantic_scale_prediction_via_hierarchical_diffusion_language_models.md)
- [\[ACL 2026\] Compact Example-Based Explanations for Language Models](compact_example-based_explanations_for_language_models.md)
- [\[ICLR 2026\] CHAMMI-75: Pre-training multi-channel models with heterogeneous microscopy images](../../ICLR2026/llm_pretraining/chammi-75_pre-training_multi-channel_models_with_heterogeneous_microscopy_images.md)
- [\[ACL 2026\] SCRIPT: A Subcharacter Compositional Representation Injection Module for Korean Pre-Trained Language Models](script_a_subcharacter_compositional_representation_injection_module_for_korean_p.md)
- [\[ACL 2026\] Fine-tuning vs. In-context Learning in Large Language Models: A Formal Language Learning Perspective](fine-tuning_vs_in-context_learning_in_large_language_models_a_formal_language_le.md)

</div>

<!-- RELATED:END -->
