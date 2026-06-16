---
title: >-
  [Paper Note] Toward Consistent World Models with Multi-Token Prediction and Latent Semantic Enhancement
description: >-
  [ACL 2026][Pretraining][World Models] This paper theoretically analyzes how Multi-Token Prediction (MTP) induces representation contractivity through a gradient coupling mechanism, facilitating the emergence of belief states. However, it reveals a "structural hallucination" issue in MTP (illegal shortcuts in latent space) and proposes the LSE-MTP framework
tags:
  - ACL 2026
  - Pretraining
  - World Models
date: 2026-05-08
content_hash: 92251bd22830b1f9
---
# Toward Consistent World Models with Multi-Token Prediction and Latent Semantic Enhancement

**Conference**: ACL 2026  
**arXiv**: [2604.06155](https://arxiv.org/abs/2604.06155)  
**Code**: [GitHub](https://github.com/QiminZhong/LSE-MTP)  
**Area**: World Models / LLM Representation Learning  
**Keywords**: Multi-token prediction, World models, Belief states, Structural hallucinations, Latent semantic enhancement

## TL;DR
This paper theoretically analyzes how Multi-Token Prediction (MTP) induces representation contractivity through a gradient coupling mechanism, facilitating the emergence of belief states. However, it reveals a "structural hallucination" issue in MTP (illegal shortcuts in latent space) and proposes the LSE-MTP framework. By employing latent consistency and semantic anchoring losses, it anchors predictions to true latent state trajectories, significantly improving path legality and robustness in synthetic graphs and real-world Manhattan taxi navigation.

## Background & Motivation

**Background**: World models—the ability to simulate state evolution within an environment—are a hallmark of intelligent behavior. A core debate is whether LLMs can develop consistent internal world models through next-token prediction (NTP). The optimization objective of NTP is local, focusing only on the conditional probability of the next token, which makes models proficient at capturing surface patterns but struggling to internalize deep global structures persistently.

**Limitations of Prior Work**: Multi-Token Prediction (MTP) provides more structured training signals by predicting multiple future tokens simultaneously, promoting better representation learning. However, practical observations indicate that even when MTP-trained models are accurate at the token level, their latent state evolution may violate intrinsic environmental constraints—intermediate steps are implicitly skipped, creating shortcuts that are invalid under real dynamics.

**Key Challenge**: The contractivity of MTP is "result-driven"—it constrains the alignment of future result representations but ignores the physical legality of intermediate states. Optimizing long-range predictions without explicitly constraining trajectory legality encourages the model to "prioritize results over processes." This is the root cause of "structural hallucinations."

**Goal**: (1) Theoretically characterize the gradient coupling mechanism of MTP and its contractivity; (2) Reveal the structural hallucination problem; (3) Propose a solution to bridge the gap between discrete token supervision and continuous state representations.

**Key Insight**: Use the Neural Tangent Kernel (NTK) framework to analyze gradient flow dynamics in the linearized regime, deriving contractivity theorems and cross-path gradient coupling effects.

**Core Idea**: Add latent consistency loss (aligning predicted representations with true future latent states) and semantic anchoring loss (aligning with target token embeddings) on top of MTP. This transforms the contractive force of MTP from "blind result alignment" into "trajectory-aware alignment."

## Method

### Overall Architecture
LSE-MTP adds two auxiliary losses to the standard MTP architecture. Given a backbone latent state $h_n$, $K$ step-specific transition layers $\mathcal{T}_\phi^{(k-1)}$ generate multi-step predictive representations $\hat{h}_{n,k}$. The training objective includes: (1) multi-step cross-entropy loss (standard MTP); (2) latent consistency loss (aligning predicted representations with future backbone states); (3) semantic anchoring loss (aligning predicted representations with target token embeddings). At inference time, all transition layers and auxiliary losses are discarded, and decoding maintains standard autoregressive NTP.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["Input token sequence"] --> B["Backbone network<br/>Backbone latent state h_n"]
    B --> C["K step-specific transition layers T_φ<br/>Multi-step predictive representations ĥ_n,k (k=2..K)"]
    C --> D["Multi-step cross-entropy loss<br/>(Standard MTP signal)"]
    C --> E["Latent consistency loss"]
    C --> F["Semantic anchoring loss"]
    D -->|Align| G["Target token u_n+k"]
    E -->|Align| H["True backbone latent state h_n+k−1"]
    F -->|Align (stop-grad)| I["Target token embedding E(u_n+k)"]
    B --> J["Inference: Discard transition layers and auxiliary losses<br/>Standard autoregressive NTP decoding"]
```

### Key Designs

**1. Gradient Coupling Analysis: Explaining how MTP "unintentionally" forces belief states via NTK**

Most existing MTP research remains at the level of experimental observation ("it works") but lacks a theoretical explanation for its representation learning effects. This paper analyzes the gradient flow of MTP under the linearized regime of the Neural Tangent Kernel (NTK), characterizing a "prediction coupling" effect: for two latent states $h_1 \sim_k h_2$ that are equivalent $k$ steps into the future (sharing the same $k$-th step target but different next-step targets), training one trajectory incidentally increases the prediction confidence of the corresponding token in the other trajectory (Theorem 2). Under the condition that the transition Jacobian is full rank, this coupling further induces a stable contractive force $\dot{\mathcal{D}} \leq 0$, causing latent states sharing a common future to converge locally to a unified belief state (Lemma 1). This analysis not only fills the theoretical gap in MTP representation learning but also theoretically predicts its side effects—why structural hallucinations occur.

**2. Latent Consistency Loss: Pinning predictive representations to true latent state trajectories**

The transition layers of standard MTP only need to predict the final token correctly, regardless of the path taken in latent space. Consequently, the model can learn illegal shortcuts that skip intermediate steps—this is the structural hallucination. The latent consistency loss directly blocks this shortcut by forcing each $k$-step predictive representation $\hat{h}_{n,k}$ to align with the true backbone latent state at step $n+k-1$:

$$\mathcal{L}_{latent} = \sum_{k=2}^{K} \mathbb{E}_n \big\|\hat{h}_{n,k} - h_{n+k-1}\big\|_2^2.$$

This prevents transition layers from mapping arbitrarily, requiring them to simulate true state transitions instead. This aligns the evolutionary path of predictive representations with actual encoding, shifting MTP's contractivity from "blind result alignment" back to "trajectory-aware alignment."

**3. Semantic Anchoring Loss: Providing a semantic reference frame for predictive representations**

Aligning latent trajectories alone is insufficient; predictive representations might still drift into semantically meaningless regions. The semantic anchoring loss pulls the predictive representation back into the semantic space of the target token by aligning it with the token's embedding vector:

$$\mathcal{L}_{semantic} = \sum_{k=2}^{K} \mathbb{E}_n \big\|\hat{h}_{n,k} - \text{sg}(\mathbf{E}(u_{n+k}))\big\|_2^2,$$

where $\text{sg}(\cdot)$ denotes stop-gradient and $\mathbf{E}(\cdot)$ is the model's embedding layer. Since the embedding layer itself encodes token semantics, anchoring the predictive representation to this space provides a stable semantic coordinate, preventing it from deviating into meaningless internal encodings and enhancing interpretability.

### Loss & Training
The total loss is $\mathcal{L}_{total} = \mathcal{L}_{ce} + \lambda_l \mathcal{L}_{latent} + \lambda_s \mathcal{L}_{semantic}$, with default weights $\lambda_l = \lambda_s = 0.1$. A 6-layer Transformer (6 heads, 120 latent dimensions) is used on 100-node graphs, trained for 20,000 iterations. Auxiliary components are discarded during inference to maintain standard autoregressive decoding.

## Key Experimental Results

### Main Results
Representation alignment on ER graphs and USG (Urban Street Graphs) (Structure Gain = Sim(F) - random baseline):

| Model | ER k=2 Gain | ER k=3 Gain | USG k=2 Gain | USG k=3 Gain |
|------|-------------|-------------|--------------|--------------|
| 1TP (NTP) | 0.027 | 0.022 | -0.005 | 0.018 |
| 2TP (MTP) | 0.210 | 0.074 | 0.214 | 0.066 |
| 4TP (MTP) | 0.176 | 0.162 | 0.178 | 0.180 |
| 4TP (LSE-MTP) | - | - | - | - |

Belief compression (cosine similarity of latent states with the same target and position):

| Model | K | ER G=,P= | USG G=,P= |
|------|---|----------|-----------|
| NTP | 1 | 0.29 | 0.22 |
| MTP | 4 | 0.44 | 0.32 |
| LSE-MTP | 4 | **0.46** | **0.38** |

### Ablation Study

| Configuration | ER ISP ↓ | ER Legal Prob ↑ | USG ISP ↓ | USG Legal Prob ↑ |
|------|----------|-----------------|-----------|------------------|
| NTP (1TP) | 2.7e-5 | 0.995 | 2.2e-5 | 0.998 |
| MTP (3TP) | 7.8e-5 | 0.992 | 7.3e-5 | 0.994 |
| LSE-MTP (3TP) | **2.1e-5** | **0.996** | **1.8e-5** | **0.998** |

### Key Findings
- MTP indeed induces representation alignment (Structure Gain increases by 8-21×), supporting the theoretically predicted contractivity.
- However, MTP simultaneously increases the probability of illegal shortcut paths (ISP increases from 2.7e-5 to 7.8e-5), validating the existence of structural hallucinations.
- LSE-MTP reduces ISP to levels lower than NTP while maintaining the advantages of belief compression, successfully resolving structural hallucinations.
- On real-world Manhattan taxi navigation data, LSE-MTP significantly improves path legality and robustness to perturbations.

## Highlights & Insights
- The theoretical analysis of MTP is profound—using the NTK framework to rigorously derive cross-path gradient coupling and contractivity is more convincing than simple experimental observation. The intuition behind "prediction coupling" in Theorem 2 (training one path indirectly aids another sharing a future) is particularly brilliant.
- The concept of "structural hallucinations" is highly valuable—it reveals a fundamental limitation of MTP: token-level accuracy does not equal trajectory-level consistency. This serves as an important warning for all LLM training utilizing MTP.
- The design of the linear model experiment is clever—constructing a minimal case with 5D orthogonal basis vectors transparently demonstrates how gradient coupling leads to the reinforcement of unobserved transitions.

## Limitations & Future Work
- The theoretical analysis is based on the linearized regime (lazy training), and actual deep networks may deviate from this assumption.
- Experiments were only validated on graph navigation tasks and not tested on natural language tasks.
- LSE-MTP requires true latent states as training signals; alternative solutions may be needed in scenarios where true states are inaccessible.
- Future work could explore methods for applying LSE-MTP in large-scale LLM pre-training.

## Related Work & Insights
- **vs Standard MTP (Gloeckle et al., 2024)**: Standard MTP only performs multi-step token prediction without latent space constraints, leading to structural hallucinations; LSE-MTP adds trajectory-level constraints.
- **vs DreamerV3**: DreamerV3 learns world models in RL using latent state prediction; LSE-MTP introduces similar ideas into the MTP framework for language models.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Theoretical analysis of MTP and the concept of structural hallucinations are original contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐ Sufficient theory validation with synthetic and real data, though missing NLP task experiments.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous theoretical derivation, ingenious experimental design, and clear narrative.
- Value: ⭐⭐⭐⭐⭐ Provides a theoretical foundation and practical methodology for understanding and improving MTP.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Pre-Training Curriculum for Multi-Token Prediction in Language Models](../../ACL2025/llm_pretraining/pre-training_curriculum_for_multi-token_prediction_in_language_models.md)
- [\[NeurIPS 2025\] Next Semantic Scale Prediction via Hierarchical Diffusion Language Models](../../NeurIPS2025/llm_pretraining/next_semantic_scale_prediction_via_hierarchical_diffusion_language_models.md)
- [\[CVPR 2025\] Improving Autoregressive Visual Generation with Cluster-Oriented Token Prediction](../../CVPR2025/llm_pretraining/improving_autoregressive_visual_generation_with_cluster-oriented_token_predictio.md)
- [\[ACL 2026\] Fine-tuning vs. In-context Learning in Large Language Models: A Formal Language Learning Perspective](fine-tuning_vs_in-context_learning_in_large_language_models_a_formal_language_le.md)
- [\[ICLR 2026\] CHAMMI-75: Pre-training multi-channel models with heterogeneous microscopy images](../../ICLR2026/llm_pretraining/chammi-75_pre-training_multi-channel_models_with_heterogeneous_microscopy_images.md)

</div>

<!-- RELATED:END -->
