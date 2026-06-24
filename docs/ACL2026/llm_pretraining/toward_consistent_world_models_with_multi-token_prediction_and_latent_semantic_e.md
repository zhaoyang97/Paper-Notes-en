---
title: >-
  [Paper Note] Toward Consistent World Models with Multi-Token Prediction and Latent Semantic Enhancement
description: >-
  [ACL 2026][LLM Pretraining][Multi-token prediction] This paper theoretically analyzes how Multi-Token Prediction (MTP) induces representation contractivity through a gradient coupling mechanism, facilitating the emergence of belief states. However, it also reveals the "structural hallucination" problem of MTP (illegal shortcuts in latent space). The proposed LSE-MTP framework anchors predictions to true latent state trajectories through latent consistency and semantic anchori…
tags:
  - "ACL 2026"
  - "LLM Pretraining"
  - "Multi-token prediction"
  - "World models"
  - "Belief states"
  - "Structural hallucinations"
  - "Latent semantic enhancement"
date: 2026-05-08
content_hash: 6c25d1d9efaa83cd
---

# Toward Consistent World Models with Multi-Token Prediction and Latent Semantic Enhancement

**Conference**: ACL 2026  
**arXiv**: [2604.06155](https://arxiv.org/abs/2604.06155)  
**Code**: [GitHub](https://github.com/QiminZhong/LSE-MTP)  
**Area**: World Models / LLM Representation Learning  
**Keywords**: Multi-token prediction, World models, Belief states, Structural hallucinations, Latent semantic enhancement

## TL;DR
This paper theoretically analyzes how Multi-Token Prediction (MTP) induces representation contractivity through a gradient coupling mechanism, facilitating the emergence of belief states. However, it also reveals the "structural hallucination" problem of MTP (illegal shortcuts in latent space). The proposed LSE-MTP framework anchors predictions to true latent state trajectories through latent consistency and semantic anchoring losses, significantly improving path legality and robustness in synthetic graphs and real-world Manhattan taxi navigation.

## Background & Motivation

**Background**: World models (the ability to simulate state evolution within an environment) are hallmarks of intelligent behavior. Whether LLMs can develop consistent internal world models through next-token prediction (NTP) is a central debate. The optimization objective of NTP is local—focusing only on the conditional probability of the next token, which makes models adept at capturing surface patterns but difficult to internalize deep global structures persistently.

**Limitations of Prior Work**: Multi-Token Prediction (MTP) provides more structured training signals by simultaneously predicting multiple future tokens, promoting better representation learning. However, practical observations indicate that even models trained with MTP, while accurate at the token level, may evolve latent states that violate essential environmental constraints—intermediate steps are implicitly skipped, creating shortcuts that are invalid under real dynamics.

**Key Challenge**: The contractivity of MTP is "outcome-driven"—it constrains the alignment of future outcome representations but ignores the physical legality of intermediate states. Optimizing long-range predictions without explicitly constraining trajectory legality encourages the model to "prioritize outcomes over processes." This is the root cause of "structural hallucinations."

**Goal**: (1) Theoretically characterize the gradient coupling mechanism and contractivity of MTP; (2) Reveal the structural hallucination problem; (3) Propose a solution to bridge the gap between discrete token supervision and continuous state representation.

**Key Insight**: Analyzing the gradient flow dynamics of MTP under a linearized regime using the Neural Tangent Kernel (NTK) framework to derive contractivity theorems and cross-path gradient coupling effects.

**Core Idea**: Adding a latent consistency loss (aligning predicted representations with actual future latent states) and a semantic anchoring loss (aligning with target token embeddings) on top of MTP. This transforms the contracting force of MTP from "blind outcome alignment" into "trajectory-aware alignment."

## Method

### Overall Architecture
LSE-MTP adds two auxiliary losses to the standard MTP architecture. Given a backbone latent state $h_n$, $K$ step-specific transition layers $\mathcal{T}_\phi^{(k-1)}$ generate multi-step predicted representations $\hat{h}_{n,k}$. The training objective includes: (1) multi-step cross-entropy loss (standard MTP); (2) latent consistency loss (aligning predicted representations with future backbone states); (3) semantic anchoring loss (aligning predicted representations with target token embeddings). At inference time, all transition layers and auxiliary losses are discarded, and decoding maintains standard autoregressive NTP.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["Input token sequence"] --> B["Backbone network<br/>Backbone latent state h_n"]
    B --> C["K step-specific transition layers T_φ<br/>Multi-step predicted representation ĥ_n,k (k=2..K)"]
    C --> D["Multi-step cross-entropy loss<br/>(Standard MTP signal)"]
    C --> E["Latent consistency loss"]
    C --> F["Semantic anchoring loss"]
    D -->|Alignment| G["Target token u_n+k"]
    E -->|Alignment| H["True backbone latent state h_n+k−1"]
    F -->|Alignment (stop-grad)| I["Target token embedding E(u_n+k)"]
    B --> J["Inference: Discard transition layers & auxiliary losses<br/>Standard autoregressive NTP decoding"]
```

### Key Designs

**1. Gradient Coupling Analysis: Explaining how MTP "unintentionally" forces belief states using NTK**

Most existing MTP research stays at the experimental observation of "it works," lacking a theoretical explanation of its representation learning effects. This paper analyzes the gradient flow of MTP under the linearized regime of the Neural Tangent Kernel (NTK), characterizing a "prediction coupling" effect: for two latent states $h_1 \sim_k h_2$ that are equivalent for the $k$-step future (sharing the same $k$-th step target but having different next-step targets), training one trajectory incidentally raises the prediction confidence of the corresponding token for the other trajectory (Theorem 2). Under the condition that the transition Jacobian is full rank, this coupling further induces a stable contracting force $\dot{\mathcal{D}} \leq 0$, causing latent states sharing a common future to locally converge to a unified belief state (Lemma 1). This analysis not only fills the theoretical gap in MTP representation learning but also theoretically predicts its side effects—why structural hallucinations occur.

**2. Latent Consistency Loss: Pinning predicted representations to the true latent state trajectory**

In standard MTP, transition layers only need the final token prediction to be correct, regardless of the path taken in latent space. Consequently, the model can learn illegal shortcuts that skip intermediate steps—this is structural hallucination. The latent consistency loss directly blocks this shortcut by forcing each $k$-step predicted representation $\hat{h}_{n,k}$ to align with the true backbone latent state at step $n+k-1$:

$$\mathcal{L}_{latent} = \sum_{k=2}^{K} \mathbb{E}_n \big\|\hat{h}_{n,k} - h_{n+k-1}\big\|_2^2.$$

This prevents transition layers from mapping arbitrarily, requiring them instead to simulate real state transitions, ensuring the evolution path of predicted representations remains consistent with actual encoding.

**3. Semantic Anchoring Loss: Providing a semantic reference frame for predicted representations**

Aligning the latent state trajectory alone is insufficient, as predicted representations might still drift into semantically meaningless regions. The semantic anchoring loss pulls predicted representations back into the semantic space of the target token—aligning them with the embedding vector of that token:

$$\mathcal{L}_{semantic} = \sum_{k=2}^{K} \mathbb{E}_n \big\|\hat{h}_{n,k} - \text{sg}(\mathbf{E}(u_{n+k}))\big\|_2^2,$$

where $\text{sg}(\cdot)$ is stop-gradient and $\mathbf{E}(\cdot)$ is the model's embedding layer. Since the embedding layer itself encodes token semantics, anchoring the predicted representation to this space provides a stable semantic coordinate, preventing it from deviating into meaningless internal encodings and enhancing interpretability.

### Loss & Training
The total loss is $\mathcal{L}_{total} = \mathcal{L}_{ce} + \lambda_l \mathcal{L}_{latent} + \lambda_s \mathcal{L}_{semantic}$, with default values $\lambda_l = \lambda_s = 0.1$. A 6-layer Transformer (6 heads, latent dimension 120) is used on a 100-node graph, trained for 20,000 iterations. During inference, all auxiliary components are discarded, maintaining standard autoregressive decoding.

## Key Experimental Results

### Main Results
Representation alignment on ER graphs and USG (Urban Street Graphs) (Structure Gain = Sim(F) - random baseline):

| Model | ER k=2 Gain | ER k=3 Gain | USG k=2 Gain | USG k=3 Gain |
|------|-------------|-------------|--------------|--------------|
| 1TP (NTP) | 0.027 | 0.022 | -0.005 | 0.018 |
| 2TP (MTP) | 0.210 | 0.074 | 0.214 | 0.066 |
| 4TP (MTP) | 0.176 | 0.162 | 0.178 | 0.180 |
| 4TP (LSE-MTP) | - | - | - | - |

Belief Compression (Cosine similarity of latent states with same target and position):

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
- LSE-MTP maintains the advantages of belief compression while reducing ISP to levels lower than NTP, successfully resolving structural hallucinations.
- On real Manhattan taxi navigation data, LSE-MTP significantly improves path legality rates and robustness to perturbations.

## Highlights & Insights
- The theoretical analysis of MTP is profound—strictly deriving cross-path gradient coupling and contractivity via the NTK framework is more convincing than simple experimental observation. The "prediction coupling" intuition in Theorem 2 (training one path indirectly helps another path sharing the same future) is particularly brilliant.
- The concept of "structural hallucinations" is highly valuable—it reveals a fundamental limitation of MTP: token-level accuracy does not equate to trajectory-level consistency. This serves as an important warning for all LLM training utilizing MTP.
- The linear model experiments are cleverly designed—using 5-dimensional orthogonal basis vectors to construct a minimal case that transparently demonstrates how gradient coupling can lead to the reinforcement of unobserved transitions.

## Limitations & Future Work
- The theoretical analysis is based on a linearized regime (lazy training); actual deep networks may deviate from this assumption.
- Experiments were only validated on graph navigation tasks and not tested on natural language tasks.
- LSE-MTP requires true latent states as training signals; alternative solutions are needed for scenarios where true states are unobtainable.
- Future work could explore methods for applying LSE-MTP in actual LLM pre-training.

## Related Work & Insights
- **vs Standard MTP (Gloeckle et al., 2024)**: Standard MTP involves multi-step token prediction without latent space constraints, leading to structural hallucinations; LSE-MTP adds trajectory-level constraints.
- **vs DreamerV3**: DreamerV3 learns world models in RL using latent state prediction; LSE-MTP introduces similar ideas into the MTP framework for language models.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The theoretical analysis of MTP and the concept of structural hallucinations are original contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐ Synthetic graphs plus real-world data provide sufficient theoretical validation, though language task experiments are missing.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous theoretical derivation, clever experimental design, and clear narrative.
- Value: ⭐⭐⭐⭐⭐ Provides a theoretical foundation and practical methods for understanding and improving MTP.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Pre-Training Curriculum for Multi-Token Prediction in Language Models](../../ACL2025/llm_pretraining/pre-training_curriculum_for_multi-token_prediction_in_language_models.md)
- [\[ICLR 2026\] Beyond Multi-Token Prediction: Pretraining LLMs with Future Summaries](../../ICLR2026/llm_pretraining/beyond_multi-token_prediction_pretraining_llms_with_future_summaries.md)
- [\[NeurIPS 2025\] Next Semantic Scale Prediction via Hierarchical Diffusion Language Models](../../NeurIPS2025/llm_pretraining/next_semantic_scale_prediction_via_hierarchical_diffusion_language_models.md)
- [\[ICLR 2026\] ssToken: Self-modulated and Semantic-aware Token Selection for LLM Fine-tuning](../../ICLR2026/llm_pretraining/sstoken_self-modulated_and_semantic-aware_token_selection_for_llm_fine-tuning.md)
- [\[CVPR 2025\] Improving Autoregressive Visual Generation with Cluster-Oriented Token Prediction](../../CVPR2025/llm_pretraining/improving_autoregressive_visual_generation_with_cluster-oriented_token_predictio.md)

</div>

<!-- RELATED:END -->
