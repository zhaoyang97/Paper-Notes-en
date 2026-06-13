---
title: >-
  [Paper Note] SkelHCC: A Hyperbolic CLIP-Driven Cache Adaptation Framework for Skeleton-based One-Shot Action Recognition
description: >-
  [ICML 2026][Video Understanding][Skeleton Action Recognition] SkelHCC migrates CLIP to Hyperbolic space, explicitly aligning skeleton-language representations at three granularities: "joint → body part → whole body." It…
tags:
  - "ICML 2026"
  - "Video Understanding"
  - "Skeleton Action Recognition"
  - "One-Shot Learning"
  - "Hyperbolic CLIP"
  - "LLM Guidance"
  - "Multi-granularity Cache"
date: 2026-05-08
content_hash: e213445fe9fe3cec
---

# SkelHCC: A Hyperbolic CLIP-Driven Cache Adaptation Framework for Skeleton-based One-Shot Action Recognition

**Conference**: ICML 2026  
**arXiv**: [2606.03610](https://arxiv.org/abs/2606.03610)  
**Code**: To be confirmed  
**Area**: Video Understanding / Skeleton-based Action Recognition / One-Shot Learning  
**Keywords**: Skeleton Action Recognition, One-Shot Learning, Hyperbolic CLIP, LLM Guidance, Multi-granularity Cache

## TL;DR
SkelHCC migrates CLIP to Hyperbolic space, explicitly aligning skeleton-language representations at three granularities: "joint → body part → whole body." It employs training-free multi-granularity voting cache inference using LLM-generated body part importance masks, achieving a 9% improvement over Prev. SOTA on NTU120 one-shot action recognition with only 0.5M trainable parameters.

## Background & Motivation

**Background**: Skeleton-based action recognition understands actions from human joint sequences. One-shot skeleton action recognition (OSAR) is a high-value but extremely challenging setting—each new class has only one sample, and traditional supervised learning struggle to generalize.

**Limitations of Prior Work**:
- **Difficult Representation Alignment**: Human skeletons are naturally tree-structured (joint → body part → whole body), but existing methods mostly perform modeling in Euclidean space, failing to encode such hierarchical dependencies; skeleton representations are insufficiently aligned with high-level semantic action descriptions.
- **Improper Adaptation Strategies**: Updating backbones in one-shot scenarios often leads to overfitting or requires complex fine-tuning pipelines; there is a lack of inference-time "context-aware" mechanisms to inform the model which body parts are most critical.

**Key Challenge**: One-shot recognition requires both robust cross-modal representations and fast, low-parameter inference-time adaptation; conventional fine-tuning is infeasible under data scarcity.

**Goal**: Propose a unified framework to simultaneously solve (1) cross-modal representations that explicitly encode skeleton hierarchy and (2) training-free, context-aware inference-time adaptation.

**Key Insight**: The negative curvature of Hyperbolic geometry is naturally suited for tree structures (joint graph $\delta$-hyperbolicity has been measured in Appendix I); LLM knowledge can specify "which joints are important for which action," which can be integrated directly into similarity calculations as masks.

**Core Idea**: Utilize Hyperbolic CLIP to learn three-granularity aligned representations (EH-HCLIP), followed by LLM-guided Multi-granularity Voting Cache (LMV-Cache) for training-free adaptation during inference.

## Method

### Overall Architecture
Two stages:

- **Training Phase**: EH-HCLIP is trained on base classes—the CLIP text encoder and skeleton backbone are frozen, and only a lightweight MLP adapter (0.5M parameters) is trained to project skeleton/text features onto the Lorentz Hyperbolic manifold for hierarchical alignment.
- **Inference Phase**: For a query sample of a new class, the framework calculates (1) the skeleton-skeleton cache similarity (cache logit) between query and support samples and (2) the skeleton-text similarity (HCLIP logit) with text prompts, performing classification via residual fusion.

### Key Designs

1. **Explicitly Hierarchical Hyperbolic CLIP (EH-HCLIP)**:

    - **Function**: Learns skeleton-text alignment in Hyperbolic space, explicitly encoding the three-level human structure.
    - **Mechanism**: The skeleton is partitioned into body joints (BJ), body parts (BP), and full body (FB) based on biological anatomical priors, with text descriptions generated for each granularity using LLM prompts. Euclidean features are projected onto the Lorentz manifold via exponential mapping $\tilde{S} = \exp_M^{O}(S)$. Lorentzian distance $d_{\mathbb{L}, c}(\cdot)$ is used for contrastive probabilities. The EHHC loss is a weighted sum: $\mathcal{L}_{EHHC} = \sum_i \frac{\alpha_i}{2} (\mathcal{L}_{HCL}(\tilde{S}^{(i)}, \tilde{T}^{(i)}) + \mathcal{L}_{HCL}(\tilde{T}^{(i)}, \tilde{S}^{(i)}))$; additionally, Hyperbolic Entailment Loss (HEL) uses entailment cone constraints to enforce partial order relations.
    - **Design Motivation**: Hyperbolic space volume grows exponentially with the radius, enabling tree representation in low dimensions; it aligns with the $\delta$-hyperbolicity of skeleton graphs; multi-granularity contrast allows the model to focus on both local key joints and global context.

2. **LLM-guided Multi-granularity Voting Cache (LMV-Cache)**:

    - **Function**: Performs training-free inference-time adaptation—the three-granularity skeleton features of support samples are stored as keys and labels as values in a cache; query samples are classified via weighted voting.
    - **Mechanism**: For each action category, binary masks $\mathcal{M}^{BJ}, \mathcal{M}^{BP}$ indicating "which joints/body parts are critical for that action" are generated offline via GPT-4. During inference, joint-level and body-part-level similarities are computed using Hadamard products: $\text{Sim}^{BJ} = \alpha_2 \frac{1}{V} \sum_i [\phi(S_q^{BJ}, S_s^{BJ}) \odot \mathcal{M}^{BJ}]_i$, followed by multi-granularity matrix voting to merge into the cache logit.
    - **Design Motivation**: LLMs provide action-level semantic priors ("jumping uses legs, clapping uses hands"); applying the prior directly to similarity computation avoids prior vanishing during inference (vs. CrossGLG); the voting mechanism softens hard classification into multi-granularity consensus, improving robustness.

3. **Dual Logit Inference via Residual Fusion**:

    - **Function**: Combines the advantages of skeleton-skeleton cache retrieval and skeleton-text retrieval.
    - **Mechanism**: $\text{logit}_{SkelHCC} = \text{logit}_{Cache} + \gamma \cdot \text{logit}_{HCLIP}$, where $\gamma$ balances the two signals. EH-HCLIP itself can be viewed as a special "text cache."
    - **Design Motivation**: Skeleton-skeleton similarity is sensitive to appearance variations (different forms of the same action), while skeleton-text similarity emphasizes semantics; the two are complementary.

### Loss & Training
- **Loss**: $\mathcal{L}_{EH\text{-}HCLIP} = \mathcal{L}_{EHHC} + \lambda \mathcal{L}_{EHHE} + \mathcal{L}_{CE}$, with $\lambda = 0.1$.
- **Key Hyperparameters**: Hyperbolic curvature $c = 0.1$, similarity temperature $\beta = 1.0$, granularity weights $\alpha_1 = \alpha_2 = \alpha_3 = 0.5$ (adaptive later).

## Key Experimental Results

### Main Results

One-shot action recognition on NTU RGB+D 120 / 60 and PKU-MMD II (accuracy under different numbers of base classes):

| Dataset | Method | 20 Base | 60 Base | 100 Base | Backbone Update | Adaptation Params |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| NTU120 | CrossGLG (SOTA) | 45.3 | 62.1 | 62.6 | ✓ | 1.7M |
| NTU120 | **Ours** | **52.0** | **67.4** | **71.6** | ✗ | **0.5M** |
| NTU60 | CrossGLG | — | 75.6 | — | ✓ | — |
| NTU60 | **Ours** | — | **84.1** | — | ✗ | **0.5M** |
| P-MMD | SkeletonX | — | 38.3 | — | — | — |
| P-MMD | **Ours** | — | **40.0** | — | — | **0.5M** |

**Key Observation**: NTU120 reaches 71.6% under the 100 base class setting, which is 9.0% higher than CrossGLG, despite having only 1/3 the parameters and a frozen backbone.

### Ablation Study

Module effectiveness (NTU120, 100 base classes):

| Method | Accuracy | Gain |
| :--- | :--- | :--- |
| CLIP (Euclidean) + Cache | 62.9 | — |
| HCLIP + Cache | 64.8 | +1.9 |
| EH-HCLIP + Cache | 67.6 | +4.7 |
| CLIP + LMV-Cache | 66.2 | +3.3 |
| HCLIP + LMV-Cache | 68.2 | +5.3 |
| **EH-HCLIP + LMV-Cache (Full)** | **71.6** | **+8.7** |

Comparison of mask types (NTU120, 100 base classes):

| Mask | Accuracy | Change |
| :--- | :--- | :--- |
| None | 68.5 | — |
| Random | 66.3 | -2.2 |
| Learnable | 68.6 | +0.1 |
| Self-Attention | 67.1 | -1.4 |
| LLM Mask (BP) | 69.9 | +1.4 |
| **LLM Mask (BP + BJ)** | **71.6** | **+3.1** |

### Key Findings
- Hyperbolic CLIP improves 1.9% over Euclidean CLIP; adding "Explicit Hierarchy" yields an additional 2.8%—structure priors are essential.
- Removing BJ + BP multi-granularity (keeping only FB) leads to a 3-4% drop, highlighting the importance of multi-granularity for one-shot robustness.
- Random or attention-based masks result in performance drops; LLM-generated semantic masks are the only strategy providing stable improvements, indicating LLM knowledge is more reliable than model-learned "importance."

## Highlights & Insights
- **Natural Fit Between Hyperbolic Space and Skeletons**: The paper provides hard evidence for using Hyperbolic geometry by measuring the $\delta$-hyperbolicity of skeleton graphs, rather than simply applying Hyperbolic CLIP.
- **Inference-time LLM Knowledge**: Unlike CrossGLG, which distills LLM knowledge during training, LMV-Cache encodes it directly into masks used during inference, preventing knowledge "vanishing."
- **Parameter-efficient One-shot Adaptation**: Freezing the backbone and using a 0.5M MLP adapter is a pragmatic response to one-shot data scarcity, utilizing 3.4× fewer parameters than CrossGLG.
- **Multi-granularity Soft Voting**: Converting one-shot hard classification into soft voting significantly enhances robustness, a concept transferrable to other structured data tasks.

## Limitations & Future Work
- Restricted to one-shot settings; fusion of multiple support samples for few-shot scenarios (weighted average? prototypes?) is not detailed.
- Evaluations are limited to the skeleton modality; extensions to RGB, depth, or multi-view were only mentioned in the conclusion.
- LLM masks require calling GPT-4 for each new action class; while this is a one-time cost, scalability concerns remain for massive action libraries.
- Hyperbolic curvature was only tested at $c = 0.1$; optimal values for different datasets have not been systematically scanned.

## Related Work & Insights
- **vs APSR / uDTW / SL-DML**: Traditional metric learning methods reside in Euclidean space and fail to encode skeleton tree structures; this work addresses this at the representation space level.
- **vs GAP / CrossGLG**: These also use LLM priors, but GAP is fully supervised and CrossGLG's LLM knowledge is not active during inference; this work uses masks to keep LLM knowledge in the decision-making loop.
- **vs HyperbolicCLIP / Hyperbolic Segmentation**: This work is not a simple reuse of Hyperbolic CLIP but is specifically designed for skeleton three-granularity hierarchy with added entailment loss.
- **Insight**: The triad of Geometric Priors (Tree → Hyperbolic) + High-level Semantic Priors (LLM) + Parameter-efficient Adaptation (MLP) can be generalized to tasks like keypoint detection and tree-structured scene graphs.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The combination of Hyperbolic space, explicit hierarchy, and LLM mask voting is a first in OSAR.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive results across three authoritative datasets and extensive ablations on mask types.
- **Writing Quality**: ⭐⭐⭐⭐ Logic is clear and methodology is detailed; the introduction to Hyperbolic fundamentals is slightly long.
- **Value**: ⭐⭐⭐⭐⭐ High relevance for data-scarce scenarios like rehabilitation and medical AI; parameter efficiency facilitates deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] SkeletonContext: Skeleton-side Context Prompt Learning for Zero-Shot Skeleton-based Action Recognition](../../CVPR2026/video_understanding/skeletoncontext_skeleton-side_context_prompt_learning_for_zero-shot_skeleton-bas.md)
- [\[ICCV 2025\] Frequency-Semantic Enhanced Variational Autoencoder for Zero-Shot Skeleton-based Action Recognition](../../ICCV2025/video_understanding/frequency-semantic_enhanced_variational_autoencoder_for_zero-shot_skeleton-based.md)
- [\[CVPR 2026\] SpikeTrack: A Spike-driven Framework for Efficient Visual Tracking](../../CVPR2026/video_understanding/spiketrack_a_spike-driven_framework_for_efficient_visual_tracking.md)
- [\[AAAI 2026\] SUGAR: Learning Skeleton Representation with Visual-Motion Knowledge for Action Recognition](../../AAAI2026/video_understanding/sugar_learning_skeleton_representation_with_visual-motion_knowledge_for_action_r.md)
- [\[AAAI 2026\] Task-Specific Distance Correlation Matching for Few-Shot Action Recognition](../../AAAI2026/video_understanding/task-specific_distance_correlation_matching_for_few-shot_action_recognition.md)

</div>

<!-- RELATED:END -->
