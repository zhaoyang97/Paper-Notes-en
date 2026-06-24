---
title: >-
  [Paper Note] Bad Students Make Great Teachers: Active Learning Accelerates Large-Scale Visual Understanding
description: >-
  [ECCV 2024][Multimodal Efficiency][Active Learning] This work proposes ClassAct/ActiveCLIP, which utilizes small, low-cost proxy models to compute "learnability" scores for data points to prioritize training data. This reduces training updates for large-scale visual classifiers and multimodal models by 46% and 51% respectively, achieves up to 25% total compute savings, and stands as the first active learning method to achieve net positive compute savings in large-scale pre-tr…
tags:
  - "ECCV 2024"
  - "Multimodal Efficiency"
  - "Active Learning"
  - "Data Selection"
  - "Scaling Laws"
  - "CLIP"
  - "Learnability Score"
date: 2026-05-08
content_hash: a20ff2ea0602d840
---

# Bad Students Make Great Teachers: Active Learning Accelerates Large-Scale Visual Understanding

**Conference**: ECCV 2024  
**arXiv**: [2312.05328](https://arxiv.org/abs/2312.05328)  
**Code**: None  
**Area**: Multimodal VLM  
**Keywords**: Active Learning, Data Selection, Scaling Laws, CLIP, Learnability Score

## TL;DR

This work proposes ClassAct/ActiveCLIP, which utilizes small, low-cost proxy models to compute "learnability" scores for data points to prioritize training data. This reduces training updates for large-scale visual classifiers and multimodal models by 46% and 51% respectively, achieves up to 25% total compute savings, and stands as the first active learning method to achieve net positive compute savings in large-scale pre-training.

## Background & Motivation

The training of large-scale vision and language models follows power-law scaling — incremental improvements in model performance require orders-of-magnitude increases in computation. A key characteristic of this regime is that training data is uniformly sampled. Active data selection improves data efficiency by prioritizing training on the most relevant samples, but it has not been widely adopted because existing methods fail to satisfy three key conditions simultaneously:

- **Generality across models and tasks**: No single algorithm is effective across various models and tasks.
- **Scalability to large datasets**: Many methods fail to scale even on medium-sized datasets like ImageNet.
- **Net compute savings**: Whether the total FLOPs are truly saved when accounting for the overhead of data selection.

The **Key Challenge** lies in the fact that while model-based data selection methods can significantly improve learning efficiency, the computation overhead of selection itself is often comparable to or even exceeds the savings in subsequent training. The **Key Insight** of this paper is: **using extremely small proxy models to approximate the data selection scores of large models**, thereby reducing the selection overhead to an almost negligible level.

## Method

### Overall Architecture

The overall framework is based on Online Batch Selection: it first uniformly samples a large super-batch from the training set, computes a score for each sample using a scoring model, and then performs non-uniform sampling based on these scores to obtain a smaller sub-batch for training. The framework consists of three model components:
- **Reference Model**: A pre-trained small model that provides baseline losses.
- **Online Model**: Shared architecture with the reference model, trained in parallel with the learner.
- **Learner Model**: The actual large model targeted for training.

### Key Designs

1. **Design of Data Scoring Metric**:

    - **Function**: Defines how to compute priority scores for each data point.
    - **Mechanism**: Compares three scoring strategies:
        - **Hard instances first** (hard): $s^{hard}(x|\theta) = \ell(x|\theta)$, prioritizing samples with high loss.
        - **Easy instances first** (easy): $s^{easy}(x|\theta) = -\ell(x|\theta)$, prioritizing samples with low loss (denoising).
        - **Learnability**: $s^{learn}(x|\theta^t, \theta^*) = \ell(x|\theta^t) - \ell(x|\theta^*)$, combining the advantages of both.
    - **Design Motivation**: Hard-first introduces noise and unlearnable samples; easy-first excludes noise but misses valuable difficult samples; learnability selection targets samples that "the reference model can do well but the current learner cannot," which are truly improvable through training.

2. **Cross-Scale Proxy Scoring (Core Innovation of ClassAct/ActiveCLIP)**:

    - **Function**: Replaces the large learner model with extremely small models for data scoring.
    - **Mechanism**: Introduces a third "online model" that has the same architecture and size as the reference model, replacing the learner's role in the learnability formula. The scoring cost is reduced from $F_{act} = F_{ref} + F_{learn}$ to $F_{act} = 2F_{ref}$.
    - **Design Motivation**: The original RHO-loss requires inference via the learner model for scoring, which does not reduce the selection overhead. Experiments show that learnability scoring is highly robust to model downscaling — even if the scoring model is 1000x smaller than the learner (ViT-Mu vs. ViT-L), it still provides a 16% speedup.
    - **Key Findings**: Easy-first scoring is extremely sensitive to model downscaling, whereas learnability scoring degrades gracefully.

3. **Multimodal Adaptation (ActiveCLIP/ActiveSigLIP)**:

    - **Function**: Extends the framework to CLIP/SigLIP multimodal pre-training.
    - **Mechanism**: The learner is trained with contrastive loss, while scoring uses a simplified image-text dot-product similarity as the actor loss.
    - **Design Motivation**: Contrastive loss is computationally expensive (requiring softmax over the entire batch); using simple dot-product similarity drastically reduces scoring overhead.

4. **Online Reference Model Training**:

    - **Function**: Eliminates the two-step process of pre-training the reference model.
    - **Mechanism**: The reference model is trained in parallel on the super-batch (10x larger batch) with the learning rate set to 2x.
    - **Design Motivation**: Smaller models can be computed over larger batches, quickly converging to a good scoring policy.

### Loss & Training

- Classification task (ClassAct): Both the learner and the scorers use standard cross-entropy loss.
- Multimodal task (ActiveCLIP): The learner uses contrastive loss, while scoring uses the dot product of image-text embeddings.
- The filtering target ratio is fixed at 50% ($\rho=B/b=2$) across all experiments, selecting half from each super-batch.
- Non-uniform sampling is performed according to Softmax probabilities: $\pi(x_i) = \text{Softmax}(\{s_i\})$.

## Key Experimental Results

### Main Results: Large-Scale Classification (JFT-300M)

| Configuration | Learner Speedup | Total Compute Speedup | Notes |
|------|-----------|-----------|------|
| ViT-B scoring → ViT-L (learnability) | 31% | Net negative compute | Scoring model is too large |
| ViT-S scoring → ViT-L (learnability) | 28% | ~20% net positive | Net positive compute regime |
| ViT-Ti scoring → ViT-L (learnability) | 26% | ~25% net positive | Best overall efficiency |
| ViT-Mu scoring → ViT-L (learnability) | 16% | Net positive | 1000x smaller but still effective |
| ViT-S scoring → ViT-L (easy-first) | <10% | Net negative | Sensitive to downscaling |

### Main Results: Multimodal Pre-training

| Method | Training Volume | IN-1K ZS Top-1 | COCO im2txt | COCO txt2im |
|------|-------|----------------|-------------|-------------|
| CLIP | 13B | 68.3 | 52.4 | 33.1 |
| OpenCLIP | 34B | 70.2 | 59.4 | 42.3 |
| ActiveCLIP | 3B | 71.3 | 57.7 | 43.0 |
| ActiveCLIP | 8B | 72.2 | 60.7 | 44.9 |
| SigLIP | 3B | 72.1 | 60.7 | 42.7 |
| ActiveSigLIP | 3B | 72.0 | 63.5 | 45.3 |

### Ablation Study

| Configuration | Learner Speedup | Compute Speedup | Notes |
|------|-----------|---------|------|
| RHO (Tiny ref + B online + B learner) | 0% | -79% | Original RHO fails |
| ClassAct-HO (Tiny+Tiny+B, held-out) | 18% | 3% | Held-out reference |
| ClassAct (Tiny+Tiny+B, in-domain) | 18% | 3% | In-domain reference equivalent |
| ClassAct-Online (Tiny+Tiny+B, online) | 17% | 2% | No pre-trained reference needed |

### Key Findings

- Learnability scoring is extremely robust to the scaling of the scoring model, remaining effective even with a 1000x downscaling, whereas easy-first scoring is highly sensitive.
- Data selection strategies generalize across tasks: reference models trained on the high-quality LTIP dataset perform better when guiding large-scale ALIGN training.
- Scaling laws generalize to active learning settings: consistent speedups are achieved across different compute budgets.
- Training reference models online (without pre-training steps) achieves comparable performance to pre-trained reference models.

## Highlights & Insights

- **First large-scale active learning method with net positive compute savings**: Resolves the long-standing community dilemma of "data filtering overhead $\ge$ training savings."
- **Counter-intuitive finding that "Bad Students Make Great Teachers"**: Extremely small models (1000x smaller than the learner), despite having poor performance on their own, can effectively guide data selection for large models.
- **Discovery of the Pareto Frontier**: Reveals the optimal trade-off surface between scoring computation overhead and training iteration savings.
- **Complementary to data filtering methods**: ActiveCLIP is complementary to both static data cleaning (DataComp) and new training objectives (SigLIP), achieving new SOTA performance when combined.

## Limitations & Future Work

- Experiments are restricted to a 50% filtering ratio; more aggressive filtering might bring greater gains but also higher risks.
- Only two tasks (image classification and multimodal contrastive learning) were validated; the method was not extended to language models, video, or generative modeling.
- Sensitivity analysis of the temperature parameter (Softmax) in the scoring strategy is not thoroughly examined.
- The quality of the reference model's training data significantly affects downstream performance (LTIP >> ALIGN); how to automatically select the optimal reference data remains an open question.

## Related Work & Insights

- **RHO-loss** (Mindermann et al., 2022): Introduced the concept of learnability but failed to achieve net positive compute savings. This work resolves this via cross-scale proxies.
- **DoReMi** (Xie et al., 2023): Uses small proxy models to determine data mixture proportions (for language models) with a similar concept but at a different granularity.
- **DataComp** (Gadre et al., 2023): A static data filtering method, which is complementary to the dynamic selection in this paper.
- **Insight**: **The "knowledge" of small models might be more valuable than we think** — they do not need to make highly accurate predictions, but only need to correctly rank the data quality.

## Rating

- Novelty: ⭐⭐⭐⭐ First to achieve net compute-positive active learning in large-scale pre-training; elegant cross-scale proxy concept.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Validated across multiple large-scale datasets (JFT-300M, ALIGN, WebLI), scaling laws analysis, and rich ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear logic, rigorous computational analysis, and highly intuitive Pareto frontier visualizations.
- Value: ⭐⭐⭐⭐⭐ Directly practical for large-scale training, significantly reducing training bills.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] EvoComp: Learning Visual Token Compression for Multimodal Large Language Models via Semantic-Guided Evolutionary Labeling](../../CVPR2026/vlm_efficiency/evocomp_learning_visual_token_compression_for_multimodal_large_language_models_v.md)
- [\[ACL 2025\] Sharper and Faster mean Better: Towards More Efficient Vision-Language Model for Hour-scale Long Video Understanding](../../ACL2025/vlm_efficiency/sophia_efficient_long_video.md)
- [\[ECCV 2024\] Groma: Localized Visual Tokenization for Grounding Multimodal Large Language Models](groma_localized_visual_tokenization_for_grounding_multimodal_large_language_mode.md)
- [\[ECCV 2024\] IVTP: Instruction-Guided Visual Token Pruning for Large Vision-Language Models](ivtp_instruction-guided_visual_token_pruning_for_large_vision-language_models.md)
- [\[CVPR 2026\] LazyVAR: Accelerating Visual Autoregressive Models via Scale-wise Token Pruning and Parallel Group Decoding](../../CVPR2026/vlm_efficiency/lazyvar_accelerating_visual_autoregressive_models_via_scale-wise_token_pruning_a.md)

</div>

<!-- RELATED:END -->
