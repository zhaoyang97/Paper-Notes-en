---
title: >-
  [Paper Note] From Past To Path: Masked History Learning for Next-Item Prediction in Generative Recommendation
description: >-
  [ACL 2026][Image Generation][Generative Recommendation] This paper proposes Masked History Learning (MHL), a training framework that introduces masked history reconstruction as an auxiliary task alongside autoregressive…
tags:
  - "ACL 2026"
  - "Image Generation"
  - "Generative Recommendation"
  - "Masked History Learning"
  - "Entropy-Guided"
  - "Curriculum Learning"
  - "Sequential Recommendation"
date: 2026-05-08
content_hash: 6b988ec935f36afd
---

# From Past To Path: Masked History Learning for Next-Item Prediction in Generative Recommendation

**Conference**: ACL 2026  
**arXiv**: [2509.23649](https://arxiv.org/abs/2509.23649)  
**Code**: [GitHub](https://github.com/CQU-MM-Intelligent-Lab/MHL)  
**Area**: Image Generation  
**Keywords**: Generative Recommendation, Masked History Learning, Entropy-Guided, Curriculum Learning, Sequential Recommendation

## TL;DR

This paper proposes Masked History Learning (MHL), a training framework that introduces masked history reconstruction as an auxiliary task alongside autoregressive training in generative recommendation. By combining entropy-guided adaptive masking strategies and curriculum learning schedulers, the model shifts from merely predicting "what's next" to understanding "why this path formed," significantly outperforming SOTA on three datasets.

## Background & Motivation

**Background**: Generative recommendation is an emerging paradigm that encodes items as semantic ID sequences and leverages pretrained language models (e.g., T5) or LLMs to directly generate item identifiers, offering advantages in flexibility and scalability.

**Limitations of Prior Work**: Existing generative recommendation models (TIGER, HSTU, RPG, etc.) almost exclusively rely on autoregressive next-item prediction training. This "left-to-right" paradigm is inherently biased toward local context and fails to capture deep historical dependencies and complex user intents within behavioral paths. Models excel at local prediction but lack global understanding, making them susceptible to noise and short-term biases (short-term myopia problem).

**Key Challenge**: When recent interaction history is truncated, existing SOTA models experience performance collapse (e.g., after a photography enthusiast sequentially purchases camera, tripod, camera bag, and lens, the model focuses only on "lens" to predict lens accessories, ignoring that "camera body" is the true intent driving subsequent storage card purchases).

**Goal**: Enable generative recommendation models to not only learn "what's next" but also understand "why this purchase path formed."

**Key Insight**: Drawing inspiration from masked language modeling in NLP, introduce history reconstruction as an auxiliary task during autoregressive training.

**Core Idea**: Jointly optimize two objectives—next-item prediction and masked history reconstruction—by using information entropy to guide the selection of the most informative historical positions for masking, and employing curriculum learning to smoothly transition from historical understanding to future prediction.

## Method

### Overall Architecture

MHL adds a masked history reconstruction branch on top of semantic ID-based generative recommendation. Each item is encoded as a K-digit semantic ID, and a Transformer decoder generates contextual hidden states. During training, two losses are jointly optimized: (1) $\mathcal{L}_{next}$ predicts the semantic ID of the next item; (2) $\mathcal{L}_{mask}$ reconstructs masked historical items from contextual states. During inference, only autoregressive generation is used without masking.

### Key Designs

1. **Multi-Granularity Masking Strategy**:

    - Function: Provides diverse learning signals for history reconstruction
    - Mechanism: Designs three masking granularities—item-level (replaces entire K-digit semantic ID to learn inter-item dependencies), token-level (replaces single digits to learn intra-item token relationships), and hybrid-level (randomly selects item-level or token-level). Experiments show token-level + R→E→Inf curriculum learning achieves best results
    - Design Motivation: Different masking granularities force the model to understand historical sequence structure and dependencies from multiple perspectives

2. **Entropy-Guided Adaptive Masking**:

    - Function: Intelligently selects the most informative historical positions for masked reconstruction
    - Mechanism: At each training step, performs gradient-free forward pass on the unmasked sequence to compute prediction entropy $\mathcal{H}_t^k$ at each position. High-entropy positions indicate insufficient model understanding, corresponding to critical decision points or complex semantic units. Positions are ranked by entropy in descending order, and top-N positions are masked, where N is sampled from uniform distribution $\mathcal{U}(1, \lfloor\gamma \cdot T\rfloor)$ to prevent overfitting
    - Design Motivation: Random masking treats all positions equally, ignoring the uneven information density in user behaviors; entropy guidance ensures reconstruction targets always focus on the model's weakest positions

3. **Three-Phase Curriculum Training Scheduler**:

    - Function: Bridges the gap between masked training and mask-free inference
    - Mechanism: Phase I (Random Masking Warmup)—low-ratio random masking establishes baseline reconstruction capability and stable optimization; Phase II (Entropy-Guided Masking + Adaptive Decay)—switches to high-ratio entropy-guided masking to encourage deep historical understanding, exponentially decays masking ratio $\gamma \leftarrow \max(\gamma_{min}, \gamma \cdot \eta)$ when validation performance plateaus; Phase III (Inference Alignment)—sets $\gamma=0$, trains only $\mathcal{L}_{next}$ to eliminate train-inference discrepancy
    - Design Motivation: Direct use of entropy-guided masking causes training instability (unreliable early entropy estimates), while the three-phase transition achieves smooth shift from "understanding paths" to "generating paths"

### Loss & Training

$$\mathcal{L}_{MHL} = \lambda_1 \mathcal{L}_{next} + \lambda_2 \mathcal{L}_{mask}$$

where $\mathcal{L}_{next}$ is the standard cross-entropy loss for next-item semantic ID, and $\mathcal{L}_{mask}$ is the reconstruction cross-entropy loss for masked positions.

## Key Experimental Results

### Main Results

| Dataset | Metric | MHL | RPG(Prev. SOTA) | Gain |
|---------|--------|-----|-----------------|------|
| Beauty | R@10 | .0795 | .0745 | +6.7% |
| Beauty | N@10 | .0495 | .0436 | +13.5% |
| Toys | R@10 | .0903 | .0778 | +16.1% |
| Toys | N@10 | .0564 | .0460 | +22.6% |
| Sports | R@10 | .0511 | .0436 | +17.2% |
| Sports | N@10 | .0298 | .0246 | +21.1% |

### Ablation Study

| Config | Key Metric | Note |
|--------|-----------|------|
| Token-level + R→E→Inf | Best | Optimal across three datasets |
| No masking (baseline RPG) | .0745 R@10 | Next-item prediction only |
| Random masking | Improvement | Simple masking is effective |
| Entropy-guided only (no curriculum) | Performance drop | Unstable due to unreliable early entropy estimates |
| Truncated recent history | MHL vs SOTA +40% | MHL shows robust performance under history truncation |

### Key Findings
- Masked history reconstruction auxiliary task brings improvements across all three granularities (item/token/hybrid)
- Token-level masking performs best due to finer-grained learning signals
- Entropy-guided masking must be paired with curriculum learning—direct use causes training instability
- When recent interaction history is truncated, MHL demonstrates strong robustness (+40%), proving it learns deep intent rather than relying solely on recency effects
- Phase III (Inference Alignment) is critical—skipping this phase results in significant performance degradation

## Highlights & Insights
- **Training Paradigm Innovation**: Introduces masked language modeling concepts into autoregressive training for generative recommendation, shifting from "predicting outcomes" to "understanding processes"
- **Profound "Understand Past to Predict Future" Insight**: Robustness experiments (history truncation) convincingly demonstrate the value of deep historical understanding
- **Elegant Entropy-Guided + Curriculum Learning Combination**: The two components complement each other—entropy guidance provides high-quality signals, while curriculum learning ensures training stability and inference alignment
- **Strong Method Generalizability**: As a training framework, it can be directly applied to various semantic ID-based generative recommendation models

## Limitations & Future Work
- **Limited Dataset Scale**: Validated only on three categories from Amazon Reviews 2014; testing needed on larger scales and more domains
- **Additional Computational Overhead**: Entropy-guided masking requires extra gradient-free forward passes to compute entropy at each step
- **Hyperparameter Sensitivity**: The three curriculum learning phase transition points and masking ratios require tuning
- Future exploration: Application to LLM-based recommendation systems, cross-domain recommendation, multimodal recommendation

## Related Work & Insights
- **vs BERT4Rec/S3-Rec**: Uses bidirectional encoder + masked prediction for discriminative recommendation; MHL adds masked reconstruction auxiliary task on unidirectional decoder while maintaining generative capability
- **vs TIGER/RPG**: SOTA generative recommendation with pure autoregressive next-item prediction; MHL significantly improves through history reconstruction auxiliary task
- **vs HSTU**: Facebook's industrial-scale sequential recommendation model; MHL outperforms on all metrics

## Rating
- Novelty: ⭐⭐⭐⭐ Introducing masked learning into generative recommendation training paradigm is meaningful innovation; entropy-guided + curriculum learning combination is cleverly designed
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-dataset, multi-baseline comparisons, comprehensive ablation studies and robustness analysis
- Writing Quality: ⭐⭐⭐⭐ Photography enthusiast example is intuitive and easy to understand; method motivation is clearly articulated
- Value: ⭐⭐⭐⭐ Provides new training paradigm for generative recommendation; historical understanding perspective is insightful

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Multi-Aspect Cross-modal Quantization for Generative Recommendation](../../AAAI2026/image_generation/multi-aspect_cross-modal_quantization_for_generative_recommendation.md)
- [\[ICCV 2025\] PanoLlama: Generating Endless and Coherent Panoramas with Next-Token-Prediction LLMs](../../ICCV2025/image_generation/panollama_generating_endless_and_coherent_panoramas_with_next-token-prediction_l.md)
- [\[NeurIPS 2025\] Next Semantic Scale Prediction via Hierarchical Diffusion Language Models](../../NeurIPS2025/image_generation/next_semantic_scale_prediction_via_hierarchical_diffusion_language_models.md)
- [\[ICCV 2025\] AnimeGamer: Infinite Anime Life Simulation with Next Game State Prediction](../../ICCV2025/image_generation/animegamer_infinite_anime_life_simulation_with_next_game_state_prediction.md)
- [\[CVPR 2026\] AS-Bridge: A Bidirectional Generative Framework Bridging Next-Generation Astronomical Surveys](../../CVPR2026/image_generation/as-bridge_a_bidirectional_generative_framework_bridging_next-generation_astronom.md)

</div>

<!-- RELATED:END -->
