---
title: >-
  [Paper Note] Hierarchical Policy Optimization for Simultaneous Translation of Unbounded Speech
description: >-
  [ACL 2026][Segmentation][Simultaneous Translation] This paper proposes Hierarchical Policy Optimization (HPO), which post-trains LLM-based simultaneous speech translation models through hierarchical reward design…
tags:
  - "ACL 2026"
  - "Segmentation"
  - "Simultaneous Translation"
  - "Reinforcement Learning"
  - "Hierarchical Reward"
  - "LLM Speech Translation"
  - "GRPO"
date: 2025-04-17
content_hash: 48d9414e4cd2760b
---

# Hierarchical Policy Optimization for Simultaneous Translation of Unbounded Speech

**Conference**: ACL 2026  
**arXiv**: [2604.21045](https://arxiv.org/abs/2604.21045)  
**Code**: [https://github.com/owaski/HPO](https://github.com/owaski/HPO)  
**Area**: Segmentation  
**Keywords**: Simultaneous Translation, Reinforcement Learning, Hierarchical Reward, LLM Speech Translation, GRPO

## TL;DR
This paper proposes Hierarchical Policy Optimization (HPO), which post-trains LLM-based simultaneous speech translation models through hierarchical reward design, suppressing latency optimization when translation quality falls below threshold, achieving +7 COMET translation quality improvement at 1.5-second latency.

## Background & Motivation

**Background**: Simultaneous speech translation (SST) requires generating translations while receiving partial speech input. Recently, LLM-based methods that model SST as multi-turn dialogue tasks, leveraging KV cache reuse to eliminate redundant computation, have become the mainstream approach for handling unbounded long speech (e.g., InfiniSST).

**Limitations of Prior Work**: These methods heavily rely on synthetic read-write trajectory data for supervised fine-tuning (SFT), but existing trajectory synthesis methods have significant flaws. Word-alignment-tool-based methods ignore the future context needed for translation timing; LLM-simulated interpreter methods produce unstable segmentation, failing to guarantee valid read-write trajectories. This leads to suboptimal SFT data quality and learned erroneous behaviors.

**Key Challenge**: A natural trade-off exists between translation quality and latency. When directly using reinforcement learning to jointly optimize both, the latency reward is easier to optimize (simply translating earlier reduces latency regardless of translation correctness), causing models to over-optimize latency at the expense of translation quality.

**Goal**: Design a post-training method to correct SFT model errors while stably balancing optimization of translation quality and latency.

**Key Insight**: The authors observe that the fundamental reason latency reward dominates optimization is the scale difference and asymmetric optimization difficulty between the two rewards. By introducing a "quality gate" mechanism — allowing latency optimization only after translation quality meets the threshold — an effective hierarchical constraint is established.

**Core Idea**: Use a hierarchical reward structure to constrain GRPO training: when translation quality is below threshold, latency reward is set to worst value, ensuring the model prioritizes accuracy before pursuing speed.

## Method

### Overall Architecture
The system consists of a streaming speech encoder and an LLM translator. Speech is chunked by fixed duration; the encoder incrementally encodes each new chunk reusing previous KV cache, and the LLM decodes the next translation segment based on interleaved speech features and previously generated translations. HPO samples multiple translation hypotheses on top of this SFT model, computes hierarchical rewards, and optimizes with GRPO.

### Key Designs

1. **Sentence-Level Segmentation and Alignment (SEGALE)**:

    - Function: Segment long-text translation hypotheses into sentences and align with reference translations
    - Mechanism: Use spaCy for sentence segmentation, then an embedding-based aligner with adaptive search to align hypothesis and reference sentences, simultaneously identifying over-translation ($R_k = \phi$) and under-translation ($H_k = \phi$)
    - Design Motivation: Traditional mwersegmenter forces alignment by minimizing word error rate, assigning alignment even to meaningless gibberish, producing spuriously high scores (reward hacking) when paired with non-robust neural metrics

2. **Hierarchical Reward**:

    - Function: Jointly evaluate translation quality and latency, preventing latency over-optimization
    - Mechanism: For each aligned sentence pair, compute quality score $q^{j,k}$ (MetricX) and latency score $l^{j,k}$ (LAAL) separately. If quality score falls below threshold $q_{\text{thres}}$, latency score is set to worst value $l_{\max}$. Then average across all sentences within each hypothesis, apply group normalization separately for within-group samples, and finally compute weighted sum for final reward $r^j = \bar{q}^j - \lambda \cdot \bar{l}^j$
    - Design Motivation: Direct weighted combination of quality and latency rewards causes the model to prefer "translating early but poorly"; through quality gating, low latency is only rewarded when translation is sufficiently good, establishing a "quality first, speed second" optimization priority

3. **Group Normalization Strategy**:

    - Function: Eliminate scale differences between quality and latency rewards
    - Mechanism: For $n$ hypotheses sampled from the same prompt, apply group normalization (subtract mean, divide by standard deviation) separately to quality and latency scores, bringing both rewards to the same scale
    - Design Motivation: Quality metrics (e.g., MetricX range -25 to 0) and latency metrics (seconds) have inherently different scales; direct addition causes training instability

### Loss & Training
GRPO framework is used, sampling 16 translation trajectories per speech segment, with clipped importance sampling and on-policy KL divergence regularization. MetricX serves as the quality reward model, quality threshold $q_{\text{thres}} = -5$, latency weight $\lambda = 0.5$, KL penalty weight 0.01. Training on 3 × 8×H100 nodes for approximately 20 hours (500 steps), with 1 node dedicated to reward computation.

## Key Experimental Results

### Main Results
On the ACL 60/60 dev set for En→Zh/De/Ja three directions, HPO significantly outperforms the InfiniSST baseline on COMET, MetricX, and BLEURT metrics.

| Direction | Latency (s) | COMET Gain | MetricX Gain | BLEURT Gain |
|------|---------|-----------|-------------|------------|
| En→Zh | ~1.5s | +7 | +1.25 | +4 |
| En→De | ~1.5s | Significant | Significant | Significant |
| En→Ja | ~1.5s | Significant | Significant | Significant |

### Ablation Study

| Config | StreamLAAL | COMET | MetricX |
|------|-----------|-------|---------|
| SFT (baseline) | 1216 | 0.7348 | -4.52 |
| Normalize | 1555 | 0.7977 | -3.41 |
| Normalize + Truncation (SeqPO) | 1805 | 0.8058 | -3.39 |
| Normalize + Hierarchical-Doc | 1544 | 0.8157 | -3.27 |
| Normalize + Hierarchical-Sent (HPO) | 1383 | 0.8234 | -3.21 |

### Key Findings
- Sentence-level hierarchical reward (Hierarchical-Sent) comprehensively outperforms document-level hierarchical reward and simple truncation methods, achieving better translation quality at lower latency
- MetricX is the only one among six quality reward functions that performs consistently across all automatic metrics and human evaluation
- BLEU is the sole exception where HPO does not necessarily outperform baseline; Gemini evaluation also confirms that neural reward-based optimization may lead to reward hacking
- Models using mwersegmenter exploit weaknesses in segmentation and metrics for reward hacking, with gibberish hypotheses also receiving high scores

## Highlights & Insights
- The hierarchical reward design is highly clever: rather than simply weighting two objectives, it establishes a hard "quality first" constraint, allowing speed optimization only when translation quality is sufficient. This approach generalizes to all "primary + auxiliary objective" multi-objective RL scenarios
- Exposing the mwersegmenter + neural metric reward hacking vulnerability, where gibberish text achieves near-perfect MetricX scores after segmentation, is an important finding for the translation evaluation field
- HPO even surpasses offline translation model quality in some settings, demonstrating that RL post-training can not only correct SFT errors but also discover strategies superior to offline full-context translation

## Limitations & Future Work
- Validated only on one architecture (InfiniSST), one data synthesis method, and English as source language across three directions
- MetricX as reward model remains imperfect, sometimes favoring fluency over accuracy, potentially causing reward hacking
- BLEU and Gemini evaluation reveal over-optimization risks with neural rewards, requiring more robust quality reward models
- Effects of applying HPO to offline translation models (standard GRPO) are unexplored

## Related Work & Insights
- **vs InfiniSST (SFT)**: InfiniSST only uses synthetic trajectories for SFT; HPO corrects erroneous behaviors through RL post-training, significantly outperforming SFT at all latency levels
- **vs SeqPO-SiMT**: SeqPO handles latency reward through truncation + normalization but is limited to text translation without supporting unbounded speech; HPO's hierarchical reward outperforms truncation in ablation
- **vs Traditional RL-SST**: Previous RL methods are all based on encoder-decoder Transformer text translation; HPO is the first to extend RL to LLM-based unbounded speech simultaneous translation

## Rating
- Novelty: ⭐⭐⭐⭐ Hierarchical reward approach is simple yet effective, though the GRPO framework itself is not a new contribution
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three language directions, six reward functions, multi-dimensional ablation, human evaluation, reward hacking analysis — very comprehensive
- Writing Quality: ⭐⭐⭐⭐⭐ Problem motivation is clear, method description is logically rigorous, figures are well-designed
- Value: ⭐⭐⭐⭐ Directly guides RL training for simultaneous translation; hierarchical reward concept has universal applicability

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Universal Multi-Domain Translation via Diffusion Routers](../../ICLR2026/segmentation/universal_multi-domain_translation_via_diffusion_routers.md)
- [\[CVPR 2026\] ConceptPrism: Concept Disentanglement in Personalized Diffusion Models via Residual Token Optimization](../../CVPR2026/segmentation/conceptprism_concept_disentanglement_in_personalized_diffusion_models_via_residu.md)
- [\[CVPR 2026\] GeoGuide: Hierarchical Geometric Guidance for Open-Vocabulary 3D Semantic Segmentation](../../CVPR2026/segmentation/geoguide_hierarchical_geometric_guidance_for_open-vocabulary_3d_semantic_segment.md)
- [\[CVPR 2026\] RSONet: Region-guided Selective Optimization Network for RGB-T Salient Object Detection](../../CVPR2026/segmentation/rsonet_region-guided_selective_optimization_network_for_rgb-t_salient_object_det.md)
- [\[AAAI 2026\] Breaking the Stealth-Potency Trade-off in Clean-Image Backdoors with Generative Trigger Optimization](../../AAAI2026/segmentation/breaking_the_stealth-potency_trade-off_in_clean-image_backdoors_with_generative_.md)

</div>

<!-- RELATED:END -->
