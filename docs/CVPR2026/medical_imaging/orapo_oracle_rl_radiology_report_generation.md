---
title: >-
  [Paper Note] OraPO: Oracle-educated Reinforcement Learning for Data-efficient and Factual Radiology Report Generation
description: >-
  [CVPR 2026][Medical Imaging][radiology report generation] OraPO is proposed as an adaptive hybrid RL framework combining GRPO and DPO for data-efficient radiology report generation. It dynamically switches between GRPO and DPO via Zero-Reward Rate detection, and employs a FactScore-based clinical fact-level reward. Using only 1K samples (vs. 227K for baselines), OraPO achieves state-of-the-art clinical F1 scores of 0.341/0.357 on CheXpert Plus and MIMIC-CXR.
tags:
  - CVPR 2026
  - Medical Imaging
  - radiology report generation
  - reinforcement-learning
  - GRPO
  - DPO
  - fact-based reward
  - data efficiency
date: 2026-05-08
content_hash: aca27e392a1df106
---

# OraPO: Oracle-educated Reinforcement Learning for Data-efficient and Factual Radiology Report Generation

**Conference**: CVPR 2026
**arXiv**: [2509.18600](https://arxiv.org/abs/2509.18600)
**Code**: Not available
**Area**: Medical Imaging / LLM / Report Generation
**Keywords**: radiology report generation, reinforcement-learning, GRPO, DPO, fact-based reward, data efficiency

## TL;DR
OraPO is proposed as an adaptive hybrid RL framework combining GRPO and DPO for data-efficient radiology report generation. It dynamically switches between GRPO and DPO via Zero-Reward Rate detection, and employs a FactScore-based clinical fact-level reward. Using only 1K samples (vs. 227K for baselines), OraPO achieves state-of-the-art clinical F1 scores of 0.341/0.357 on CheXpert Plus and MIMIC-CXR.

## Background & Motivation
**State of the Field**: Mainstream radiology report generation (RRG) relies on multi-stage training, large-scale paired corpora, and large backbone models, incurring substantial computational and data costs.

**Limitations of Prior Work**: (a) Vanilla GRPO fails on RRG — approximately 30% of rollout groups receive all-zero rewards, leading to gradient vanishing and wasted computation; (b) designing sentence-level rewards that capture clinical accuracy rather than surface similarity is difficult; (c) large-scale annotated data is hard to acquire.

**Root Cause**: GRPO generates numerous zero-reward groups when early outputs are highly uncertain, and these failed explorations are entirely wasted; a mechanism is needed to convert failed rollouts into useful learning signals.

**Paper Goals**: (a) Achieve state-of-the-art report generation with minimal data (1K samples); (b) design rewards that capture clinical factual correctness; (c) resolve the GRPO zero-reward problem.

**Starting Point**: Monitor the Zero-Reward Rate and automatically switch to DPO when ZRR is high — using ground-truth reports as positives and failed rollouts as negatives, thereby converting wasted exploration into preference learning signals.

**Core Idea**: Use EMA-smoothed ZRR to detect GRPO failure moments, dynamically incorporate DPO to convert failed rollouts into preference pairs, and apply atomic fact-level rewards to capture clinical correctness.

## Method

### Overall Architecture
A lightweight VLM based on Qwen2.5-VL (3B). For each prompt, $K$ rollouts are sampled and evaluated with FactScore-based rewards. GRPO and DPO losses are adaptively mixed via EMA-smoothed ZRR.

### Key Designs

1. **Zero-Reward Rate (ZRR) Detection and Adaptive Mixing**:

    - **Function**: Detects the proportion of zero-reward GRPO groups and automatically supplements with DPO training signals when ZRR is high.
    - **Mechanism**: ZRR is smoothed via EMA and mapped to a mixing weight $w_i$; $\mathcal{L}_\text{OraPO} = (1-w_i)\mathcal{L}_\text{GRPO} + w_i\mathcal{L}_\text{DPO}$. Ground-truth reports serve as DPO positives; all GRPO rollouts serve as negatives.
    - **Design Motivation**: Approximately 30% of GRPO groups receive all-zero rewards in early training, causing gradient vanishing. DPO does not rely on absolute rewards — requiring only preference ordering — and can learn from failures.

2. **FactScore-based Reward (FactS)**:

    - **Function**: Designs a dense reward based on atomic clinical facts rather than report-level text similarity.
    - **Mechanism**: (a) Atomic clinical facts are extracted from generated reports using GPT-4; (b) entailment checking is performed against 14 CheXpert pathology labels; (c) precision/recall are computed, and $F_\beta$ ($\beta > 1$ to emphasize recall) is used as the reward.
    - **Design Motivation**: Clinically, missed detections (false negatives) are more dangerous than false alarms (false positives), hence the emphasis on recall; atomic facts more accurately capture clinical correctness than sentence-level BLEU/ROUGE.

3. **Oracle-educated DPO Component**:

    - **Function**: Converts GRPO failed rollouts into DPO negatives, with ground-truth reports as positives.
    - **Mechanism**: No additional data is required; GRPO sampling results are directly reused.
    - **Design Motivation**: Creates a self-reinforcing flywheel — poor rollouts → DPO negatives → better policy → more informative rollouts → higher rewards.

### Loss & Training
- $\mathcal{L}_\text{OraPO} = (1-w_i)\mathcal{L}_\text{GRPO} + w_i\mathcal{L}_\text{DPO}$, with $w_\min=0.05$, $w_\max=0.15$, $\gamma=2.0$ (sharpening)
- EMA momentum $\alpha=0.5$; $K$ = group sample size
- Base model: Qwen2.5-VL (3B), 4× A10 GPUs
- Inference speed: 3.3s/image (vs. 25.2s for GPT-5)

## Key Experimental Results

### Main Results: CheXpert Plus (Clinical F1)

| Method | Train Size | Precision | Recall | F1 |
|---|---|---|---|---|
| MambaXray-L (CVPR'25) | 1.27M | 0.377 | 0.319 | 0.335 |
| R2GenGPT | 223K | 0.315 | 0.244 | 0.260 |
| OraPO (Ours) | **1K** | 0.237 | **0.832** | **0.341** |

### MIMIC-CXR Results

| Method | Train Size | F1 |
|---|---|---|
| MambaXray-L | 1.27M | 0.340 |
| MCA-RG | 227K | 0.335 |
| OraPO (Ours) | **1K** | **0.357** |

### Ablation Study

| Configuration | Train Size | Precision | Recall | F1 |
|---|---|---|---|---|
| Base model | 0 | 0.097 | 0.104 | 0.034 |
| + GRPO only | 1K | 0.026 | 0.162 | 0.089 |
| + GRPO + FactS | 1K | 0.204 | 0.605 | 0.291 |
| + FactS + OraPO (400) | 400 | 0.217 | 0.732 | 0.296 |
| + FactS + OraPO (1K) | 1K | 0.237 | 0.832 | 0.341 |

### Key Findings
- The FactS reward contributes most significantly: F1 improves from 0.089 to 0.291 (+200%).
- OraPO further improves F1 by 17.2% (+37.5% recall) on top of FactS.
- OraPO with only 400 samples already outperforms GRPO+FactS with 1K samples (0.296 vs. 0.291).
- SFT severely degrades recall: from 0.732 to 0.176, as SFT causes the model to become overly conservative.
- The 3B model outperforms GPT-4.1 (F1: 0.341 vs. 0.253) with 7.6× faster inference.

## Highlights & Insights
- **Converting GRPO failures into DPO signals is the core innovation**: No exploration is wasted — "failed" rollouts become negatives for preference learning.
- **FactScore atomic fact-level reward**: More accurately measures clinical correctness than report-level BLEU/ROUGE and is transferable to other medical text generation tasks.
- **Extreme data efficiency**: 1K samples outperform models trained on 1.27M samples — a three-order-of-magnitude efficiency gain.
- **Recall-first clinical design**: $\beta > 1$ emphasizes recall, consistent with the clinical principle that false negatives are more dangerous than false positives.

## Limitations & Future Work
- Precision is low (0.237 vs. 0.377 for baselines); excessive pursuit of recall may introduce more false positives.
- FactScore relies on GPT-4 for atomic fact extraction, incurring additional API costs.
- Validation is limited to chest X-rays; extension to other imaging modalities (CT, MRI, etc.) has not been explored.
- The 14 CheXpert labels may be insufficient to cover all clinical findings.

## Related Work & Insights
- **vs. MambaXray-L**: Requires 1.27M samples, whereas OraPO uses only 1K and achieves higher F1, demonstrating the potential of RL in low-data regimes.
- **vs. GPT-5**: OraPO inference takes 3.3s vs. 25.2s for GPT-5, with no API costs; however, GPT-5 is marginally superior on human gold evaluation.
- OraPO can serve as a paradigm template for low-data medical report generation.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The OraPO hybrid GRPO/DPO framework combined with FactScore reward is an elegant system design.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Two datasets, complete ablations, human gold evaluation, and GPT comparison.
- **Writing Quality**: ⭐⭐⭐⭐ Method motivation is clear; ZRR analysis is intuitive.
- **Value**: ⭐⭐⭐⭐⭐ Extreme data efficiency has substantial practical value in medical imaging.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Unleashing Video Language Models for Fine-grained HRCT Report Generation](unleashing_video_language_models_for_fine-grained_hrct_report_generation.md)
- [\[CVPR 2026\] CURE: Curriculum-guided Multi-task Training for Reliable Anatomy Grounded Report Generation](cure_curriculum-guided_multi-task_training_for_reliable_anatomy_grounded_report_.md)
- [\[CVPR 2026\] Towards Efficient Medical Reasoning with Minimal Fine-Tuning Data](towards_efficient_medical_reasoning_with_minimal_fine-tuning_data.md)
- [\[CVPR 2026\] CHIPS: Efficient CLIP Adaptation via Curvature-aware Hybrid Influence-based Data Selection](chips_efficient_clip_adaptation_via_curvature-aware_hybrid_influence-based_data_.md)
- [\[CVPR 2026\] MedGRPO: Multi-Task Reinforcement Learning for Heterogeneous Medical Video Understanding](medgrpo_multi-task_reinforcement_learning_for_heterogeneous_medical_video_unders.md)

<!-- RELATED:END -->
