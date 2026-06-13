---
title: >-
  [Paper Note] OraPO: Oracle-educated Reinforcement Learning for Data-efficient and Factual Radiology Report Generation
description: >-
  [CVPR2026][Medical Imaging][Radiology report generation] OraPO (Oracle-educated GRPO) injects lightweight DPO supervision when GRPO exploration fails…
tags:
  - "CVPR2026"
  - "Medical Imaging"
  - "Radiology report generation"
  - "GRPO"
  - "DPO"
  - "reinforcement learning"
  - "data efficiency"
  - "clinical factuality scoring"
date: 2026-05-08
content_hash: 764b6b075d069071
---

# OraPO: Oracle-educated Reinforcement Learning for Data-efficient and Factual Radiology Report Generation

**Conference**: CVPR2026  
**arXiv**: [2509.18600](https://arxiv.org/abs/2509.18600)  
**Code**: To be confirmed  
**Area**: Medical Imaging  
**Keywords**: Radiology report generation, GRPO, DPO, reinforcement learning, data efficiency, clinical factuality scoring

## TL;DR

OraPO (Oracle-educated GRPO) injects lightweight DPO supervision when GRPO exploration fails, converting zero-reward rollouts into preference pairs. Combined with a FactScore reward, the method achieves SOTA radiology report generation on CheXpert Plus and MIMIC-CXR (F1=0.341/0.357) using only 1K training samples and a 3B model—reducing training data by 2–3 orders of magnitude compared to prior best methods.

## Background & Motivation

**Clinical urgency**: Radiology backlogs are severe—29% consultant vacancy rate in the UK, ~14,000 unfilled positions in the US with only ~1,150 graduates per year. AI-assisted drafting has been shown to reduce reporting time by 24%.

**High cost of existing methods**: Current RRG approaches rely on multi-stage training (domain pre-training → vision-language alignment → task fine-tuning) and large paired corpora (≥223K samples); some use >13B parameter models with substantial GPU budgets.

**Exploration failure in GRPO**: Applying GRPO to RRG with a base VLM lacking radiology domain knowledge causes ~30% of groups to produce all-zero rewards in the first 50 steps, leading to vanished gradients and wasted rollouts.

**Inadequacy of existing fixes**: Resampling until non-zero rewards appear (DAPO) or increasing group size both raise compute costs; alternating SFT+RL still discards low-quality rollouts.

**Reward design difficulty**: Unlike math or coding with binary verification, radiology reports are long, multi-fact narratives. Metrics such as BLEU/CIDEr capture only surface fluency and insufficiently penalize sentence-level factual errors or cross-sentence contradictions.

**Data efficiency gap**: Prior work focuses on training stability rather than data efficiency, leaving training set size and epoch count largely unchanged.

## Method

### Overall Architecture

OraPO is a single-stage, pure RL training framework (no SFT or alignment pre-training), consisting of two core components:

- **OraPO algorithm**: Built on GRPO; when all rollouts in a sampled group receive zero reward, DPO supervision is dynamically injected—constructing preference pairs with the ground-truth report as the positive sample and zero-reward rollouts as negatives.
- **FactScore reward (FactS)**: Atomic clinical facts are extracted from generated reports and checked for entailment against ground-truth labels, yielding dense, interpretable sentence-level rewards.

### Key Design 1: Zero-Reward Rate (ZRR) Adaptive Mixing

For each prompt $x_i$, the proportion of zero-reward rollouts among $K$ samples is computed as $z_i$, smoothed via exponential moving average (EMA, $\alpha=0.5$) to obtain $\tilde{z}_i^{(t)}$, then mapped to a mixing weight:

$$w_i^{(t)} = \text{clip}(w_{\min} + (w_{\max} - w_{\min})[\tilde{z}_i^{(t)}]^\gamma, w_{\min}, w_{\max})$$

where $w_{\min}=0.05$, $w_{\max}=0.15$, $\gamma=2.0$. The final OraPO objective is:

$$\mathcal{L}_{\text{OraPO}} = \frac{1}{B}\sum_{i=1}^{B}[(1 - w_i^{(t)})\mathcal{L}_{\text{GRPO}} + w_i^{(t)}\mathcal{L}_{\text{DPO}}]$$

- High ZRR → DPO dominates (oracle education); low ZRR → GRPO dominates (exploration exploitation).
- DPO positives = ground-truth reports; negatives = all zero-reward rollouts (free negatives requiring no additional annotation or generation).

### Key Design 2: FactScore Reward (FactS)

Three-step pipeline:

1. **Atomic fact extraction**: GPT-4.1 extracts a set of atomic clinical statements $\mathcal{F}(\hat{y}_i)$ from the generated report.
2. **Label-level entailment checking**: For each of the 14 CheXpert labels, the fact set is checked for entailment; contradictions are treated as false positives.
3. **$F_\beta$ reward computation**: Per-instance precision/recall is used to compute an $F_\beta$ score with $\beta > 1$ to emphasize recall (penalizing missed findings).

### Loss & Training

- GRPO loss: Standard clipped PPO ratio + KL regularization; DR.GRPO is applied to mitigate length bias.
- DPO loss: Standard DPO + LN-DPO for sequence-length-normalized preference margin.
- Both losses are dynamically mixed via ZRR weights, forming a self-reinforcing data flywheel: better model → higher-quality negatives → stronger reward signal → better model.

## Key Experimental Results

### Main Results

| Dataset | Method | Precision | Recall | F1 | Training Samples |
|:------|:-----|:---------|:-------|:---|:--------|
| CheXpert Plus | MambaXray-L (CVPR25) | 0.377 | 0.319 | 0.335 | 1.27M |
| CheXpert Plus | **OraPO (Ours)** | **0.237** | **0.832** | **0.341** | **1K** |
| MIMIC-CXR | MambaXray-L (CVPR25) | 0.371 | 0.321 | 0.340 | 1.27M |
| MIMIC-CXR | **OraPO (Ours)** | **0.242** | **0.891** | **0.357** | **1K** |

- Achieves SOTA F1 on CheXpert Plus (0.341); Recall improves over prior best by **+160.8%**.
- F1=0.357 on MIMIC-CXR, a +5.0% gain over MambaXray-L, with Recall +153.8%.
- Only 1K training samples vs. 1.27M for MambaXray-L—a ~1,270× reduction.

### Ablation Study

| FactS | GRPO | DPO | Training Size | Precision | Recall | F1 |
|:------|:-----|:----|:------|:---------|:-------|:---|
| ✗ | ✗ | ✗ | 0 | 0.097 | 0.104 | 0.034 |
| ✗ | ✓ | ✗ | 1K | 0.026 | 0.162 | 0.089 |
| ✓ | ✓ | ✗ | 1K | 0.204 | 0.605 | 0.291 |
| ✓ | ✓ | ✓ | 400 | 0.217 | 0.732 | 0.296 |
| ✓ | ✓ | ✗ (SFT) | 1K | 0.171 | 0.176 | 0.106 |
| ✓ | ✓ | ✓ | 1K | **0.237** | **0.832** | **0.341** |

### Key Findings

- **FactS is the core driver**: Adding FactS raises F1 from 0.089 to 0.291 (+227%), demonstrating that accuracy-based rewards are wholly insufficient for RRG.
- **OraPO further improves F1 by 17.2%** on top of FactS alone; OraPO with 400 samples already outperforms FactS-only with 1K samples.
- **Replacing DPO with SFT causes collapse**: GRPO+SFT yields only Recall=0.176 and F1=0.106, as SFT teaches the model how to say correct things but not how to avoid saying incorrect ones.
- **Gold-label validation**: On radiologist-annotated CheXpert validation data, OraPO surpasses MambaXray-L (F1 0.288 vs. 0.280) and GPT-4.1 (F1 0.288 vs. 0.253).
- **Inference efficiency**: The 3B model runs at 3.3s/image, vs. 25.2s/image for GPT-5 Thinking.

## Highlights & Insights

- **Extreme data efficiency**: 1K samples surpass SOTA trained on millions; training data reduced by 2–3 orders of magnitude, requiring only 4×A10 GPUs.
- **First GRPO+DPO integration**: Recycling failed exploration as preference negatives is an elegant approach with near-zero additional overhead.
- **ZRR adaptive mechanism**: Automatically balances oracle education and RL exploration, forming a positive feedback flywheel.
- **FactScore reward**: Anchors report quality assessment to atomic clinical fact entailment checking, more clinically meaningful than BLEU/CIDEr.
- **Recall-oriented design**: High sensitivity (0.832/0.891) aligns with clinical requirements—missed diagnoses carry far greater consequences than false positives.

## Limitations & Future Work

- **Low precision** (0.237/0.242): The high-recall regime introduces a non-trivial false-positive rate, requiring final review by radiologists.
- **FactS depends on GPT-4.1** for fact extraction and entailment checking, introducing external API costs and potential instability.
- **Validated only on chest X-ray RRG**; not extended to other imaging modalities (CT, MRI) or other clinical tasks.
- **Experiments limited to the 3B model**; scaling behavior with larger or smaller models remains unexplored.
- **Hyperparameters** such as $w_{\min}$/$w_{\max}$ require tuning, and the search range reported in the paper is relatively narrow.

## Related Work & Insights

- **RRG method evolution**: CNN-RNN/Transformer seq2seq → knowledge-guided generation → multi-stage pre-training → LLM-driven instruction fine-tuning; the shared bottleneck is data and compute intensity.
- **GRPO variants**: Originated in DeepSeekMath; DR.GRPO addresses length bias; DAPO resamples for non-zero rewards—none target data efficiency.
- **DPO variants**: SimPO (length normalization), ORPO (reference-free), KTO (unary signal); OraPO uses DPO as an oracle step when GRPO fails.
- **Strongest baselines**: MambaXray-L (CVPR25, 1.27M samples, F1=0.335/0.340), CheXagent (8.5M samples).

## Rating

- Novelty: ⭐⭐⭐⭐ — The GRPO+DPO fusion is novel; the ZRR adaptive mixing design is concise and effective.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Two large datasets, 28+ baselines, detailed ablations, gold-label validation, and comparisons with commercial APIs.
- Writing Quality: ⭐⭐⭐⭐ — Problem motivation is clear, method derivation is complete, and experimental analysis is thorough.
- Value: ⭐⭐⭐⭐⭐ — Extreme data efficiency has strong practical value in medical settings; recall-oriented design aligns well with clinical needs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] CAME-Grad: The Double Dilemma in Multi-Task Radiology Report Generation — A Gradient Dynamics Analysis and Solution](../../ICML2026/medical_imaging/the_double_dilemma_in_multi-task_radiology_report_generation_a_gradient_dynamics.md)
- [\[CVPR 2026\] Unleashing Video Language Models for Fine-grained HRCT Report Generation](unleashing_video_language_models_for_fine-grained_hrct_report_generation.md)
- [\[CVPR 2026\] MedCLIPSeg: Probabilistic Vision-Language Adaptation for Data-Efficient and Generalizable Medical Image Segmentation](medclipseg_probabilistic_vision-language_adaptation_for_data-efficient_and_gener.md)
- [\[CVPR 2026\] MedGRPO: Multi-Task Reinforcement Learning for Heterogeneous Medical Video Understanding](medgrpo_multi-task_reinforcement_learning_for_heterogeneous_medical_video_unders.md)
- [\[CVPR 2026\] CURE: Curriculum-guided Multi-task Training for Reliable Anatomy Grounded Report Generation](cure_curriculum-guided_multi-task_training_for_reliable_anatomy_grounded_report_.md)

</div>

<!-- RELATED:END -->
