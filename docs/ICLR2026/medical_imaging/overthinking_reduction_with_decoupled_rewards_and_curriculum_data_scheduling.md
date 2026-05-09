---
title: >-
  [Paper Note] Overthinking Reduction with Decoupled Rewards and Curriculum Data Scheduling
description: >-
  [ICLR 2026][Medical Imaging][overthinking] This paper theoretically identifies two fundamental flaws in existing length-penalty approaches—incorrectly penalizing high-entropy exploration tokens and erroneously rewarding redundant tokens—and proposes the DeCS framework. Through decoupled token-level rewards and curriculum batch scheduling, DeCS reduces reasoning tokens by over 50% across 7 benchmarks while maintaining or even improving model performance.
tags:
  - ICLR 2026
  - Medical Imaging
  - overthinking
  - decoupled rewards
  - curriculum learning
  - RLVR
  - NRP
date: 2026-05-08
content_hash: 898a600fc4fcaa63
---

# Overthinking Reduction with Decoupled Rewards and Curriculum Data Scheduling

**Conference**: ICLR 2026
**arXiv**: [2509.25827](https://arxiv.org/abs/2509.25827)
**Code**: [github.com/pixas/DECS](https://github.com/pixas/DECS)
**Area**: Medical Imaging
**Keywords**: overthinking, decoupled rewards, curriculum learning, RLVR, NRP

## TL;DR

This paper theoretically identifies two fundamental flaws in existing length-penalty approaches—incorrectly penalizing high-entropy exploration tokens and erroneously rewarding redundant tokens—and proposes the DeCS framework. Through decoupled token-level rewards and curriculum batch scheduling, DeCS reduces reasoning tokens by over 50% across 7 benchmarks while maintaining or even improving model performance.

## Background & Motivation

**Background**: Large reasoning models (LRMs) demonstrate strong reasoning capabilities via RLVR, yet suffer from a severe "overthinking" problem—models continue generating redundant reasoning steps after reaching a correct answer, resulting in poor inference efficiency.

**Limitations of Prior Work**: Existing methods encourage concise reasoning by incorporating a length penalty into the correctness reward, $r'(\boldsymbol{o}_i) = r(\boldsymbol{o}_i) - \gamma |\boldsymbol{o}_i|$, but efficiency gains often come at the cost of performance degradation, failing to achieve an optimal efficiency–performance trade-off.

**Key Challenge**: A fundamental misalignment exists between trajectory-level length rewards and token-level policy optimization: (1) negative advantages are back-propagated to all tokens, incorrectly suppressing valid high-entropy exploration tokens (e.g., "wait", "however"); (2) redundant tokens following the NRP in shorter trajectories still receive positive advantages and are thus erroneously reinforced.

**Goal**: To precisely identify and penalize redundant tokens while protecting tokens that genuinely contribute to reasoning, thereby achieving truly lossless reasoning compression.

**Key Insight**: The paper defines the "Necessary Reasoning Prefix" (NRP) as a criterion, decoupling rewards at the NRP boundary and applying differentiated reward signals to tokens before and after it.

**Core Idea**: Train a lightweight discriminator to identify NRP boundaries; tokens within the NRP receive a maximum reward, while redundant tokens beyond the NRP receive a progressively decaying penalty. Curriculum scheduling controls the proportion of easy samples to preserve high-entropy exploration capacity.

## Method

### Overall Architecture

(1) Fine-tune a lightweight language model $\mathcal{M}_{\text{judge}}$ to detect the NRP boundary in each correct trajectory; (2) design decoupled token-level rewards, granting $r_+$ to tokens within the NRP and position-decaying low rewards to redundant tokens beyond it; (3) apply a curriculum scheduling strategy that adaptively adjusts the proportion of easy samples $\kappa_m$ based on the NRP ratio in the current batch.

### Key Designs

1. **NRP Detection and Decoupled Reward**:
   - **Function**: Precisely identify the boundary in each correct trajectory beyond which the correct answer can already be derived, and assign differentiated token-level rewards accordingly.
   - **Mechanism**: A lightweight model $\mathcal{M}_{\text{judge}}$ segments the reasoning process into chunks $\{s_1, \ldots, s_{|S|}\}$ and judges whether each chunk already contains the correct answer: $j_{s_c} \sim \mathcal{M}_{\text{judge}}(\cdot | q, s_c, y^*)$. The NRP is defined as the first chunk containing the correct answer together with all preceding chunks. Token-level rewards are: $r_{i,j} = r_+ \cdot \mathbf{1}_{\text{correct}}$ (when $j \leq K_{o_i}^*$), or $r_{i,j} = (r_0 - (r_+ - r_0)L_i/L_{\max}) \cdot \mathbf{1}_{\text{correct}}$ (when $j > K_{o_i}^*$ and the token is a thinking token).
   - **Design Motivation**: Theorem 2 proves that under sequence-level length rewards, the gradient signal for the first redundant token immediately following the NRP satisfies $\mathcal{J}(A; j=K^*+1) > 0$—i.e., the model is encouraged to continue generating rather than stopping. The decoupled reward ensures any leading redundant token beyond the NRP receives a negative advantage, thereby exploiting the autoregressive property to suppress the entire redundant segment.

2. **Curriculum Prompt Schedule**:
   - **Function**: Adaptively control the proportion of easy samples (prompts for which all rollouts are correct) in training batches.
   - **Mechanism**: $\kappa_m = \text{clip}(\kappa_{m-1} + \beta(\mathcal{R}_m - \mathcal{R}_{m-1}), 0, \kappa_m^0)$, where $\mathcal{R}_m$ is the NRP ratio among correct sequences in the current batch. When the NRP ratio increases (i.e., redundancy decreases), more easy samples are permitted in training. Theorem 1 provides the condition $\kappa \sigma_L < C$ for maintaining the generation probability of high-entropy tokens.
   - **Design Motivation**: Easy samples are the primary source of efficiency gains (since length becomes the only discriminating signal when all rollouts are correct), but an excessive proportion of easy samples causes the logit decline of high-entropy tokens to dominate the batch gradient, leading to performance degradation. Curriculum scheduling achieves a dynamic balance between exploration and compression.

3. **Theoretical Analysis Framework**:
   - **Function**: Provide theoretical grounding for the method design.
   - **Mechanism**: Lemma 1 establishes a linear relationship between logit changes and advantages under policy gradient; Lemma 2 proves that length penalties cause the expected logit change of high-entropy tokens to be strictly negative; Theorem 1 gives a necessary and sufficient condition for maintaining high-entropy tokens during batch learning; Theorem 2 proves that sequence-level length rewards cannot halt generation at the NRP boundary.
   - **Design Motivation**: The failure of existing methods is not an accidental empirical phenomenon but is theoretically grounded—this directly guides the precise design of decoupled rewards and curriculum scheduling.

### Loss & Training

A GRPO-based PPO surrogate loss (Eq. 3) is used, with token-level advantage $A_{i,j}^{\text{DeCS}} = (r_{i,j} - \text{mean})/\text{std}$. Hyperparameters: $r_+=1.1$, $r_0=1.0$, $\beta=0.2$. The training set is DeepScaleR (40k math problems) with 16 rollouts per prompt. Base models are DS-1.5B and DS-7B, trained with the veRL framework.

## Key Experimental Results

### Main Results

| Dataset | Metric | DeCS (1.5B) | Base (1.5B) | Best Baseline | Note |
|---------|--------|-------------|-------------|---------------|------|
| 7-benchmark avg. | Pass@1 | 47.78 | 45.21 | 45.83 (ThinkPrune) | +2.57; both efficiency and performance improved |
| 7-benchmark avg. | #Token | 4000 | 9340 | 3975 (ThinkPrune) | 57.17% reduction |
| 7-benchmark avg. | AES | 0.74 | 0.00 | 0.62 (ThinkPrune) | Best AES |
| AIME2024 (1.5B) | Pass@1 | 31.25 | 27.99 | 29.87 (TLMRE) | +3.26 gain |
| AIME2024 (1.5B) | #Token | 5550 | 12202 | 5306 (ThinkPrune) | 54.5% reduction |
| 7-benchmark avg. (7B) | Pass@1 | 62.48 | 61.57 | 62.17 (ThinkPrune) | +0.91 |
| 7-benchmark avg. (7B) | #Token | 3968 | 7857 | 4940 (ThinkPrune) | 49.5% reduction |

### Ablation Study

| Configuration | Pass@1 | #Token | Note |
|---------------|--------|--------|------|
| DR only (Decoupled Reward) | Improved but ~25% redundancy remains | Limited | Without scheduling, high-entropy tokens are over-suppressed |
| CS only (Curriculum Schedule) | Performance degraded | Limited reduction | Without decoupled reward, redundancy cannot be precisely penalized |
| DR+CS (Full DeCS) | Optimal | Maximum reduction | The two components are complementary |
| Qwen3-4B backbone | 69.72 (+1.32) | 4115 (54.8% reduction) | AES 0.61; good generalizability across backbones |

### Key Findings

- DeCS reduces tokens by over 50% while maintaining or improving Pass@1; the Pass@K curve nearly overlaps with that of the base model, confirming that exploration capacity is preserved.
- Although the NRP detector is trained on math corpora, it remains effective on out-of-domain tasks (GPQA-D: 56.33% reduction; LCB: 33.52% reduction).
- Token-level analysis shows that DeCS primarily reduces "self-correction/verification" and "conclusion" tokens, while the frequency of "exploration/alternative" tokens remains almost unchanged.

## Highlights & Insights

- The theoretical analysis is the central contribution: two theorems precisely characterize two failure modes of sequence-level length rewards, not only explaining why existing methods are suboptimal but also directly guiding the design of decoupled rewards. This research paradigm—first proving failure theoretically, then designing targeted solutions—is worth emulating.
- The concept of the NRP is both concise and profound: "the shortest prefix after which the correct answer can first be derived" operationalizes the vague notion of "overthinking" into actionable token-level labels.

## Limitations & Future Work

- The quality of the NRP detector directly affects the method's performance; detection errors may cause necessary reasoning steps to be penalized.
- The current chunk segmentation relies on predefined delimiters (e.g., newlines); finer-grained semantic segmentation may yield further improvements.
- Experiments cover only mathematics, programming, and scientific reasoning; generalization to soft tasks such as natural language reasoning has not been validated.

## Related Work & Insights

- **vs. ThinkPrune**: Although ThinkPrune reduces a comparable number of tokens, part of the reduction comes from necessary reasoning tokens (low PNRP score), leading to performance degradation; DeCS precisely reduces only the non-NRP portion.
- **vs. LC-R1**: LC-R1 retains ~10% redundancy; DeCS achieves further compression through decoupled rewards.
- **vs. GRPO + length penalty**: Lemma 2 theoretically proves that GRPO with a length penalty inevitably degrades high-entropy tokens; DeCS protects these tokens via the NRP.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The theoretical analysis is rigorous and directly informs method design; the NRP concept and decoupled reward scheme are strongly original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 7 benchmarks, 2 model scales, backbone generalization, and 5 research question analyses; extremely comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Theoretical derivations are rigorous, analysis is thorough, and visualizations are rich.
- **Value**: ⭐⭐⭐⭐⭐ — Addresses the core efficiency challenge of reasoning LLMs; the practical value of achieving over 50% compression without performance loss is exceptionally high.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Ditch the Denoiser: Emergence of Noise Robustness in Self-Supervised Learning from Data Curriculum](../../NeurIPS2025/medical_imaging/ditch_the_denoiser_emergence_of_noise_robustness_in_self-supervised_learning_fro.md)
- [\[CVPR 2026\] CURE: Curriculum-guided Multi-task Training for Reliable Anatomy Grounded Report Generation](../../CVPR2026/medical_imaging/cure_curriculum-guided_multi-task_training_for_reliable_anatomy_grounded_report_.md)
- [\[ACL 2026\] From Answers to Arguments: Toward Trustworthy Clinical Diagnostic Reasoning with Toulmin-Guided Curriculum Goal-Conditioned Learning](../../ACL2026/medical_imaging/from_answers_to_arguments_toward_trustworthy_clinical_diagnostic_reasoning_with_.md)
- [\[AAAI 2026\] FaNe: Towards Fine-Grained Cross-Modal Contrast with False-Negative Reduction and Text-Conditioned Sparse Attention](../../AAAI2026/medical_imaging/fane_towards_fine-grained_cross-modal_contrast_with_false-negative_reduction_and.md)
- [\[ICLR 2026\] Distributional Consistency Loss: Beyond Pointwise Data Terms in Inverse Problems](distributional_consistency_loss_beyond_pointwise_data_terms_in_inverse_problems.md)

</div>

<!-- RELATED:END -->
