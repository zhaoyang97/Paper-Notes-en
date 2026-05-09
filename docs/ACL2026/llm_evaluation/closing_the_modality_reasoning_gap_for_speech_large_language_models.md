---
title: >-
  [Paper Note] Closing the Modality Reasoning Gap for Speech Large Language Models
description: >-
  [ACL 2026][LLM Evaluation][Speech LLM] This paper proposes TARS (Trajectory Alignment for Reasoning in Speech), a reinforcement learning-based framework that aligns speech-conditioned reasoning trajectories with text-conditioned ones via two dense reward signals—representation alignment and behavior alignment. TARS achieves state-of-the-art performance at the 7B scale, with a Modality Recovery Rate (MRR) approaching or exceeding 100%.
tags:
  - ACL 2026
  - LLM Evaluation
  - Speech LLM
  - Modality Reasoning Gap
  - Reinforcement Learning
  - Representation Alignment
  - Trajectory Alignment
date: 2026-05-08
content_hash: 502fbe6283b4c846
---

# Closing the Modality Reasoning Gap for Speech Large Language Models

**Conference**: ACL 2026
**arXiv**: [2601.05543](https://arxiv.org/abs/2601.05543)
**Code**: [https://github.com/AmphionTeam/TARS](https://github.com/AmphionTeam/TARS)
**Area**: LLM Evaluation
**Keywords**: Speech LLM, Modality Reasoning Gap, Reinforcement Learning, Representation Alignment, Trajectory Alignment

## TL;DR

This paper proposes TARS (Trajectory Alignment for Reasoning in Speech), a reinforcement learning-based framework that aligns speech-conditioned reasoning trajectories with text-conditioned ones via two dense reward signals—representation alignment and behavior alignment. TARS achieves state-of-the-art performance at the 7B scale, with a Modality Recovery Rate (MRR) approaching or exceeding 100%.

## Background & Motivation

**Background**: Speech Large Language Models (Speech LLMs) have made remarkable progress, adopting a three-stage architecture of speech encoder + adapter + text LLM to leverage the reasoning capabilities of text LLMs for speech inputs.

**Limitations of Prior Work**: Reasoning performance under speech input is significantly weaker than under text input, a phenomenon termed the "modality reasoning gap." (1) Input-side fusion methods (e.g., training adapters with frozen LLMs) achieve only superficial alignment, as subtle representation differences are amplified across Transformer layers; (2) Output-side supervision methods (e.g., knowledge distillation) enforce token-level matching in an offline manner, but strict matching is an unattainable target given that the speech-conditioned distribution differs from text, and exposure bias is a persistent issue.

**Key Challenge**: The underlying logical reasoning process should be modality-invariant, yet existing methods either align only input representations or enforce output alignment offline, neither of which effectively guides the alignment of reasoning trajectories themselves.

**Goal**: Design an online policy optimization framework that simultaneously aligns internal representations and external behaviors under both speech and text conditions, thereby eliminating the modality reasoning gap.

**Key Insight**: Leveraging the GRPO reinforcement learning framework with text-conditioned reasoning trajectories as a moving reference, the paper designs asymmetric rewards: for speech completions, the model jointly optimizes task accuracy, representation alignment, and behavior alignment; for text completions, only accuracy is optimized.

**Core Idea**: Through online policy exploration and dense alignment rewards, the speech modality co-evolves with continuously improving text reasoning capabilities, avoiding the exposure bias inherent in offline supervision.

## Method

### Overall Architecture

TARS is built upon GRPO and generates both speech-conditioned and text-conditioned completions for each prompt. The text branch is optimized with base rewards and simultaneously serves as the alignment reference for the speech branch. The speech branch receives additional dense rewards for representation and behavior alignment on top of base rewards. As training progresses, the text branch continuously improves, providing an increasingly strong alignment target for the speech branch.

### Key Designs

1. **Representation Alignment Reward**:

    - Function: Reduce layer-wise hidden state drift between speech- and text-conditioned representations.
    - Mechanism: For each Transformer layer $l$, the hidden states $\mathbf{H}^{(l)}$ of reasoning tokens are mean-pooled into a fixed vector $\bar{\mathbf{h}}^{(l)}$, and the average cosine similarity across $L$ layers between speech and text completions is computed: $R_{\text{rep}} = \frac{1}{L}\sum_{l=1}^{L}\text{CosSim}(\bar{\mathbf{h}}_{\text{speech}}^{(l)}, \bar{\mathbf{h}}_{\text{text}}^{(l)})$
    - Design Motivation: Representation drift is a significant source of the modality reasoning gap; layer-wise hidden state alignment provides coarse-grained but dense representation-level feedback.

2. **Behavior Alignment Reward**:

    - Function: Ensure semantic consistency between final outputs under speech and text conditions.
    - Mechanism: An external embedding model (e.g., Qwen3-Embedding-0.6B) is used to compute the semantic cosine similarity between speech completion $y_{\text{speech}}$ and text reference $y_{\text{text}}^*$: $R_{\text{beh}} = \text{CosSim}(\mathcal{E}(y_{\text{speech}}), \mathcal{E}(y_{\text{text}}^*))$
    - Design Motivation: Behavior alignment complements representation alignment by allowing the model to learn diverse effective reasoning trajectories, requiring only that the final semantic behavior be consistent.

3. **Modality-Specific Normalization**:

    - Function: Prevent speech completions from persistently receiving negative advantages due to systematically lower scores.
    - Mechanism: Advantages are computed separately within each modality group: $\hat{A}_{i,m} = r_{i,m} - \mu_m$, where $\mu_m$ is the mean reward within modality $m$.
    - Design Motivation: Under naïve group-level normalization, text completions naturally receive higher rewards, suppressing the learning signal for speech completions.

### Loss & Training

The DAPO loss estimator (a Dr. GRPO variant) is used. The total reward is $R_{\text{total}} = R_{\text{base}} + \alpha \cdot R_{\text{rep}} + \beta \cdot R_{\text{beh}}$ ($\alpha = \beta = 1.0$), where $R_{\text{base}} = R_{\text{acc}} + 0.5 \cdot R_{\text{fmt}}$. Training applies LoRA fine-tuning to all linear layers while freezing the audio encoder and projector.

## Key Experimental Results

### Main Results

**Accuracy (%) of 7B Models on Reasoning Benchmarks**

| Model | MMSU(A) | MMSU(T) | OBQA(A) | OBQA(T) | Avg(A) | MRR |
|-------|---------|---------|---------|---------|--------|-----|
| Qwen2.5-Omni | 61.51 | 67.94 | 81.09 | 84.40 | 71.30 | 91.76% |
| TARS(Qwen2.5-Omni) | **67.96** | 68.54 | **85.71** | 88.57 | **76.84** | **98.89%** |
| Phi-4-MM | 54.81 | 72.15 | 71.65 | 84.62 | 63.23 | 79.59% |
| TARS(Phi-4-MM) | **70.14** | 75.76 | **89.45** | 91.87 | **79.80** | **100.45%** |

### Ablation Study

| Training Strategy | MMSU(A) | OBQA(A) | Avg(A) | MRR |
|-------------------|---------|---------|--------|-----|
| SFT | 60.83 | 81.54 | 71.19 | 89.36% |
| DPO | 59.99 | 79.78 | 69.89 | 89.39% |
| Standard GRPO | 63.73 | 82.86 | 73.30 | 94.04% |
| + Rep. Alignment | 65.91 | 84.40 | 75.16 | 96.43% |
| + Beh. Alignment | 66.20 | 84.84 | 75.52 | 95.57% |
| + Both (TARS) | **67.96** | **85.71** | **76.84** | **98.89%** |

### Key Findings

- TARS raises MRR to 100.45% on Phi-4-MM, meaning speech reasoning performance surpasses that of text.
- Representation alignment and behavior alignment are complementary: each individually contributes approximately 2% improvement, and their combination yields superior results.
- Modality-specific normalization is critical for training stability; naïve normalization suppresses speech learning signals.
- TARS improves not only speech performance but also text performance simultaneously (Qwen2.5-Omni: 76.17→78.56%).
- The end-to-end model outperforms cascaded ASR+LLM systems, indicating that direct speech processing avoids ASR error propagation.

## Highlights & Insights

- The asymmetric reward design is elegant: the text branch continuously improves by optimizing only task accuracy while simultaneously providing an increasingly strong alignment target for the speech branch, forming a co-evolutionary dynamic.
- Even when task accuracy for all speech completions is zero (in challenging reasoning scenarios), alignment rewards still provide effective gradient signals.
- The MRR > 100% result suggests that knowledge acquired through speech processing can in turn enhance text reasoning.

## Limitations & Future Work

- Evaluation is limited to multiple-choice QA benchmarks; effectiveness on free-form generation tasks remains to be verified.
- Mean pooling used in representation alignment may lose positional information; more fine-grained alignment strategies could be more effective.
- Training relies on synthetic speech; robustness to real-world speech (with noise and accents) requires further investigation.

## Related Work & Insights

- **vs. AlignChat/DeSTA**: These methods freeze the LLM and train only the adapter, achieving alignment solely at the input side; TARS aligns the entire reasoning trajectory through RL.
- **vs. Knowledge Distillation**: KD enforces token-level matching offline and is subject to exposure bias; TARS avoids this issue through online exploration.
- **vs. SoundMind-RL**: A concurrent work that also applies RL to speech reasoning training, but relies solely on sparse rule-based rewards and lacks dense alignment signals.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First application of a trajectory-aligned RL framework to eliminate the speech-text reasoning gap.
- Experimental Thoroughness: ⭐⭐⭐⭐ Two base models, multiple baselines, and detailed ablations, though evaluation benchmarks are limited.
- Writing Quality: ⭐⭐⭐⭐⭐ Problem formulation is clear, method intuition is strong, and equations are concise.
- Value: ⭐⭐⭐⭐⭐ The MRR > 100% result is a milestone finding that opens a new direction for multimodal reasoning alignment.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] MTR-DuplexBench: Towards a Comprehensive Evaluation of Multi-Round Conversations for Full-Duplex Speech Language Models](mtr-duplexbench_towards_a_comprehensive_evaluation_of_multi-round_conversations_.md)
- [\[ACL 2026\] AnchorMem: Anchored Facts with Associative Contexts for Building Memory in Large Language Models](anchormem_anchored_facts_with_associative_contexts_for_building_memory_in_large_.md)
- [\[ACL 2026\] Attribution, Citation, and Quotation: A Survey of Evidence-based Text Generation with Large Language Models](attribution_citation_and_quotation_a_survey_of_evidence-based_text_generation_wi.md)
- [\[ACL 2026\] ReTraceQA: Evaluating Reasoning Traces of Small Language Models in Commonsense Question Answering](retraceqa_evaluating_reasoning_traces_of_small_language_models_in_commonsense_qu.md)
- [\[ACL 2026\] E2EDev: Benchmarking Large Language Models in End-to-End Software Development Task](e2edev_benchmarking_large_language_models_in_end-to-end_software_development_tas.md)

<!-- RELATED:END -->
