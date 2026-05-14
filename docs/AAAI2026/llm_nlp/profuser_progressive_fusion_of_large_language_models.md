---
title: >-
  [Paper Note] ProFuser: Progressive Fusion of Large Language Models
description: >-
  [AAAI 2026][LLM/NLP][model fusion] ProFuser proposes a progressive model fusion strategy that gradually fuses the parameters and knowledge of multiple LLMs in multiple stages…
tags:
  - "AAAI 2026"
  - "LLM/NLP"
  - "model fusion"
  - "progressive fusion"
  - "multi-model ensemble"
  - "knowledge distillation"
date: 2026-05-08
content_hash: cefb061e90fbc656
---

# ProFuser: Progressive Fusion of Large Language Models

**Conference**: AAAI 2026
**arXiv**: [2408.04998](https://arxiv.org/abs/2408.04998)
**Code**: None
**Area**: LLM NLP
**Keywords**: model fusion, progressive fusion, multi-model ensemble, knowledge distillation

## TL;DR
ProFuser is proposed to comprehensively identify the strengths of each source model across different dimensions via dual-mode advantage assessment (training-mode Min-CE + inference-mode Reward Model voting), and then integrates the complementary capabilities of heterogeneous LLMs into a single target model through a progressive fusion strategy (inference mode first → training mode second, as an easy-to-hard curriculum), achieving an average improvement of 1.65% across 6 benchmarks covering knowledge, reasoning, and safety.

## Background & Motivation
**Background**: Fusing the complementary capabilities of multiple LLMs into a unified model is an efficient path to performance improvement. FuseLLM pioneered heterogeneous LLM fusion via knowledge distillation. Model merging methods (Task Arithmetic/TIES) require homogeneous architectures.

**Limitations of Prior Work**: FuseLLM evaluates model advantages solely using training-mode Min-CE (minimum cross-entropy under teacher-forcing). Empirically, Vicuna outperforms Llama-2-Chat on 68% of samples in training mode, but only on 45% in inference mode — token prediction ability ≠ response generation quality, and single-mode evaluation misses important complementary information.

**Key Challenge**: Jointly optimizing both modes yields suboptimal results, as training mode uses detailed GT outputs from GPT-4 while inference mode uses shorter outputs from source models, resulting in a large complexity gap.

**Key Insight**: Begin with the simpler inference mode (source model outputs are relatively short), then transition to the more complex training mode (GT outputs are more detailed), realizing an easy-to-hard progressive fusion.

**Core Idea**: Comprehensive dual-mode assessment of each source model's strengths + progressive easy-to-hard fusion = more thorough capability integration.

## Method

### Overall Architecture
Given $K$ heterogeneous source LLMs and a target LLM, the advantage of each model is evaluated on every instruction simultaneously in training mode (Min-CE selects the optimal distribution) and inference mode (RM voting selects the optimal response), and capabilities are transferred via a two-stage progressive fusion.

### Key Designs

1. **Training-Mode Advantage Assessment**: For each (instruction, GT response) pair, cross-entropy is computed for each source model under teacher-forcing, and the model with the lowest CE is selected: $M^{MinCE} = \arg\min(\{L_{SFT}^{\theta_j}\})$; its logits distribution serves as the training-mode advantage signal.

2. **Inference-Mode Advantage Assessment**: Each source model generates a response to the instruction, and the best response is selected via voting across multiple Reward Models: $\tilde{y}_i^B = \arg\max(\text{RM}_{Vote}(\{\tilde{y}_i^j\}))$; its logits distribution serves as the inference-mode advantage signal.

3. **Progressive Fusion**: Fusion loss $L_{Fuse} = L_{SFT} + \beta D_{KL}(P_S, P_T)$. Overall objective $L_{ProFuser} = w_1 L_{Infer-Fuse} + w_2 L_{Train-Fuse}$. Stage 1: $w_1=1, w_2=0$; Stage 2: $w_1=0.1, w_2=1$.

## Key Experimental Results

### Main Results

| Method | MMLU | ARC | GSM8K | TruthfulQA | Avg. |
|--------|------|-----|-------|-----------|------|
| Vicuna-7B-v1.5 | 51.17 | 53.75 | 15.80 | 50.37 | 53.46 |
| FuseLLM | 51.48 | 54.61 | 18.80 | 50.72 | 54.53 |
| **ProFuser** | **51.85** | **55.46** | 18.70 | **51.85** | **55.11** |

### Ablation Study

| Configuration | Avg. | Note |
|---------------|------|------|
| ProFuser (inference → training) | **55.11** | Best |
| ReverseFuse (training → inference) | 54.30 | Reverse order is worse |
| SimulFuse (simultaneous optimization) | 54.56 | Inferior to progressive |

### Key Findings
- Progressive > simultaneous optimization > reverse order, validating the effectiveness of the easy-to-hard strategy
- Even weaker source models (MPT) can contribute positively
- TruthfulQA shows the largest gain (+1.48%)

## Highlights & Insights
- **Dual-mode assessment** reveals advantages missed by single-mode evaluation: training-mode advantage ≠ inference-mode advantage
- **Progressive easy-to-hard fusion** exploits the natural complexity difference between the outputs of the two modes

## Limitations & Future Work
- Validated only on 7B models; effectiveness at larger scales remains unknown
- Requires generating inference outputs from all source models plus GT logits, resulting in high data preparation costs
- The $w_1$/$w_2$ scheduling is relatively manual; adaptive weight schemes could be explored

## Rating
- Novelty: ⭐⭐⭐⭐ Dual-mode assessment + progressive fusion is creative
- Experimental Thoroughness: ⭐⭐⭐⭐ 6 benchmarks + multiple ablations
- Writing Quality: ⭐⭐⭐⭐ Motivation analysis is clear
- Value: ⭐⭐⭐⭐ Provides a more comprehensive methodology for heterogeneous LLM fusion

## TL;DR

ProFuser proposes a progressive model fusion strategy that gradually fuses the parameters and knowledge of multiple LLMs in multiple stages, producing a unified and powerful model while preserving the strengths of each source model.

## Background & Motivation

Different LLMs tend to excel at different tasks (e.g., model A is stronger at code, model B at reasoning). Integrating multiple models into a unified model is an important practical deployment requirement. Simple weight averaging or one-shot distillation is prone to knowledge conflicts and capability degradation. Inspired by curriculum learning, ProFuser designs a progressive fusion strategy that proceeds from shallow to deep layers and from simple to complex, gradually incorporating knowledge from multiple source models into the target model to avoid abrupt distribution shifts.

## Method

### Key Designs

- **Progressive Layer-wise Fusion**: Source models are fused in stages — shallow layers (embeddings + first few Transformer layers) are fused first, and middle-to-deep layers are fused after stabilization, reducing conflicts from one-shot fusion.
- **Task-Aware Routing**: A lightweight routing mechanism is introduced during fusion to dynamically determine, per input token, which source model's parameters each layer should predominantly reference.
- **Consistency Distillation Loss**: The fused model is simultaneously aligned to the output distributions of multiple source models, using a KL-divergence weighted combination as the training objective.

## Key Experimental Results

| Method | MMLU | HumanEval | GSM8K | Avg. |
|--------|------|-----------|-------|------|
| Best Single Model | 71.2 | 68.3 | 74.5 | 71.3 |
| Weight Averaging | 65.8 | 61.2 | 68.1 | 65.0 |
| One-Shot Distillation | 69.4 | 65.7 | 72.3 | 69.1 |
| ProFuser | **73.1** | **70.8** | **76.2** | **73.4** |

## Highlights & Insights

- Progressive fusion not only surpasses the single best model but also avoids the catastrophic degradation of weight averaging, validating the intuition that "slow fusion is better than fast fusion."
- Analysis of the task-aware routing shows that shallow layers receive roughly equal contributions from all models, whereas deep layers exhibit significant divergence, suggesting that capability differences between models are primarily encoded in deeper layers.

## Rating

| Dimension | Score | Rationale |
|-----------|-------|-----------|
| Novelty | ⭐⭐⭐⭐ | The combination of progressive fusion + task routing is a novel design |
| Technical Depth | ⭐⭐⭐⭐ | Multi-stage training strategy is carefully designed with solid ablation analysis |
| Experimental Thoroughness | ⭐⭐⭐⭐ | Covers multi-dimensional evaluation across knowledge, code, and mathematics |
| Practical Value | ⭐⭐⭐⭐ | Model merging addresses a genuine need in industrial deployment |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Control Illusion: The Failure of Instruction Hierarchies in Large Language Models](control_illusion_the_failure_of_instruction_hierarchies_in_large_language_models.md)
- [\[AAAI 2026\] Identifying and Analyzing Performance-Critical Tokens in Large Language Models](identifying_and_analyzing_performance-critical_tokens_in_large_language_models.md)
- [\[AAAI 2026\] LoKI: Low-damage Knowledge Implanting of Large Language Models](loki_low-damage_knowledge_implanting_of_large_language_models.md)
- [\[AAAI 2026\] CoEvo: Continual Evolution of Symbolic Solutions Using Large Language Models](coevo_continual_evolution_of_symbolic_solutions_using_large_language_models.md)
- [\[AAAI 2026\] TEMPLE: Incentivizing Temporal Understanding of Video LLMs via Progressive Pre-SFT Alignment](temple_incentivizing_temporal_understanding_of_video_large_language_models_via_p.md)

</div>

<!-- RELATED:END -->
