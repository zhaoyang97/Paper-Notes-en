---
title: >-
  [Paper Note] DipLLM: Fine-Tuning LLM for Strategic Decision-Making in Diplomacy
description: >-
  [ICML 2025][LLM Pretraining][Diplomacy] This paper proposes DipLLM, which decomposes the exponential combinatorial action space of Board Game Diplomacy into unit-level decision sequences through an autoregressive factorization framework, and fine-tunes an LLM to learn equilibrium strategies, outperforming Cicero using only 1.5% of its training data.
tags:
  - "ICML 2025"
  - "LLM Pretraining"
  - "Diplomacy"
  - "LLM agent"
  - "fine-tuning"
  - "autoregressive factorization"
  - "Nash equilibrium"
date: 2026-05-08
content_hash: 883331045d0e7bc2
---

# DipLLM: Fine-Tuning LLM for Strategic Decision-Making in Diplomacy

**Conference**: ICML 2025  
**arXiv**: [2506.09655](https://arxiv.org/abs/2506.09655)  
**Code**: None  
**Area**: LLM Pre-training  
**Keywords**: Diplomacy, LLM agent, fine-tuning, autoregressive factorization, Nash equilibrium

## TL;DR
This paper proposes DipLLM, which decomposes the exponential combinatorial action space of Board Game Diplomacy into unit-level decision sequences through an autoregressive factorization framework, and fine-tunes an LLM to learn equilibrium strategies, outperforming Cicero using only 1.5% of its training data.

## Background & Motivation
**Background**: The action space of Diplomacy can reach up to $10^{64}$, and traditional methods rely on equilibrium search to generate large amounts of game data.

**Limitations of Prior Work**: Cicero's CoShar-piKL requires 448 GPUs for game simulation, incurring immense computational overhead. LLM prompting methods perform poorly in complex strategic environments.

**Key Challenge**: While LLMs possess strong general reasoning capabilities, making direct decisions is nearly impossible when facing $26^{34}$ action combinations.

**Goal**: Can LLMs be fine-tuned to learn equilibrium strategies with a small amount of data?

**Key Insight**: Autoregressive factorization + weighted SFT based on unit Q-value.

**Core Idea**: Autoregressive factorization aligns the next-token prediction of LLMs with unit-by-unit decisions.

## Method

### Overall Architecture
The TextDiplomacy module converts the board state into text, and the LLM sequentially generates actions for each unit to form a joint strategy.

### Key Designs
1. **Autoregressive Factorization**: $\boldsymbol{\pi}_i(a_i^{1:D}|s) = \prod_{d=1}^{D} \pi_i^d(a_i^d | s, a_i^{1:d-1})$, where each step only selects from approximately 26 actions.

2. **Equilibrium Strategy Objective**: Defines the unit Q-value $Q_i^d$ and proves that the decomposed joint strategy is equivalent to piKL-Hedge (Theorem 1), converging to an approximate Nash equilibrium in two-player zero-sum games (Theorem 2).

3. **Fine-Tuning Loss**: $\max_{\pi_\phi} \mathbb{E}[\log \pi_\phi(a_i^d|s,a_i^{1:d-1}) \cdot \exp\{Q_i^d\}]$, which consists of an SFT term and equilibrium weights.

### Loss & Training
LLaMA 3 8B + LoRA ($\alpha=32$, rank=16), AdamW lr=2e-4, 5 epochs, with only about 500 game sessions of data.

## Key Experimental Results

### Main Results (1v6 Competition)

| Agent | SoS Score ↑ | Win Rate ↑ | Survived ↑ | Defeated ↓ |
|-------|------------|-----------|-----------|-----------|
| **DipLLM** | **23.0%** | **22.3%** | **50.3%** | **27.4%** |
| Cicero | 20.8% | 20.5% | 50.1% | 29.4% |
| DNVI | 6.6% | 4.3% | 31.1% | 64.6% |
| DipNet | 4.2% | 2.1% | 24.3% | 73.6% |

### Ablation Study

| Configuration | SoS | Win | Defeated | Description |
|------|-----|-----|---------|------|
| AF + Fine-tune | **29.4%** | **25.2%** | **29.0%** | Full |
| AF only | 9.9% | 6.7% | 53.3% | No equilibrium learning |
| FT only (No AF) | 0.8% | 0.0% | 80.8% | Action space too large |
| Baseline | 0.2% | 0.0% | 95.7% | No AF, No FT |

### Key Findings
- Both AF and FT are indispensable; the inference efficiency of DipLLM is 5-10 times that of Cicero.
- Fine-tuning with 100 games of data outperforms DipNet, and fine-tuning with 500 games achieves a 6.7% lead.

## Highlights & Insights
- Autoregressive factorization perfectly aligns with the LLM token prediction paradigm.
- Theoretical guarantees provide a solid foundation for the proposed method.
- Outperforming the SOTA with only 1.5% of the data highlights the leverage effect of pre-trained LLM knowledge.

## Limitations & Future Work
- Still relies on external data generation.
- Only tested on the "no-press" version.
- The potential of integrating with online search has not been fully explored.

## Related Work & Insights
- The core idea is similar to the autoregressive Q-function in Q-Transformer.
- The potential of LLMs in game environments is still far from fully exploited.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First study to fine-tune an LLM for Diplomacy equilibrium strategies.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive comparisons, ablations, and case studies.
- Writing Quality: ⭐⭐⭐⭐ Clear framework.
- Value: ⭐⭐⭐⭐⭐ Significant improvement in data efficiency.

---

## Supplementary Reflections

### Relation to Domain Trends
The research direction of this paper is closely related to several major trends in current AI research: model capability evaluation and reliability assurance, parameter-efficient fine-tuning and model compression, as well as AI safety and alignment. From a methodological standpoint, this paper represents an exploration into the deep mechanisms of LLMs, helping drive the research paradigm shift from empirically driven to theoretically driven.

### Specific Suggestions for Future Work
1. Combine the core idea with other modalities (vision, audio, multimodal) to verify the cross-modal universality of the method.
2. Validate the conclusions on larger-scale models (70B+) and newer architectures (such as Mixture-of-Experts).
3. Explore the possibility of combining this approach with reinforcement learning and online learning to achieve dynamic adaptation.
4. Develop automated evaluation and optimization tools to lower the barriers to adopting this method.
5. Consider the intersection with LLM alignment research to explore the coordinated optimization of safety and performance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Data Difficulty and the Generalization--Extrapolation Tradeoff in LLM Fine-Tuning](../../ICML2026/llm_pretraining/data_difficulty_and_the_generalization--extrapolation_tradeoff_in_llm_fine-tunin.md)
- [\[ICLR 2026\] Token-level Data Selection for Safe LLM Fine-tuning](../../ICLR2026/llm_pretraining/token-level_data_selection_for_safe_llm_fine-tuning.md)
- [\[ICLR 2026\] ssToken: Self-modulated and Semantic-aware Token Selection for LLM Fine-tuning](../../ICLR2026/llm_pretraining/sstoken_self-modulated_and_semantic-aware_token_selection_for_llm_fine-tuning.md)
- [\[ACL 2025\] Data Whisperer: Efficient Data Selection for Task-Specific LLM Fine-Tuning via Few-Shot In-Context Learning](../../ACL2025/llm_pretraining/data_whisperer_data_selection.md)
- [\[ICLR 2026\] Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning](../../ICLR2026/llm_pretraining/pre-training_llm_without_learning_rate_decay_enhances_supervised_fine-tuning.md)

</div>

<!-- RELATED:END -->
