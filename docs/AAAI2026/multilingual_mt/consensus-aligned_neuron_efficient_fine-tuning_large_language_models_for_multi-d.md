---
title: >-
  [Paper Note] Consensus-Aligned Neuron Efficient Fine-Tuning Large Language Models for Multi-Domain Machine Translation
description: >-
  [AAAI 2026][Multi-domain machine translation] This paper proposes CANEFT, which uses mutual information (MI) to identify consensus-aligned neurons in LLMs that are consistently important across domains, and fine-tunes only these neurons to achieve efficient adaptation for multi-domain machine translation (MDMT). CANEFT outperforms PEFT baselines such as LoRA across 3 LLMs and 10 translation domains without introducing any additional parameters.
tags:
  - AAAI 2026
  - Multi-domain machine translation
  - neuron selection
  - mutual information
  - parameter-efficient fine-tuning
  - LLM
date: 2026-05-08
content_hash: 30d1a7d47119bd11
---

# Consensus-Aligned Neuron Efficient Fine-Tuning Large Language Models for Multi-Domain Machine Translation

**Conference**: AAAI 2026
**arXiv**: [2602.05694](https://arxiv.org/abs/2602.05694)
**Code**: [GitHub](https://github.com/fortunatekiss/CANEFT)
**Area**: Multilingual Translation
**Keywords**: Multi-domain machine translation, neuron selection, mutual information, parameter-efficient fine-tuning, LLM

## TL;DR
This paper proposes CANEFT, which uses mutual information (MI) to identify consensus-aligned neurons in LLMs that are consistently important across domains, and fine-tunes only these neurons to achieve efficient adaptation for multi-domain machine translation (MDMT). CANEFT outperforms PEFT baselines such as LoRA across 3 LLMs and 10 translation domains without introducing any additional parameters.

## Background & Motivation

### State of the Field

Multi-domain machine translation (MDMT) requires a single model to cover multiple domains such as legal, medical, and subtitles. LLMs exhibit strong general-purpose translation capabilities, but domain adaptation remains a challenge.

### Limitations of Prior Work

LoRA suffers from parameter interference during multi-domain fine-tuning — adapting to one domain degrades performance on others.

### Root Cause

Adapter-based methods introduce separate modules for each domain, leading to parameter and memory costs that grow linearly with the number of domains.

### Starting Point

In-context learning (ICL) relies on high-quality in-domain examples and performs poorly for MDMT.

**Root Cause**: Can a PEFT method be designed that introduces no additional parameters and avoids inter-domain interference?

**Starting Point**: Inspired by neuroscience — consensus-based communication enhances neural alignment among group members. By analogy, neurons in LLMs that consistently encode translation knowledge across domains should exist. The goal is to identify and fine-tune only these neurons.

**Core Idea**: Use mutual information to measure the association between neuron importance and domain labels, select neurons with high MI across all domains as "consensus-aligned neurons," and fine-tune only these neurons.

## Method

### Overall Architecture
A three-step pipeline: (1) identify task-relevant neurons via activation-gradient analysis; (2) select consensus-aligned neurons across domains using mutual information; (3) fine-tune only the parameters of these neurons.

### Key Designs

1. **Task-Relevant Neuron Identification**:

    - Function: Identify neurons in FFN layers relevant to the translation task.
    - Mechanism: Neuron importance $I_{l,j}^{(d)} = \mathbb{E}[|A_{l,j}^{(d)} \cdot G_{l,j}^{(d)}|]$ (absolute value of activation × gradient).
    - Theoretical basis: A first-order Taylor expansion proves this metric approximates the change in loss when the neuron is removed.

2. **Consensus Neuron Selection via Mutual Information**:

    - Function: Select a subset of task-relevant neurons that are consistently important across all domains.
    - Mechanism: Importance scores are discretized, then mutual information $MI_{l,j}$ between each neuron and the domain label is computed.
    - Selection criterion: $\mathcal{N}_{MDCA} = \{(l,j) | \min MI_{l,j} \geq \gamma\}$ — neurons must reach the MI threshold across all domains.
    - Design Motivation: Neurons important only in certain domains are excluded to avoid domain bias; only cross-domain consistently important neurons are selected.

3. **Neuron-Efficient Fine-Tuning**:

    - A binary mask $M$ is constructed to allow gradient updates only for parameters corresponding to consensus neurons.
    - $\nabla W_m \leftarrow \nabla W_m \odot M$
    - Covers the up, down, and gate projection matrices of FFN layers.
    - No additional parameters are introduced — lighter than LoRA.

### Loss & Training
Standard translation cross-entropy loss with gradient masking. Validated on LLaMA2-7B-Chat, Qwen2.5-7B, and Gemma2-9B.

## Key Experimental Results

### Main Results (De→En, 5 Domains)

| Method | Trainable Params | IT BLEU | Law BLEU | Med BLEU | Avg. |
|--------|-----------------|---------|----------|----------|------|
| Base (zero-shot) | 0 | 30.0 | 44.1 | 35.5 | 32.8 |
| Full FT | 7B | 47.9 | 47.1 | 35.3 | 38.9 |
| LoRA | ~20M | 35.8 | 49.9 | 33.5 | 36.2 |
| **CANEFT** | Minimal | **Best** | **Best** | **Best** | **+1.3 BLEU** |

CANEFT outperforms both Full FT and LoRA on both seen and unseen domains.

### Key Findings
- Parameter interference in LoRA is confirmed — training on the Medical domain leads to a notable drop on the IT domain.
- Fine-tuning only consensus neurons produces no inter-domain interference and generalizes to unseen domains.
- MI-based selection outperforms pure gradient- or activation-based selection, as the latter tends to select domain-specific neurons.
- Consistent results across 3 LLMs indicate that the method is robust to model architecture.

## Highlights & Insights
- The concept of **"consensus alignment"** draws an elegant analogy from neuroscience — group consensus communication maps to cross-domain consensus neurons.
- **Parameter-free PEFT** — lighter than LoRA/Adapter and theoretically less prone to interference.
- **MI selection criterion design** — requiring neurons to be important across *all* domains rather than *any* domain effectively prevents domain bias.

## Limitations & Future Work
- Neuron identification requires forward and backward passes over data from all domains, resulting in non-trivial initialization costs.
- The threshold $\gamma$ requires tuning.
- Validation is limited to machine translation; applicability to other multi-task scenarios remains unexplored.
- Only FFN neurons are fine-tuned; attention layers are not addressed.

## Related Work & Insights
- **vs. LoRA**: LoRA suffers from parameter interference; CANEFT avoids this via neuron selection and introduces no additional parameters.
- **vs. Adapter**: Adapter-based methods introduce per-domain modules with costs growing with domain count; CANEFT uses a single unified set of consensus neurons.
- **vs. Language-Specific Neurons**: Methods such as LAPE identify language-specific neurons, whereas CANEFT targets cross-domain consensus neurons — the two approaches have opposing objectives.

## Rating
- Novelty: ⭐⭐⭐⭐ MI-based consensus neuron selection represents a novel PEFT paradigm.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 3 LLMs, 10 domains, De→En and Zh→En, and generalization to unseen domains.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation and complete theoretical derivation.
- Value: ⭐⭐⭐⭐ A practical, parameter-free PEFT solution.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Focusing on Language: Revealing and Exploiting Language Attention Heads in Multilingual Large Language Models](focusing_on_language_revealing_and_exploiting_language_attention_heads_in_multil.md)
- [\[AAAI 2026\] Mitigating Content Effects on Reasoning in Language Models through Fine-Grained Activation Steering](mitigating_content_effects_on_reasoning_in_language_models_through_fine-grained_.md)
- [\[NeurIPS 2025\] Exploring the Translation Mechanism of Large Language Models](../../NeurIPS2025/multilingual_mt/exploring_the_translation_mechanism_of_large_language_models.md)
- [\[CVPR 2026\] MMTIT-Bench: A Multilingual and Multi-Scenario Benchmark with Cognition-Perception-Reasoning Guided Text-Image Machine Translation](../../CVPR2026/multilingual_mt/mmtit-bench_a_multilingual_and_multi-scenario_benchmark_with_cognition-perceptio.md)
- [\[AAAI 2026\] Bridging the Multilingual Safety Divide: Efficient, Culturally-Aware Alignment for Global South Languages](bridging_the_multilingual_safety_divide_efficient_culturally-aware_alignment_for.md)

<!-- RELATED:END -->
