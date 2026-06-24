---
title: >-
  [Paper Note] Registering Source Tokens to Target Language Spaces in Multilingual Neural Machine Translation
description: >-
  [ACL 2025][Multilingual & Machine Translation][Multilingual Machine Translation] The Registering method is proposed: by inserting a set of target language markers (registers) between source and target language tokens and modifying the attention mask to make target generation rely solely on the activation of registers, the off-target problem in multilingual translation is thoroughly resolved, enabling the small-scale MITRE-913M model to outperform NLLB-3.3B.
tags:
  - "ACL 2025"
  - "Multilingual & Machine Translation"
  - "Multilingual Machine Translation"
  - "Registering Mechanism"
  - "Attention Mask"
  - "Off-target Problem"
  - "Decoder-only"
date: 2026-05-08
content_hash: afc9812031ca576c
---

# Registering Source Tokens to Target Language Spaces in Multilingual Neural Machine Translation

**Conference**: ACL 2025  
**arXiv**: [2501.02979](https://arxiv.org/abs/2501.02979)  
**Code**: [GitHub](https://github.com/zhiqu22/mitre)  
**Area**: Multilingual Translation  
**Keywords**: Multilingual Machine Translation, Registering Mechanism, Attention Mask, Off-target Problem, Decoder-only

## TL;DR

The Registering method is proposed: by inserting a set of target language markers (registers) between source and target language tokens and modifying the attention mask to make target generation rely solely on the activation of registers, the off-target problem in multilingual translation is thoroughly resolved, enabling the small-scale MITRE-913M model to outperform NLLB-3.3B.

## Background & Motivation

Multilingual Neural Machine Translation (MNMT) aims to support arbitrary translation among multiple languages. While traditional specialized MNMT models feature smaller parameter sizes and support zero-shot translation, their performance has consistently lagged behind Large Language Models (LLMs).

The core bottleneck is the **off-target problem**: the translation output fails to target the correct language. This occurs because, although a language tag $l_y$ at the beginning of the input sequence indicates the target language, its influence is diluted by a large number of source language tokens in the attention mechanism, preventing the generation process from strictly adhering to the target language.

Existing solutions have their respective limitations:
- **LAVS**: Adds language-specific vocabulary, which is costly and hinders knowledge sharing.
- **CL**: Aligns cross-lingual semantic representations, alleviating the issue indirectly.
- **LCS**: Uses target language embeddings to bias encoder representations, restricted to Encoder-Decoder architectures.
- **TDO**: Dictates Decoder-only models into two stages to enhance translation instructions.

## Method

### Overall Architecture

The core idea of Registering is to insert a set of **register tokens** (registers) between the source and target sequences. Through a carefully designed attention mask, the generation of target tokens is **completely isolated from the source tokens**, forcing them to access source language information solely through the registers.

Given the source sequence $\boldsymbol{x}' = l_y, x_1, \ldots, x_I$, a register sequence $\boldsymbol{r} = r_1, \ldots, r_{I+1}$ (equal in length to the source sequence) is inserted, modifying the generation process to:

$$y_j = \text{decoder}(\boldsymbol{x}', \boldsymbol{r}, \boldsymbol{y}_{<j})$$

### Key Designs I: Register Initialization and Attention Mask

**Register Initialization**: All register tokens are initialized with the embedding of the target language tag $l_y$ (since each register should ultimately reside within the target language space).

**Attention Mask Rules** (modified based on the prefix Dec-only scheme):
1. **$\boldsymbol{r}$ attends to $\boldsymbol{x}'$**: Registers can see all source tokens.
2. **Bidirectional attention within $\boldsymbol{r}$**: Registers can interact with each other.
3. **$y_j$ only attends to $\boldsymbol{r}$ and $\boldsymbol{y}_{<j}$**: Target tokens cannot see source tokens, acquiring information only indirectly via registers.

The effect of this design: each $r_i$ captures the semantics of its position-aligned source token $x_i$ and "translates" it into the target language space. Generation is strictly constrained within the target language space.

### Key Designs II: No Extra Parameters

Registering is implemented purely by modifying the attention mask, **introducing zero extra parameters**. This allows it to be seamlessly applied to any Dec-only model, achieving superior parameter efficiency compared to Enc-dec.

### Loss & Training

Standard cross-entropy loss is used, computing the loss only on target tokens:

$$\mathcal{L}_{ce} = -\sum_{\boldsymbol{x}', \boldsymbol{y} \in \mathbb{C}} \sum_{j=1}^{J} \log p(y_j \mid \boldsymbol{x}', \boldsymbol{r}, \boldsymbol{y}_{<j})$$

The pretrained MITRE model is trained on 9.3B sentence pairs, covering 24 languages and 194 translation directions. The vocabulary is trained via SentencePiece with a size of 160K. Data collection employs a Bridge Language strategy, selecting two high-resource languages from each language family as bridge languages.

## Key Experimental Results

### Main Results 1: EC-40 Benchmark (Zero-Shot Translation)

spBLEU results across 1640 translation directions on the EC-40 benchmark (excerpt from Table 1, 24-layer configuration):

| Model | Params | sup. | zero. | avg. | off-target(%) |
|------|--------|------|-------|------|---------------|
| Enc-dec vanilla | 418M | 30.28 | 8.64 | 9.69 | 26.69 |
| + CL | 418M | 30.54 | 10.79 | 11.76 | 19.99 |
| + LAVS | 430M | 30.20 | 10.03 | 11.01 | 21.73 |
| Dec-only vanilla | 368M | 29.97 | 9.84 | 10.82 | 19.01 |
| + TDO | 368M | 30.23 | 10.40 | 11.37 | 23.14 |
| **+ Ours** | **368M** | 29.88 | **12.26** | **13.12** | **3.65** |

The off-target rate decreases from 26.69% to **3.65%**, nearly resolving the off-target problem entirely. The spBLEU improvement in zero-shot translation significantly outperforms all baselines.

### Main Results 2: MITRE Pretrained Model vs LLMs

Comparison of spBLEU results after large-scale pretraining (excerpt from Table 3):

| Model | Params | avg. spBLEU |
|------|--------|-------------|
| M2M-1.2B | 1.2B | 24.69 |
| NLLB-3.3B | 3.3B | 30.01 |
| GPT-3.5 Turbo | - | 28.66 |
| GPT-4o mini | - | 31.09 |
| **MITRE-913M** | **913M** | **31.15** |

With only 913M parameters, MITRE-913M outperforms NLLB-3.3B (+1.14) and GPT-3.5 Turbo, performing on par with GPT-4o mini.

### Ablation Study

**Architecture Selection**: Dec-only + Registering outperforms Enc-dec due to higher parameter efficiency and because registers naturally adapt to the attention mechanism of Dec-only architectures.

**Scalability**: When scaling the model from 12 layers to 24 layers, other methods show diminishing gains (dropping from 3.88 to 2.15), whereas Registering maintains consistent gains (5.19→3.62), demonstrating superior scalability.

**Fine-tuning Adaptability**: Pretrained MITRE exhibits robust adaptability in both LoRA and full-parameter fine-tuning, significantly enhancing translation quality in specific directions using only the Flores dev set (997 sentence pairs per direction).

### Key Findings

1. The root cause of the off-target problem is the dilution of the language tag by source tokens in the attention mechanism.
2. The activation of registers indeed reflects the semantics of source tokens within the target language space (validated through attention and representation distribution analyses).
3. Simply adding language-specific parameters (LAVS) is not a cost-efficient strategy.
4. Dec-only architectures are superior to Enc-dec in terms of parameter efficiency.

## Highlights & Insights

- **Conceptual Elegance**: Registering can be conceptualized as a representation-level chain-of-thought — "rethinking" the source tokens from the perspective of the target language.
- **Zero Extra Parameters**: Implemented solely by modifying the attention mask, keeping the method highly elegant.
- **Impressive Empirical Results**: The 913M parameter model reaches the level of GPT-4o mini while driving the off-target rate down to near zero.
- Methodologically, it integrates the concepts of gisting (mask-based information compression) and prefix-tuning.

## Limitations & Future Work

- The length of registers is fixed to the source sequence length, which may increase computational overhead for extremely long sentences.
- Registers are initialized uniformly with the language tag; more intelligent initialization strategies remain unexplored.
- Pretraining is restricted to the NLLB dataset, meaning data quality and coverage might limit performance on certain language pairs.
- The sequence length doubles during inference (source + registers + target), which impacts latency.

## Related Work & Insights

- **Gisting** (Mu et al., 2023): Compresses information into artificial tokens using modified attention masks, serving as the methodological inspiration for Registering.
- **NLLB** (NLLB Team, 2022): The current benchmark for MNMT and the primary focus of comparison for MITRE.
- **TDO** (Qu et al., 2024b): Prior work that enhances translation instructions via a two-stage process; Registering is its direct evolution.

## Rating

- **Novelty**: 4/5 — Cleverly transfers gisting ideas to MNMT, offering a clear and original concept.
- **Technical Depth**: 4/5 — Simple yet powerful attention mask design, thoroughly validated with large-scale pretraining.
- **Experimental Thoroughness**: 5/5 — Covers EC-40 + large-scale pretraining + fine-tuning + multi-metrics, making it extremely comprehensive.
- **Value**: 5/5 — Open-sourced model, directly usable, with high industrial value.

## Related Work & Insights

## Insights & Connections

## Rating

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Unveiling the Power of Source: Source-based Minimum Bayes Risk Decoding for Neural Machine Translation](unveiling_the_power_of_source_source-based_minimum_bayes_risk_decoding_for_neura.md)
- [\[ACL 2025\] Multi-perspective Alignment for Increasing Naturalness in Neural Machine Translation](multi-perspective_alignment_for_increasing_naturalness_in_neural_machine_transla.md)
- [\[ACL 2025\] Memorization Inheritance in Sequence-Level Knowledge Distillation for Neural Machine Translation](memorization_inheritance_seqkd.md)
- [\[ACL 2025\] THOR-MoE: Hierarchical Task-Guided and Context-Responsive Routing for Neural Machine Translation](thor-moe_hierarchical_task-guided_and_context-responsive_routing_for_neural_mach.md)
- [\[ACL 2026\] Mitigating Catastrophic Forgetting in Target Language Adaptation of LLMs via Source-Shielded Updates](../../ACL2026/multilingual_mt/mitigating_catastrophic_forgetting_in_target_language_adaptation_of_llms_via_sou.md)

</div>

<!-- RELATED:END -->
