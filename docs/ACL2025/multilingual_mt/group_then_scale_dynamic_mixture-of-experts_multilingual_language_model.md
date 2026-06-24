---
title: >-
  [Paper Note] Group then Scale: Dynamic Mixture-of-Experts Multilingual Language Model
description: >-
  [ACL 2025][Multilingual & Machine Translation][multilingual LLM] DMoE is proposed—a method combining parameter deviation-based dynamic language grouping with selective MoE layer expansion. By quantifying language similarity through only 10 steps of fine-tuning, similar languages are grouped to share the same expert. MoE expansion is applied only to layers with large parameter deviations (language-specific layers). It reduces PPL by 11.4% compared to continual pre-training acr…
tags:
  - "ACL 2025"
  - "Multilingual & Machine Translation"
  - "multilingual LLM"
  - "mixture of experts"
  - "curse of multilinguality"
  - "language grouping"
  - "parameter deviation"
date: 2026-05-08
content_hash: b22281c0988336f1
---

# Group then Scale: Dynamic Mixture-of-Experts Multilingual Language Model

**Conference**: ACL 2025  
**arXiv**: [2506.12388](https://arxiv.org/abs/2506.12388)  
**Code**: [GitHub](https://github.com/ZNLP/DMoE)  
**Area**: LLM Efficiency/Multilingual  
**Keywords**: multilingual LLM, mixture of experts, curse of multilinguality, language grouping, parameter deviation

## TL;DR
DMoE is proposed—a method combining parameter deviation-based dynamic language grouping with selective MoE layer expansion. By quantifying language similarity through only 10 steps of fine-tuning, similar languages are grouped to share the same expert. MoE expansion is applied only to layers with large parameter deviations (language-specific layers). It reduces PPL by 11.4% compared to continual pre-training across 18–128 languages, and outperforms X-ELM by 9.6% with 3.6x fewer parameters.

## Background & Motivation

**Background**: The performance of multilingual LLMs on low-and-medium-resource languages lags far behind that on high-resource languages. The core reason is the "curse of multilinguality"—a large number of languages compete for limited model capacity, causing negative transfer among dissimilar languages.

**Limitations of Prior Work**: (a) X-MOD trains an adapter for each language, which requires language identification and results in parameter counts growing linearly with the number of languages; (b) X-ELM trains separate models for each language group and fuses top-$m$ models during inference, which is costly; (c) There lacks a method to precisely quantify language similarity and locate "which layers require more capacity."

**Key Challenge**: How to flexibly expand the LLM capacity for a massive number of languages while promoting positive transfer among similar languages?

**Goal**: (a) Identify which layers are language-specific (requiring expansion); (b) Group languages to maximize intra-group similarity; (c) Support the dynamic adaptation of new languages.

**Key Insight**: Parameter deviation ($\Delta\theta^x$) is utilized to resolve both issues simultaneously: large layer-wise deviation implies that the layer requires MoE expansion; high similarity in deviation between languages implies they should be grouped to share an expert.

**Core Idea**: Fine-tune on each language for 10 steps to obtain $\Delta\theta^x$ -> Group languages based on cosine similarity -> Expand layers with large deviations into MoE (each expert serves one language group) -> Dynamically adapt new languages by copying and fine-tuning the expert of the nearest group.

## Method

### Overall Architecture
A three-step pipeline: (1) Compute parameter deviations for each language -> (2) Group languages + Select layers to expand -> (3) Train the MoE model (fine-tune corresponding experts for each language group).

### Key Designs

1. **Parameter Deviation as Language Representation**

    - **Function**: Characterize the features of each language using $\Delta\theta^x = \theta_{\text{finetuned}} - \theta_{\text{base}}$ (with only 10 steps of fine-tuning).
    - **Mechanism**: Language similarity = $\cos(\Delta\theta^x, \Delta\theta^y)$. It is observed that layers near the input/output exhibit the largest deviations (language-specific), while middle layers show smaller deviations (representing a shared "conceptual space").
    - **Design Motivation**: Better captures representation differences inside the model compared to purely linguistic features like LANG2VEC.

2. **Language Grouping (Maximin Optimization)**

    - **Function**: Evenly partition $K$ languages into $G$ groups to maximize the minimum similarity within each group.
    - **Mechanism**: A greedy algorithm—merging the most similar language pairs at each step and gradually expanding until the groups are full.
    - **Design Motivation**: High intra-group similarity leads to fewer gradient conflicts, making expert sharing more effective.

3. **Dynamic MoE Layer Expansion**

    - **Function**: Expand only the top-$\epsilon$ layers with the largest deviations into MoE layers (rather than all layers).
    - **Mechanism**: Each MoE layer contains $g$ experts (1 per language group). During training, tokens from the same language group route to the same expert. The router is trained with a language group classification loss: $\mathcal{L}_{RC} = -\sum_x \sum_i \log P_i(l|x;\theta)$.
    - **Total Loss**: $\mathcal{L} = \mathcal{L}_{CLM} + \alpha \mathcal{L}_{RC}$

4. **Dynamic Adaptation for New Languages**

    - **Function**: Locate the closest expert for a new language and copy-fine-tune it.
    - **Mechanism**: Input the new language into all experts -> The expert yielding the lowest PPL is the most similar -> Copy this expert and fine-tune it alone (with other parameters frozen).
    - **Design Motivation**: Updating only the new expert avoids affecting learned languages, thereby mitigating catastrophic forgetting.

## Key Experimental Results

### 18-Language PPL (BLOOM-560M)

| Method | Params | High-Resource Avg | Mid-Resource Avg | Low-Resource Avg | Overall Avg |
|------|--------|-----------|-----------|-----------|---------|
| BLOOM (Original) | 560M | — | — | — | 56.9 |
| + Pre-train | 560M | — | — | — | 21.6 |
| X-ELM | 5.03B | — | — | — | 21.5 |
| Branch-Train-Mix | 1.57B | — | — | — | 21.2 |
| **DMoE (6 groups)** | **937M** | — | — | — | **19.5** |

### Ablation: Comparison of Grouping Methods

| Grouping Method | 18-Language Avg PPL |
|---------|---------------|
| Random Grouping | 20.8 |
| LANG2VEC Grouping | 19.8 |
| **Parameter Deviation Grouping (Ours)** | **19.5** |

### 128-Language Scaling

| Method | Avg PPL |
|------|---------|
| + Pre-train | 26.3 |
| **DMoE** | **23.1** |

### Key Findings
- **Expanding only top layers is sufficient**: Layers near the inputs and outputs show the largest deviation. Expanding only these layers yields most of the benefits, while middle layers can be shared without performance degradation.
- **Parameter deviation grouping outperforms linguistic grouping**: The similarity "learned by the model" is more effective than "a priori linguistic knowledge"—for example, Chinese and Japanese are grouped together (sharing the Kanji writing system).
- **DMoE outperforms X-ELM (5.03B params) with only 937M parameters**—achieving 5.4x parameter efficiency.
- **New language adaptation does not harm old languages**: Freezing other parameters while only fine-tuning the new expert results in nearly unchanged PPL on old languages.
- **Scalable from 18 to 128 languages**: The grouping strategy remains effective on large-scale language sets.

## Highlights & Insights
- **"Quantifying language similarity with only 10 fine-tuning steps" is extremely efficient**: It avoids training extra language vectors or calculating gradient matrices; only 10 steps are sufficient to produce meaningful parameter deviation representations.
- **The intuition "large deviation = more capacity required" is highly precise**: This aligns with the discovery that the conceptual space resides in the middle layers, whereas language-specific layers lie at both ends.
- **The combination of language grouping and MoE is highly natural**: Compared to one adapter per language (X-MOD) or one model per language (X-ELM), language group-based MoE offers the optimal balance between parameter efficiency and performance.
- **The dynamic adaptation design is valuable for practical deployment**: Adding new languages does not require retraining the entire model, but only copying and fine-tuning the nearest expert.

## Limitations & Future Work
- **The number of groups $K$ is a hyperparameter**: It must be manually selected, and different models/language sets may require different values of $K$.
- **Routing relies on language classification**: Although explicit language IDs are not required, the router's training is essentially language group classification.
- **Validated only on small models**: Primarily BLOOM-560M and Gemma-2B; no experiments were conducted on models larger than 7B.
- **Fixed language groups**: The group structures cannot be dynamically adjusted (e.g., merged or split) after training.
- **No comparison with the latest MoE routing strategies**: Such as ST-MoE or the load balancing of Switch Transformer.

## Related Work & Insights
- **vs X-MOD (Pfeiffer et al., 2022)**: X-MOD assigns one adapter per language and requires language IDs, whereas DMoE assigns one expert per group with automatic routing.
- **vs X-ELM (Blevins et al., 2024)**: X-ELM trains a full model for each group, whereas DMoE shares most parameters and only expands key layers, achieving 5x+ higher parameter efficiency.
- **vs Branch-Train-Mix (Sukhbaatar et al., 2024)**: BTM employs domain branches, whereas DMoE utilizes language branches + dynamic layer selection.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Quantifying language similarity with parameter deviation + selective MoE expansion; the idea is precise and elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covering 18–128 languages + multiple models + multiple grouping strategy ablations, though the model scales are relatively small.
- Writing Quality: ⭐⭐⭐⭐ Clearly described method, with an intuitive three-step flowchart in Figure 2.
- Value: ⭐⭐⭐⭐⭐ Holds direct practical value for the capacity expansion and language specialization of multilingual LLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Less, but Better: Efficient Multilingual Expansion for LLMs via Layer-wise Mixture-of-Experts](less_but_better_efficient_multilingual_expansion.md)
- [\[ICLR 2026\] Multilingual Routing in Mixture-of-Experts](../../ICLR2026/multilingual_mt/multilingual_routing_in_mixture-of-experts.md)
- [\[ACL 2025\] Towards Global AI Inclusivity: A Large-Scale Multilingual Terminology Dataset (GIST)](towards_global_ai_inclusivity_a_large-scale_multilingual_terminology_dataset_gis.md)
- [\[ACL 2025\] SIFT-50M: A Large-Scale Multilingual Dataset for Speech Instruction Fine-Tuning](sift-50m_a_large-scale_multilingual_dataset_for_speech_instruction_fine-tuning.md)
- [\[ACL 2025\] Just Go Parallel: Improving the Multilingual Capabilities of Large Language Models](just_go_parallel_improving_the_multilingual_capabilities_of_large_language_model.md)

</div>

<!-- RELATED:END -->
