---
title: >-
  [Paper Note] Guess or Recall? Training CNNs to Classify and Localize Memorization in LLMs
description: >-
  [AAAI 2026][LLM/NLP][memorization taxonomy] CNNs trained on LLM attention weights are used to evaluate the alignment between memorization taxonomies and actual attention mechanisms. A new three-class taxonomy (Guess/Recall/Non-Memorized) is proposed, improving the minimum F1 from 64.7% to 89.0%, while localizing that different memorization types rely on low-layer (Guess) and high-layer (Recall) attention, respectively.
tags:
  - AAAI 2026
  - LLM/NLP
  - memorization taxonomy
  - attention analysis
  - CNN classifier
  - LLM privacy
  - verbatim memorization
date: 2026-05-08
content_hash: 309b164c3bd406df
---

# Guess or Recall? Training CNNs to Classify and Localize Memorization in LLMs

**Conference**: AAAI 2026
**arXiv**: [2508.02573](https://arxiv.org/abs/2508.02573)
**Code**: [https://github.com/orailix/cnn-4-llm-memo](https://github.com/orailix/cnn-4-llm-memo)
**Area**: LLM/NLP
**Keywords**: memorization taxonomy, attention analysis, CNN classifier, LLM privacy, verbatim memorization

## TL;DR
CNNs trained on LLM attention weights are used to evaluate the alignment between memorization taxonomies and actual attention mechanisms. A new three-class taxonomy (Guess/Recall/Non-Memorized) is proposed, improving the minimum F1 from 64.7% to 89.0%, while localizing that different memorization types rely on low-layer (Guess) and high-layer (Recall) attention, respectively.

## Background & Motivation

**Background**: LLMs verbatim memorize substantial portions of training data, raising privacy and legal concerns. Prashanth et al. proposed a four-class taxonomy (Non-memorized/Recite/Reconstruct/Recollect) that distinguishes memorization types using high-level features such as perplexity and token frequency.

**Limitations of Prior Work**: Existing taxonomies are defined based on high-level semantic features but have not been validated as reflecting distinct internal attention mechanisms. CNN classification experiments reveal frequent misclassification.

**Key Challenge**: The Recollect class (low-frequency memorization) does not correspond to a distinctive attention pattern; many samples should instead be assigned to Recite or Reconstruct.

**Goal**: To identify a memorization taxonomy that is genuinely aligned with the attention mechanisms of LLMs.

**Key Insight**: Taxonomy evaluation is reformulated as a classification task — a well-defined taxonomy should enable a CNN to readily distinguish attention weight patterns across classes.

**Core Idea**: CNN classification performance serves as a proxy for taxonomy quality, enabling a systematic search over 54 candidate taxonomies to identify the optimal one.

## Method

### Overall Architecture

1. Collect 32-extractable samples from The Pile, the training corpus of Pythia models.
2. Extract attention weight matrices $\mathbf{A}_l^h \in \mathbb{R}^{64 \times 64}$ across all layers and heads.
3. Train a CNN classifier under a given taxonomy.
4. Evaluate taxonomy–attention alignment using minimum F1.
5. Systematically search 54 candidate taxonomies to identify the optimal one.

### Key Designs

1. **CNN Classifier Architecture**

    - **Function**: Classify different memorization types based on attention weights.
    - **Mechanism**: Two convolutional layers + ReLU + Dropout + MaxPooling + two fully connected layers. The number of input channels equals the number of model layers (with max-pooling over heads).
    - **Design Motivation**: Attention patterns in memorized samples (diagonal lines from repeated subsequences, vertical stripes at newlines) are inherently translation-invariant, making CNNs a natural fit.

2. **Taxonomy Parameterization and Systematic Search**

    - **Function**: Model taxonomies as decision trees and systematically explore all candidates.
    - **Mechanism**: Two types of decision nodes are defined — Duplication-based (partitioning by training-set repetition count $\delta$) and Completion-based (assessing suffix predictability via ROUGE scores $\lambda, \gamma$). A total of 54 taxonomies are composed and evaluated.
    - **Design Motivation**: Avoids human design bias; the optimal taxonomy is identified in a data-driven manner.

3. **Winning Taxonomy: Non-Memo / Guess / Recall**

    - **Guess**: Suffix tokens are largely inferrable from the prefix (ROUGE-1 > 0.5, ROUGE-L > 0.5).
    - **Recall**: Samples genuinely recalled from training data.
    - **Key Finding**: Subdividing by repetition count is unnecessary; repetition does not alter the nature of memorization.

4. **Customized Interpretability Technique**

    - **Function**: Localize attention weight regions most contributive to each memorization type.
    - **Mechanism**: Guided backpropagation computes discriminative contributions $\mathbf{C}_l^h$, which are multiplied by actual attention values, then max-pooled across heads and averaged across samples to obtain $\Delta_l[t_0]$.
    - **Design Motivation**: Standard GradCAM fails under 36 channels, necessitating a customized approach.

### Loss & Training

Cross-entropy loss. For each taxonomy, 8 CNNs × 3 model scales × 3 epochs = 144,000 test predictions are generated. A deliberately small dataset is used to focus on salient patterns.

## Key Experimental Results

### Main Results

| Taxonomy | # Classes | Min F1 | Normalized F1 |
|----------|-----------|--------|---------------|
| Prashanth et al. | 4 | 64.7% | — |
| Best 4-class | 4 | 72.8% | 63.7% |
| **Ours (3-class)** | **3** | **89.0%** | **83.6%** |

### Memorization Localization Results

| Memorization Type | Key Layers | Attention Pattern | Notes |
|-------------------|------------|-------------------|-------|
| Guess | Low (6–9) | Diagonal pattern | Syntactic/template dependency |
| Recall | High (31–36) | Short-range interactions below diagonal | Fills gaps using adjacent tokens |
| Non-Memo | Middle layers | Broadly distributed | General language modeling capacity |

### Key Findings
- **Low-frequency memorization is an artifact**: The low F1 of Recollect indicates it does not correspond to a distinctive attention mechanism.
- **Repetition does not alter the nature of memorization**: The optimal taxonomy does not depend on repetition count.
- **Guess and Recall rely on different layers**: Low layers handle syntactic dependencies; high layers handle short-range recall.

## Highlights & Insights
- **Methodological Innovation**: The paradigm of using classifier performance to evaluate the quality of scientific taxonomies is broadly transferable.
- **Reconciling Conflicting Findings**: Stoehr et al. found low layers to be important while Menta et al. found high layers to be important; this work provides a unified explanation — low layers correspond to Guess and high layers to Recall.
- **Implications for Privacy Research**: Many "memorized" samples are in fact guessed, which has direct implications for mitigation strategies.

## Limitations & Future Work
- Validation is limited to the Pythia model family, constrained by the requirement for full access to training data.
- Only attention blocks are analyzed; FFN layers are excluded.
- Localization is performed indirectly via CNNs rather than through direct ablation.
- The 32-extractable definition is restrictive, leaving approximate memorization uncovered.

## Related Work & Insights
- **vs. Prashanth et al.**: The four-class taxonomy achieves only 64.7% F1, whereas the proposed three-class taxonomy reaches 89.0%, demonstrating that Recollect is an artifact.
- **vs. Stoehr et al.**: Their finding that low-layer attention heads are important is explained by the prevalence of Guess (comprising 54% of code samples).
- **vs. Menta et al.**: Their observation that disabling high layers reduces memorization is explained as primarily affecting Recall.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The methodology of using classifier performance to evaluate taxonomies is original.
- Experimental Thoroughness: ⭐⭐⭐⭐ Systematic search over 54 taxonomies + 3 model scales + customized interpretability.
- Writing Quality: ⭐⭐⭐⭐⭐ The logical chain from problem to method to findings is clearly and tightly structured.
- Value: ⭐⭐⭐⭐⭐ Represents a paradigm-level contribution to the understanding of LLM memorization.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] SubSpec: Speculate Deep and Accurate — Lossless and Training-Free Acceleration for Offloaded LLMs](../../NeurIPS2025/llm_nlp/speculate_deep_and_accurate_lossless_and_training-free_acceleration_for_offloade.md)
- [\[ACL 2026\] Revisiting Non-Verbatim Memorization in Large Language Models: The Role of Entity Surface Forms](../../ACL2026/llm_nlp/revisiting_non-verbatim_memorization_in_large_language_models_the_role_of_entity.md)
- [\[ICLR 2026\] PT2-LLM: Post-Training Ternarization for Large Language Models](../../ICLR2026/llm_nlp/pt2-llm_post-training_ternarization_for_large_language_models.md)
- [\[CVPR 2026\] SketchDeco: Training-Free Latent Composition for Precise Sketch Colourisation](../../CVPR2026/llm_nlp/sketchdeco_training-free_latent_composition_for_precise_sketch_colourisation.md)
- [\[AAAI 2026\] LoopLLM: Transferable Energy-Latency Attacks in LLMs via Repetitive Generation](loopllm_transferable_energy-latency_attacks_in_llms_via_repetitive_generation.md)

</div>

<!-- RELATED:END -->
