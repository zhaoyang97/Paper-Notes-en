---
title: >-
  [Paper Note] Improve Language Model and Brain Alignment via Associative Memory
description: >-
  [ACL 2025][LLM (Other)][brain alignment] By performing data augmentation on text to simulate associative memory and applying associative memory instruction tuning on LLMs, this study demonstrates that both approaches significantly improve the alignment between language models and the human brain in speech comprehension tasks, particularly in associative memory-related brain regions such as the medial temporal lobe.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "brain alignment"
  - "associative memory"
  - "fMRI"
  - "instruction tuning"
  - "GPT-2"
  - "LLaMA-2"
date: 2026-05-08
content_hash: 6d8f696aaf4514de
---

# Improve Language Model and Brain Alignment via Associative Memory

**Conference**: ACL 2025  
**arXiv**: [2505.13844](https://arxiv.org/abs/2505.13844)  
**Code**: [GitHub](https://github.com/lemonsis/Association)  
**Area**: Cognitive Neuroscience / Language Models  
**Keywords**: brain alignment, associative memory, fMRI, instruction tuning, GPT-2, LLaMA-2  

## TL;DR

By performing data augmentation on text to simulate associative memory and applying associative memory instruction tuning on LLMs, this study demonstrates that both approaches significantly improve the alignment between language models and the human brain in speech comprehension tasks, particularly in associative memory-related brain regions such as the medial temporal lobe.

## Background & Motivation

- **Core Problem**: Although language model activations can be linearly mapped to human brain fMRI activity (i.e., brain score), existing research rarely explores the role of associative memory in this alignment. Associative memory is a key cognitive process in human language comprehension that links related concepts and information.
- **Motivation**: Humans automatically generate associations when listening to stories (e.g., hearing "hospital" evokes "doctor" and "nurse"), whereas language models lack this mechanism. Will simulating associative memory content in the model input, or training the model to generate associative content, improve model-brain alignment?
- **Two Research Questions**: (1) Does simulating associative memory (data augmentation) improve the brain score? (2) Does guiding LLMs to generate associative content (instruction tuning) improve the brain score?

## Method

### Overall Architecture

A three-stage experimental design: **Brain Score Calculation** (baseline alignment) $\to$ **Associative Memory Data Augmentation** (simulating associative content) $\to$ **Instruction Tuning** (training LLMs to generate associative content). The Narratives fMRI dataset is utilized, which contains fMRI recordings of 345 participants listening to 27 English stories.

### Key Designs

1. **Brain Score Calculation**: Autoregressive models (GPT-2 / LLaMA-2) are selected to extract activations from each layer. Temporal dimensions are aligned using FIR models, and Ridge regression maps model activations to fMRI signals, with the Pearson correlation coefficient serving as the brain score. A brain score ceiling test is innovatively designed, using half of the subjects to predict the other half to estimate the upper bound of explainability.

2. **Associative Memory Data Augmentation**: Original texts are expanded into augmented texts containing associative content. Two granularities are used: sentence-level (complete semantic sentences) and word-level (noun/adjective/verb phrases). Two annotation types are employed: manual annotation (humans decide where to add associations) and GPT-4 annotation (associations are automatically generated every 4 sentences). Random augmentation is designed as a control group to demonstrate that the improvements stem from associative memory rather than data volume.

3. **Instruction Tuning (Association Dataset)**: An Association dataset with 1000 samples is constructed, where the input consists of story paragraphs + instructions encouraging associative memory, and the output consists of associative content. LLaMA-2 is fine-tuned via SFT using both LoRA and frozen-layer fine-tuning, after which brain scores are recomputed.

### Core Formulas

- **Associative Memory Score**: $\mathcal{F}(X^{(l)}) = \mathcal{R}(X_{mem}^{(l)}) - \mathcal{R}(X^{(l)})$
- **Instruction Tuning Score**: $\mathcal{M}(X^{(l)}) = (\mathcal{R}(X_{sft}^{(l)}) - \mathcal{R}(X^{(l)})) / \mathcal{R}(X^{(l)})$

## Experiments

### Brain Score Baseline

| Model | Best Layer | Highest Brain Score |
|------|:---:|:---:|
| GPT-2 | Layer 9 (of 12) | 0.126 |
| LLaMA-2 | Layer 14 (of 32) | 0.146 |

Due to its larger parameter size and more training data, LLaMA-2 achieves higher alignment. The brain score in the left hemisphere is higher than that in the right hemisphere.

### Associative Memory Augmentation Results

| Augmentation Method | Brain Score Improvement Range | Best Setting |
|------|:---:|:---:|
| Word-level - Manual | 0.0014 — 0.05 | Optimal |
| Sentence-level - Manual | Positive growth but weaker than word-level | Suboptimal |
| Word-level - GPT-4 | Positive growth but weaker than manual | — |
| Random Augmentation | 0 or negative growth | Control Group |

### Instruction Tuning Results

| Method | MTL Brain Region Improvement | Parietal Region Improvement |
|------|:---:|:---:|
| LoRA | 2%—7% | 50%—60% |
| Frozen-layer Fine-tuning | 2%—7% | 50%—60% |

### Key Findings

- Word-level associations outperform sentence-level associations—nouns, verbs, and adjectives carry denser information, which is consistent with neuroscientific research.
- Manual annotation outperforms GPT-4—GPT-4 cannot accurately judge when to trigger associations.
- Random data augmentation is ineffective or even detrimental—proving that the improvement indeed stems from associative memory.
- Associative memory instruction tuning brings significant improvements in both the medial temporal lobe (MTL) and working memory-related brain regions.
- LLaMA-2 outperforms GPT-2 in most Regions of Interest (ROIs).

## Highlights & Insights

- This study is the first to systematically investigate the impact of associative memory on language model-brain alignment, filling a gap in the intersection of cognitive neuroscience and NLP.
- The Association dataset is constructed, demonstrating that "guiding model association" can enhance brain-model alignment, which possesses theoretical novelty.
- Rigorous control experiments (random augmentation) are designed to eliminate confounding factors related to data volume.

## Limitations & Future Work

- The study only uses English stories and fMRI data from English-speaking participants, which may not generalize to other languages and cultures.
- There may be background differences between annotators and fMRI subjects, leading to an inherent mismatch between the annotated associative content and actual brain activity.
- The Association dataset contains only 1,000 samples, which is limited in scale.
- The biological mechanisms of associative memory are extremely complex, and the simulation method in this study remains highly simplified.

## Related Work & Insights

- **Language Model and Brain Alignment**: Jain & Huth (2018); Caucheteux & King (2020); Goldstein et al. (2022) — Model activations can be linearly mapped to fMRI/MEG signals.
- **Associative Memory Research**: Anderson & Bower (2014); Eichenbaum (2017) — Cognitive theories of associative memory and the role of the hippocampus in memory encoding and retrieval.
- **LLM Fine-tuning and Brain**: Moussa et al. (2024) — Fine-tuning speech models with brain-related semantics to improve alignment.

## Rating

| Dimension | Score |
|------|:---:|
| Novelty | ★★★★☆ |
| Practicality | ★★★☆☆ |
| Experimental Thoroughness | ★★★★☆ |
| Writing Quality | ★★★★☆ |
| Overall Rating | ★★★★☆ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Binary Classifier Optimization for Large Language Model Alignment](bco_binary_classifier_alignment.md)
- [\[ACL 2025\] Clue Guided Re-Assessment to Improve Reasoning in Large Language Models](clue_guided_re-assessment_to_improve_reasoning_in_large_language_models.md)
- [\[ACL 2025\] Beyond Dialogue: A Profile-Dialogue Alignment Framework Towards General Role-Playing Language Model](beyond_dialogue_a_profile-dialogue_alignment_framework_towards_general_role-play.md)
- [\[ACL 2025\] Disentangling Memory and Reasoning Ability in Large Language Models](disentangle_memory_reasoning.md)
- [\[ICLR 2026\] When Language Models Lose Their Mind: The Consequences of Brain Misalignment](../../ICLR2026/llm_nlp/when_language_models_lose_their_mind_the_consequences_of_brain_misalignment.md)

</div>

<!-- RELATED:END -->
