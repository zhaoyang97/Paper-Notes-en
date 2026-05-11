---
title: >-
  [Paper Note] BoundRL: Efficient Structured Text Segmentation through Reinforced Boundary Generation
description: >-
  [ACL 2026][Segmentation][Structured Text Segmentation] BoundRL redefines structured text segmentation as a boundary generation task — generating only each segment's start tokens rather than the complete text…
tags:
  - "ACL 2026"
  - "Segmentation"
  - "Structured Text Segmentation"
  - "Boundary Generation"
  - "RLVR"
  - "Entropy Collapse"
  - "Intermediate Candidates"
date: 2025-04-17
content_hash: 14a71de3f133cca3
---

# BoundRL: Efficient Structured Text Segmentation through Reinforced Boundary Generation

**Conference**: ACL 2026  
**arXiv**: [2510.20151](https://arxiv.org/abs/2510.20151)  
**Code**: N/A  
**Area**: Text Segmentation / Reinforcement Learning  
**Keywords**: Structured Text Segmentation, Boundary Generation, RLVR, Entropy Collapse, Intermediate Candidates

## TL;DR

BoundRL redefines structured text segmentation as a boundary generation task — generating only each segment's start tokens rather than the complete text, reducing output tokens by 90% and eliminating hallucination risk. Combined with a dual-objective reward function and selective perturbation strategy for RLVR training, a 1.7B model surpasses Claude-4 Sonnet's few-shot performance.

## Background & Motivation

**Background**: Text segmentation divides text into semantically coherent segments, widely used in document understanding, QA retrieval, and prompt optimization. Traditional methods segment at sentence or paragraph levels, but structured text (e.g., LLM prompts) contains code snippets, JSON formats, and placeholders that do not conform to traditional sentence/paragraph structures.

**Limitations of Prior Work**: (1) Traditional sentence/paragraph-level segmentation methods are inapplicable to structured text; (2) Token-level sequence labeling produces overly fragmented results; (3) Boundary classification requires classifying every token, with excessive computation; (4) Existing LLM methods (e.g., generating each segment's complete text) face high inference cost and hallucination risk.

**Key Challenge**: Structured text requires token-level fine-grained segmentation, but methods generating complete segment text have inference costs that scale linearly with input length and inevitably introduce hallucination.

**Goal**: Design an efficient token-level structured text segmentation method that achieves both low inference cost and high segmentation quality.

**Key Insight**: Transform the segmentation problem into boundary generation — only generate each segment's start token sequence and label, then locate these tokens in the original text to reconstruct complete segments.

**Core Idea**: By generating only "localization information" (start tokens) rather than "content information" (complete text), output complexity is reduced from O(|d|) to O(n) (n being the number of segments), while customized RLVR training overcomes SFT limitations.

## Method

### Overall Architecture

BoundRL training has two stages: (1) **SFT stage** — teaching the model the output format of generating start token sequences and labels; (2) **RLVR stage** — optimizing segmentation quality using a dual-objective reward function (reconstruction fidelity + semantic alignment), with perturbation-constructed intermediate candidates to alleviate entropy collapse. During inference, complete segments are reconstructed by sequentially locating start tokens in the original text.

### Key Designs

1. **Boundary Generation Output Pattern**:

    - Function: Transform text segmentation from "generating complete content" to "generating localization markers"
    - Mechanism: For input text d, the model outputs only each segment's start token sequence $\hat{s}_i$ (2-10 tokens) and label $\hat{l}_i$. During reconstruction, start token sequences are sequentially located left-to-right in the original text; text between two adjacent start positions forms one segment. Ordering constraints ensure each occurrence of identical start token sequences is uniquely assigned
    - Design Motivation: Output length is proportional to segment count rather than input length, reducing output tokens by 90%; locating from the original text rather than regenerating fundamentally eliminates hallucination

2. **Dual-Objective Reward Function**:

    - Function: Provide precise training signals for RLVR
    - Mechanism: Reward $r(\hat{T}^L) = \rho_{\text{rec}}(\hat{T}^L) \cdot \frac{\text{EM}(\hat{T}^L) + \text{F1}_{\text{char}}(\hat{T}^L)}{2}$. Reconstruction fidelity $\rho_{\text{rec}}$ measures how much of the original text can be recovered from generated segments (by character proportion); semantic alignment is measured through exact match F1 and character-level F1 against annotated segments. Different start tokens corresponding to the same boundaries receive equal rewards
    - Design Motivation: SFT incorrectly penalizes start tokens corresponding to correct boundaries and insufficiently penalizes minor token mismatches. The reward function addresses both issues

3. **Intermediate Candidate Construction**:

    - Function: Alleviate entropy collapse in RLVR training
    - Mechanism: During rollout, apply three perturbations to medium-reward candidate segmentations: (a) truncate segments (remove one word from one end), (b) extend segments (add one word to one end), (c) replace labels. Select the perturbation result with the highest reward as the intermediate candidate, selectively replacing original candidates only when reward improves (up to k replacements)
    - Design Motivation: Reference sequences may be too far from the model's current distribution, making direct learning difficult. Intermediate candidates serve as "stepping stones" bridging current generation and optimal solutions, particularly suited to this paper's continuous and dense reward function

### Loss & Training

SFT stage uses standard cross-entropy loss training for 1 epoch. RLVR stage uses GRPO (without standard deviation normalization), 6 input documents per batch, m=4 candidate segmentations per document, temperature 1.2. Checkpoints saved every 0.2 epochs, with optimal model selected on validation set.

## Key Experimental Results

### Main Results

**Synthetic Test Set Results (Qwen3-1.7b)**

| Method | ρ_rec | EM | F1_char |
|------|-------|-----|---------|
| SFT | 99.9 | 70.6 | 92.2 |
| SFT+RLVR | 99.9 | 75.2 | 93.5 |
| BoundRL | 99.9 | **77.3** | **94.8** |

**Langchain Test Set Results (Qwen3-1.7b)**

| Method | ρ_rec | EM | F1_char |
|------|-------|-----|---------|
| SFT | 86.9 | 39.1 | 73.5 |
| BoundRL | **90.6** | **47.3** | **76.8** |

### Ablation Study

- BoundRL (Qwen3-1.7b) achieves 47.3% EM on Langchain, surpassing Claude-4 Sonnet's few-shot prompting
- Intermediate candidate construction consistently improves across multiple models compared to standard RLVR
- RL-PLUS (using reference candidates instead of intermediate candidates) is slightly worse, validating the hypothesis that intermediate candidates better match the model's distribution

### Key Findings

- The boundary generation paradigm reduces output tokens by 90% while maintaining or even improving segmentation quality
- The dual-objective reward function effectively addresses inherent limitations of SFT in boundary generation tasks
- The intermediate candidate strategy provides an effective and low-cost solution to RLVR entropy collapse
- A 1.7B-parameter model trained with BoundRL surpasses Claude-4 Sonnet's few-shot performance

## Highlights & Insights

- The "generate only localization information, not content" approach is elegantly simple, fundamentally avoiding hallucination
- The perturbation strategy for intermediate candidates is cleverly designed, leveraging the continuity of the reward function
- Experimental design is comprehensive, covering three different scales of base models
- The StructSeg dataset (15.3K annotations) fills a gap in structured text segmentation evaluation

## Limitations & Future Work

- When start tokens cannot be located in the original text, corresponding segments are discarded
- Case studies are limited to LLM prompts; extension to other structured text types is needed
- For very long documents (e.g., entire books), segment count n may still be large
- Future work could extend the boundary generation method to code segmentation, legal document segmentation, and other domains

## Related Work & Insights

- Compared to traditional sequence labeling and boundary classification methods, the boundary generation paradigm achieves a better balance of efficiency and quality
- The intermediate candidate strategy shares philosophical roots with curriculum learning
- Provides valuable design patterns for RLVR application in structured output tasks

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Boundary generation paradigm is a fundamental redefinition of the text segmentation task
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-model, multi-baseline, complete ablation experiments
- Writing Quality: ⭐⭐⭐⭐ Method exposition is clear, diagrams are intuitive

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Weakly-Supervised Referring Video Object Segmentation through Text Supervision](../../CVPR2026/segmentation/wsrvos_weakly_supervised_rvos.md)
- [\[AAAI 2026\] Text-guided Controllable Diffusion for Realistic Camouflage Images Generation](../../AAAI2026/segmentation/text-guided_controllable_diffusion_for_realistic_camouflage_images_generation.md)
- [\[NeurIPS 2025\] TabRAG: Improving Tabular Document Question Answering for Retrieval Augmented Generation via Structured Representations](../../NeurIPS2025/segmentation/tabrag_improving_tabular_document_question_answering_for_retrieval_augmented_gen.md)
- [\[CVPR 2026\] PRUE: A Practical Recipe for Field Boundary Segmentation at Scale](../../CVPR2026/segmentation/prue_a_practical_recipe_for_field_boundary_segmentation_at_scale.md)
- [\[CVPR 2026\] LEMMA: Laplacian Pyramids for Efficient Marine Semantic Segmentation](../../CVPR2026/segmentation/lemma_laplacian_pyramids_for_efficient_marine_semantic_segmentation.md)

</div>

<!-- RELATED:END -->
