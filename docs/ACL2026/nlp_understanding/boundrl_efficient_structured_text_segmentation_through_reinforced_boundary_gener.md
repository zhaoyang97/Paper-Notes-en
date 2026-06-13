---
title: >-
  [Paper Note] BoundRL: Efficient Structured Text Segmentation through Reinforced Boundary Generation
description: >-
  [ACL 2026][NLP Understanding][Structured Text Segmentation] BoundRL redefines structured text segmentation as a boundary generation task—generating only the starting tokens of each segment rather than the full text. This…
tags:
  - "ACL 2026"
  - "NLP Understanding"
  - "Structured Text Segmentation"
  - "Boundary Generation"
  - "RLVR"
  - "Entropy Collapse"
  - "Intermediate Candidates"
date: 2026-05-08
content_hash: bc305b92e2472706
---

# BoundRL: Efficient Structured Text Segmentation through Reinforced Boundary Generation

**Conference**: ACL 2026 Findings  
**arXiv**: [2510.20151](https://arxiv.org/abs/2510.20151)  
**Code**: None  
**Area**: Text Segmentation/Reinforcement Learning  
**Keywords**: Structured Text Segmentation, Boundary Generation, RLVR, Entropy Collapse, Intermediate Candidates

## TL;DR

BoundRL redefines structured text segmentation as a boundary generation task—generating only the starting tokens of each segment rather than the full text. This reduces output tokens by 90% and eliminates hallucination risks. Combined with RLVR training using a dual-objective reward function and a selective perturbation strategy, a 1.7B small model outperforms Claude-4 Sonnet's few-shot performance.

## Background & Motivation

**Background**: Text segmentation divides text into semantically coherent segments, widely used for document understanding, QA retrieval, and prompt optimization. Traditional methods perform segmentation at the sentence or paragraph level, but structured text (such as LLM prompts) contains code snippets, JSON formats, and placeholders that do not conform to traditional sentence-paragraph structures.

**Limitations of Prior Work**: (1) Traditional sentence/paragraph-level segmentation methods are unsuitable for structured text; (2) Token-level sequence labeling generates overly fragmented results; (3) Boundary classification requires classifying every token, which is computationally expensive; (4) Existing LLM methods (e.g., generating the full text of each segment) face high inference costs and hallucination risks.

**Key Challenge**: Structured text requires fine-grained token-level segmentation, but methods generating full-segment text involve inference costs that grow linearly with input length and inevitably introduce hallucinations.

**Goal**: Design an efficient token-level structured text segmentation method that achieves both low inference cost and high segmentation quality.

**Key Insight**: Convert the segmentation problem into boundary generation—generating only the starting token sequence and label for each segment, then locating these tokens in the original text to reconstruct the full segments.

**Core Idea**: By generating "localization information" (starting tokens) instead of "content information" (full text), the output complexity is reduced from $O(|d|)$ to $O(n)$ (where $n$ is the number of segments), while overcoming SFT limitations through customized RLVR training.

## Method

### Overall Architecture

The training of BoundRL is divided into two stages: (1) **SFT Phase**—teaching the model to generate the output format of starting token sequences and labels; (2) **RLVR Phase**—optimizing segmentation quality using a dual-objective reward function (reconstruction fidelity + semantic alignment) and mitigating entropy collapse by constructing intermediate candidates through perturbations. During inference, full segments are reconstructed by sequentially locating starting tokens in the original text.

### Key Designs

1.  **Boundary Generation Output Pattern**:
    - **Function**: Transitions text segmentation from "generating full content" to "generating localization markers."
    - **Mechanism**: For an input text $d$, the model only outputs the starting token sequence $\hat{s}_i$ (2-10 tokens) and label $\hat{l}_i$ for each segment. During reconstruction, each starting token sequence is located sequentially from left to right in the original text. The text between two adjacent starting positions constitutes a segment. Ordering constraints ensure that even if the same starting token sequence appears multiple times, each occurrence is uniquely assigned.
    - **Design Motivation**: Output length is proportional to the number of segments rather than the input length, reducing output tokens by 90%; locating from the original text instead of regenerating fundamentally eliminates hallucinations.

2.  **Dual-Objective Reward Function**:
    - **Function**: Provides precise training signals for RLVR.
    - **Mechanism**: The reward is defined as $r(\hat{T}^L) = \rho_{\text{rec}}(\hat{T}^L) \cdot \frac{\text{EM}(\hat{T}^L) + \text{F1}_{\text{char}}(\hat{T}^L)}{2}$. Reconstruction fidelity $\rho_{\text{rec}}$ measures how much of the original text can be recovered from the generated segments (by character ratio); semantic alignment measures consistency with annotated segments via Exact Match F1 and character-level F1. Different starting tokens for the same boundary receive the same reward.
    - **Design Motivation**: SFT incorrectly penalizes starting tokens that correspond to correct boundaries and provides insufficient penalties for minor token mismatches. The reward function addresses both issues.

3.  **Intermediate Candidate Construction**:
    - **Function**: Mitigates the entropy collapse problem in RLVR training.
    - **Mechanism**: During the rollout phase, three types of perturbations are applied to candidate segmentations with medium rewards: (a) truncating a segment (removing one word from an end), (b) extending a segment (adding one word to an end), and (c) replacing labels. The perturbation result with the highest reward is selected as an intermediate candidate, selectively replacing the original candidate (up to $k$ candidates) only if the reward increases.
    - **Design Motivation**: Annotated sequences as references may be too far from the model's current distribution, making direct learning difficult. Intermediate candidates serve as "stepping stones" bridging current generation and the optimal solution, particularly suitable for the continuous and dense reward function in this work.

### Loss & Training

The SFT phase uses standard cross-entropy loss for 1 epoch. The RLVR phase employs GRPO (without standard deviation normalization), with 6 input documents per batch, $m=4$ candidate segmentations per generation, and a temperature of 1.2. Checkpoints are saved every 0.2 epoch, and the best model is selected based on the validation set.

## Key Experimental Results

### Main Results

**Synthetic Test Set Results (Qwen3-1.7b)**

| Method | $\rho_{rec}$ | EM | $F1_{char}$ |
|------|-------|-----|---------|
| SFT | 99.9 | 70.6 | 92.2 |
| SFT+RLVR | 99.9 | 75.2 | 93.5 |
| BoundRL | 99.9 | **77.3** | **94.8** |

**Langchain Test Set Results (Qwen3-1.7b)**

| Method | $\rho_{rec}$ | EM | $F1_{char}$ |
|------|-------|-----|---------|
| SFT | 86.9 | 39.1 | 73.5 |
| BoundRL | **90.6** | **47.3** | **76.8** |

### Ablation Study

- BoundRL (Qwen3-1.7b) achieves an EM of 47.3% on Langchain, surpassing Claude-4 Sonnet's few-shot prompting performance.
- Intermediate candidate construction brings consistent improvements over standard RLVR across multiple models.
- RL-PLUS (using reference candidates instead of intermediate candidates) performs slightly worse, validating the hypothesis that intermediate candidates are closer to the model's distribution.

### Key Findings

- The boundary generation paradigm reduces output tokens by 90% while maintaining or even improving segmentation quality.
- The dual-objective reward function effectively addresses the inherent limitations of SFT on boundary generation tasks.
- The intermediate candidate strategy provides an effective and low-cost solution to the entropy collapse problem in RLVR.
- Small models with 1.7B parameters trained via BoundRL can outperform Claude-4 Sonnet's few-shot performance.

## Highlights & Insights

- The approach of "generating only localization information without content" is simple and elegant, fundamentally avoiding hallucinations.
- The perturbation strategy for intermediate candidates is cleverly designed, leveraging the continuous nature of the reward function.
- The experimental design is comprehensive, covering three different scales of base models.
- The construction of the StructSeg dataset (15.3K annotations) fills a gap in structured text segmentation evaluation.

## Limitations & Future Work

- When a starting token cannot be located in the original text, the corresponding segment is discarded.
- Case studies were only conducted on LLM prompts; the method has not been extended to other types of structured text.
- For ultra-long documents (e.g., entire books), the number of segments $n$ may still be large.
- Future work could generalize the boundary generation method to fields such as code segmentation and legal document segmentation.

## Related Work & Insights

- Compared to traditional sequence labeling and boundary classification methods, the boundary generation paradigm achieves a better balance between efficiency and quality.
- The intermediate candidate strategy is consistent with the philosophy of curriculum learning.
- This work provides a valuable design pattern for applying RLVR to structured output tasks.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The boundary generation paradigm is a fundamental redefinition of the text segmentation task.
- Experimental Thoroughness: ⭐⭐⭐⭐ Complete with multiple models, multiple baselines, and ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear explanation of methodology and intuitive illustrations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Accurate and Efficient Statistical Testing for Word Semantic Breadth](accurate_and_efficient_statistical_testing_for_word_semantic_breadth.md)
- [\[ACL 2026\] Exploring Concreteness Through a Figurative Lens](exploring_concreteness_through_a_figurative_lens.md)
- [\[ACL 2026\] Filling the Gap: Is Commonsense Knowledge Generation useful for Natural Language Inference?](filling_the_gap_is_commonsense_knowledge_generation_useful_for_natural_language_.md)
- [\[ACL 2026\] Unveiling Temporal Framing in News Text](uncovering_temporal_framing_in_the_news.md)
- [\[ACL 2026\] Reasoning-Based Refinement of Unsupervised Text Clusters with LLMs](reasoning-based_refinement_of_unsupervised_text_clusters_with_llms.md)

</div>

<!-- RELATED:END -->
