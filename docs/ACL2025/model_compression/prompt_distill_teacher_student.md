---
title: >-
  [Paper Note] Prompt Candidates, then Distill: A Teacher-Student Framework for LLM-driven Data Annotation
description: >-
  [ACL 2025][Model Compression][LLM Data Annotation] This paper proposes the CanDist framework. Drawing inspiration from human "ambiguity aversion" behavior under uncertainty, it guides the LLM to output multiple candidate labels instead of a single label (candidate annotation). It then distills these annotations into a small language model (SLM) via a Distribution Refinery strategy to obtain final labels. Both theoretical and experimental results demonstrate that candidate ann…
tags:
  - "ACL 2025"
  - "Model Compression"
  - "LLM Data Annotation"
  - "Candidate Annotation"
  - "Knowledge Distillation"
  - "Teacher-Student"
  - "Text Classification"
date: 2026-05-08
content_hash: e7a5b3952df38b22
---

# Prompt Candidates, then Distill: A Teacher-Student Framework for LLM-driven Data Annotation

**Conference**: ACL 2025  
**arXiv**: [2506.03857](https://arxiv.org/abs/2506.03857)  
**Code**: [Available](https://github.com/MingxuanXia/CanDist)  
**Area**: Model Distillation / Data Annotation  
**Keywords**: LLM Data Annotation, Candidate Annotation, Knowledge Distillation, Teacher-Student, Text Classification

## TL;DR

This paper proposes the CanDist framework. Drawing inspiration from human "ambiguity aversion" behavior under uncertainty, it guides the LLM to output multiple candidate labels instead of a single label (candidate annotation). It then distills these annotations into a small language model (SLM) via a Distribution Refinery strategy to obtain final labels. Both theoretical and experimental results demonstrate that candidate annotation distillation outperforms single-label distillation.

## Background & Motivation

**Background**: LLM-driven automatic data annotation has been widely applied in NLP tasks such as text classification, NER, and sentiment analysis, significantly reducing human annotation costs.

**Limitations of Prior Work**: Existing approaches adopt an "aggressive strategy" by forcing LLMs to output a single deterministic label for each sample. When the LLM is uncertain about hard samples, this strategy often yields completely incorrect annotations, which not only wastes computational resources but also severely degrades data quality for downstream tasks.

**Key Challenge**: LLMs have limited knowledge of downstream tasks and are forced to be "overconfident" by providing a single answer when encountering uncertain samples, leading to a substantial increase in error-labeled data.

**Goal**: Can LLMs be guided to produce more valuable outputs (rather than completely incorrect labels) when they are uncertain?

**Key Insight**: Inspired by "ambiguity aversion" in human behavior, where humans tend to act conservatively rather than overconfidently under uncertainty. This philosophy is injected into the LLM annotation process, allowing LLMs to output multiple possible labels (candidate annotations), which are then distilled by an SLM to yield correct labels.

**Core Idea**: Allow uncertain LLMs to output candidate sets instead of a single label, and then leverage an SLM to distill the correct ground-truth from the candidate set; a conservative strategy is superior to overconfidence.

## Method

### Overall Architecture

CanDist consists of two stages: (1) **Prompt Candidates**: Guiding the LLM to output a candidate label set using two prompting strategies (CA_add: outputting one answer first, then appending other possible answers; CA_all: directly outputting all possible answers); (2) **Distill**: Training an SLM under the constraint of the LLM's candidate labels to progressively identify the correct label via distribution refinery.

### Key Designs

1. **Candidate Annotation Prompting Strategies (CA_add / CA_all)**
    - **Function**: Guides the LLM to output multiple potential labels when uncertain.
    - **Mechanism**: CA_add appends "if you are uncertain, please include other possible options" after the standard prompt; CA_all directly requests "all possible categories".
    - **Design Motivation**: CA_all improves the 1-α-error metric by 18-27% compared to single annotation (SA), significantly increasing correct label coverage.

2. **Distribution Refinery (DR)**
    - **Function**: Dynamically identifies correct labels from candidate labels.
    - **Mechanism**: Leverages the memorization effect of DNNs, where the SLM learns simple patterns first, allowing some true labels to emerge from false positives. The target training distribution is initialized as a uniform distribution over the candidate labels. Post-iteration target distributions are updated by re-normalizing the SLM's predicted softmax outputs within the candidate set.
    - **Design Motivation**: Directly training on uniform distributions of candidate labels is sub-optimal; dynamic refining is needed to highlight true labels.

3. **Sample Filtering and Distribution Sharpening**
    - **Function**: Handles edge cases where candidate sets do not contain the correct label, and accelerates convergence.
    - **Mechanism**: Filters "out-of-candidate" samples (where the SLM's maximum prediction falls outside the candidate set). For reliable samples (small-loss samples within classes), the distribution is sharpened using a temperature parameter. For high-confidence out-of-candidate samples, they are trained using their predicted labels.
    - **Design Motivation**: A small fraction of samples have correct labels outside the candidate set, which can interfere with the distillation process.

### Loss & Training

The overall training objective is the cross-entropy loss $\mathcal{L}_{dr} = \frac{1}{n}\sum_{i=1}^{n} l_{ce}(\boldsymbol{p}_i, \hat{\boldsymbol{q}}_i)$, where the target distribution $\hat{\boldsymbol{q}}$ is dynamically adjusted based on sample categories: reliable samples are sharpened with a temperature parameter $\gamma$, standard in-candidate samples use the standard DR distribution, and high-confidence out-of-candidate samples use the predicted class as the one-hot target. RoBERTa-Base is adopted as the SLM, and GPT-3.5 is used as the Teacher LLM.

## Key Experimental Results

### Main Results

| Method | TREC | MA | DBP | AGN | RCT | BANK |
|------|------|-----|-----|-----|-----|------|
| Zero-shot | 72.20 | 63.12 | 93.94 | 87.24 | 61.83 | 68.41 |
| Few-shot | 77.20 | 63.40 | 95.40 | 88.05 | 65.85 | 68.86 |
| FreeAL | 82.33 | 64.13 | 97.92 | 88.64 | 68.32 | 74.58 |
| **CanDist_add** | **83.13** | **64.23** | **98.72** | **89.46** | **69.77** | **76.27** |
| CanDist_all | 87.80 | 64.20 | 98.65 | 88.78 | 70.57 | 75.97 |
| SFT (Labeled) | 97.80 | 64.54 | 98.78 | 92.29 | 84.52 | 93.31 |

### Ablation Study

| Ablation Setting | Avg. Training Set Accuracy |
|---------|---------------|
| CanDist_add | 79.16 |
| CanDist_add + LLM Select | 75.42 (-3.74) |
| CanDist_all | 78.86 |
| CanDist_all + LLM Select | 74.96 (-3.90) |
| Few-shot (SA) | 74.79 |

### Key Findings

- Candidate annotation (CA_all) achieves a 14-27% gain in 1-α-error compared to single annotation (SA), alongside consistently higher F1 scores.
- CanDist comprehensively outperforms all LLM and SLM baselines on 6 datasets, including FreeAL.
- Distilling candidate annotations with an SLM is far superior to having the LLM make a secondary selection from candidates themselves (avg. difference of 3.74-3.90%). This suggests that SLM's distillation capability is superior to the LLM's secondary judgment.
- Theoretical proof (Theorem 1): Distillation from top-2 candidates yields a looser condition for achieving 100% accuracy compared to distilling from top-1 single annotations.

## Highlights & Insights

- **Elegant Analogy**: Applying human "ambiguity aversion" to LLM annotation provides a natural and convincing core idea.
- **Theoretical Insurance**: Rigorously proves that distilling candidate annotations has a better noise tolerance upper bound compared to single-label distillation.
- The distribution refinery strategy ingeniously exploits the DNN memorization effect: learning simple samples first -> revealing true labels from candidates -> iterative refining.
- The method is highly lightweight and generalizable: candidate annotations can be obtained simply by modifying the prompt without retraining the LLM.
- Reveals a counter-intuitive conclusion: SLM distillation of candidate sets > LLM secondary selection directly from candidate sets.

## Limitations & Future Work

- Validated only on text classification tasks; applicability to sequence labeling and generative tasks remains unexplored.
- The candidate annotation strategy relies on the LLM's ability to correctly follow rules like "please output all possible labels", which may fail on models with poor instruction-following capability.
- CA_all might lead to an excessively large candidate set (approaching the full label space), potentially degrading performance in scenarios with massive label spaces.
- Hyperparameters in the distribution refinery (temperature $\gamma$, high-confidence threshold $\tau$, small-loss ratio $\delta$) require tuning.
- Has not explored candidate annotation paradigms in non-LLM scenarios (such as human annotation teams).

## Related Work & Insights

- Difference from Self-Consistency (SC): SC exploits LLM randomness (temperature sampling), whereas CanDist prompts LLMs to output intrinsic uncertainty.
- Relation to FreeAL: FreeAL is also a pioneer in collaborative SLM annotation but only distills single annotations; CanDist demonstrates that distilling candidate annotations is strictly superior.
- Insight: When facing LLM uncertainty, letting the LLM candidly admit its uncertainty (by outputting candidate sets) is more valuable than forcing it to output a single answer.

## Rating

- **Novelty**: ⭐⭐⭐⭐ (The combined idea of candidate annotation paradigm and distillation is novel, with theoretical analysis as an added bonus)
- **Experimental Thoroughness**: ⭐⭐⭐⭐ (Sufficient with 6 datasets, multiple baseline comparisons, and thorough ablation, but limited to a single task type)
- **Writing Quality**: ⭐⭐⭐⭐⭐ (The motivation is clearly stated, with intuitive figures, and well-integrated theory and experiments)
- **Value**: ⭐⭐⭐⭐ (Provides a new paradigm for LLM data annotation, offering practical value for NLP practitioners)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Adversarially Robust Distillation by Reducing the Student-Teacher Variance Gap](../../ECCV2024/model_compression/adversarially_robust_distillation_by_reducing_the_student-teacher_variance_gap.md)
- [\[CVPR 2026\] Masking Teacher and Reinforcing Student for Distilling Vision-Language Models](../../CVPR2026/model_compression/masking_teacher_and_reinforcing_student_for_distilling_vision-language_models.md)
- [\[ICML 2025\] Toward Data-centric Directed Graph Learning: An Entropy-driven Approach](../../ICML2025/model_compression/toward_data-centric_directed_graph_learning_an_entropy-driven_approach.md)
- [\[NeurIPS 2025\] ORPO-Distill: Mixed-Policy Preference Optimization for Cross-Architecture LLM Distillation](../../NeurIPS2025/model_compression/orpo-distill_mixed-policy_preference_optimization_for_cross-architecture_llm_dis.md)
- [\[ACL 2025\] STUN: Structured-Then-Unstructured Pruning for Scalable MoE Pruning](stun_moe_pruning.md)

</div>

<!-- RELATED:END -->
