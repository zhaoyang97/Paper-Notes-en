---
title: >-
  [Paper Note] Revealing the Deceptiveness of Knowledge Editing: A Mechanistic Analysis of Superficial Editing
description: >-
  [ACL 2025][Knowledge Editing][Superficial Editing] This paper defines the phenomenon of "superficial editing"—where models modified by knowledge editing algorithms perform well under standard prompts but revert to the original knowledge under hand-crafted adversarial probes—and reveals through mechanistic analysis that the residual stream in early layers and specific attention heads in late layers are two crucial factors causing this phenomenon.
tags:
  - "ACL 2025"
  - "Knowledge Editing"
  - "Superficial Editing"
  - "Mechanistic Analysis"
  - "Attention Heads"
  - "Residual Stream"
date: 2026-05-08
content_hash: 940d7976da9d01e8
---

# Revealing the Deceptiveness of Knowledge Editing: A Mechanistic Analysis of Superficial Editing

**Conference**: ACL 2025  
**arXiv**: [2505.12636](https://arxiv.org/abs/2505.12636)  
**Code**: Yes  
**Area**: NLP / Knowledge Editing  
**Keywords**: Knowledge Editing, Superficial Editing, Mechanistic Analysis, Attention Heads, Residual Stream

## TL;DR

This paper defines the phenomenon of "superficial editing"—where models modified by knowledge editing algorithms perform well under standard prompts but revert to the original knowledge under hand-crafted adversarial probes—and reveals through mechanistic analysis that the residual stream in early layers and specific attention heads in late layers are two crucial factors causing this phenomenon.

## Background & Motivation

Knowledge editing aims to update specific knowledge in large language models while keeping other knowledge unaffected. Existing knowledge editing algorithms (e.g., ROME, MEMIT, PMET, AlphaEdit) achieve near-perfect performance under conventional evaluation metrics (Efficacy, Generalization, Locality). However, the authors discover a critical problem: **the edited models revert to the original knowledge when presented with carefully designed contextual inputs**. For example, while an edited model can correctly answer "Who is the President of the United States" with the new answer, the model generates the original answer again when a contextual prefix like "Is Joe Biden the President of the United States?" is added to the input.

This phenomenon suggests that the effects of current update methods are potentially "deceptive"—editing does not truly alter the model's internal knowledge representation. The authors define this as "superficial editing" and systematically investigate the underlying mechanistic causes. The motivation of this work is that relying solely on traditional metrics to evaluate knowledge editing is insufficient, and deeper evaluation methods along with a better understanding of the editing mechanism are required.

## Method

### Overall Architecture

The research workflow of this paper is divided into three stages:
1. **Formal Definition and Evaluation**: Formulating the mathematical definition of superficial editing, constructing attack probes, and systematically evaluating existing algorithms.
2. **Mechanistic Analysis**: Starting from three core Transformer components (residual stream, MLP, and attention), locating and verifying the causal factors leading to superficial editing.
3. **Generalization Validation**: Extending the analysis method to the "superficial unlearning" task to verify the generalizability of the conclusions.

### Key Designs

1. **Attack Probe Design**

    - Three types of attack prefixes are designed:
        - **Wiki(o)**: Wikipedia summary of the original answer
        - **Rep(o)**: Repetition of the original answer
        - **Que(o)**: Questions containing the subject, relation, and original answer (e.g., "Is Joe Biden the President of the United States?")
    - Attack probe = Attack prefix ⊕ Baseline prompt
    - Design Motivation: Simulating contextual interferences that the model might encounter in real-world scenarios.

2. **Quantitative Metrics for Superficial Editing**

    - **OM (Original Match)**: The ratio of predictions by the edited model under the attack probe that match the original answer.
    - **OP (Original Probability)**: The ratio of times the probability of the original answer exceeds that of the new answer.
    - Higher OM and OP indicate more severe superficial editing.

3. **Residual Stream Intervention Experiments**

    - Two forward passes are designed: "clean run" (baseline prompt) and "corrupted run" (attack probe).
    - At specific layers and token positions, the hidden states of the clean run are replaced with those of the corrupted run.
    - The focus is on two key positions: the last subject token (having large influence in early layers) and the last token (having large influence in late layers).
    - A "Residual Stream Reversal" (RRS) phenomenon is observed in late layers, where the probability of the original answer exceeds that of the new answer.

4. **Attention Head Analysis**

    - A **LOPH (Latent Original Probability of Head)** metric is proposed: computing the latent probability of the original answer in the output of each attention head via the logit lens technique.
    - Specific attention heads in late layers exhibit significantly higher LOPH values, which inject original knowledge information into the last position.
    - By applying SVD decomposition to the output matrices of the attention heads, specific left singular vectors are found to encode the original knowledge.

5. **Two Core Hypotheses**

    - **H1**: In early layers, the enrichment of new knowledge at the last subject position is inhibited, but the accumulation of original knowledge is also limited.
    - **H2**: Attention modules in late layers actively integrate original knowledge information into the last position, leading to the RRS phenomenon and triggering superficial editing.

### Loss & Training

This is an analytical work and does not involve new training strategies. The validation methods include:
- **Inhibition Score**: Measuring the degree of inhibition of new knowledge enrichment through the negative log-probability of the logit lens.
- **Ablation Study**: Zeroing out the outputs of key attention layers/heads and observing the mitigation effect on superficial editing.
- **Causal Validation**: Identifying left singular vectors encoding the original knowledge through SVD decomposition of the attention head output matrices.

## Key Experimental Results

### Main Results (LLaMA3-8B-Instruct on CF-a Dataset)

| Method | Eff. | Gen. | Loc. | Wiki-OM↓ | Wiki-OP↓ | Rep-OM↓ | Rep-OP↓ | Que-OM↓ | Que-OP↓ |
|------|------|------|------|----------|----------|---------|---------|---------|---------|
| FT | 100 | 80.51 | 52.37 | 49.45 | 51.65 | 30.68 | 35.98 | 29.07 | 31.40 |
| ROME | 100 | 94.92 | 85.08 | 54.95 | 58.24 | 61.74 | 64.02 | 38.37 | 38.37 |
| MEMIT | 100 | 94.07 | 86.10 | 52.75 | 54.95 | 40.15 | 42.42 | 37.21 | 37.21 |
| PMET | 94.92 | 85.59 | 90.00 | **70.33** | **72.43** | **66.67** | **71.97** | 39.29 | 41.67 |
| AlphaEdit | 100 | 83.90 | 88.98 | **72.53** | **73.62** | **68.18** | **71.97** | 34.52 | 35.71 |

Key Findings: PMET and AlphaEdit perform near-perfectly on traditional metrics, but have OM values exceeding 70% under Wiki attacks, indicating extremely severe superficial editing.

### Ablation Study (Impact of Attention Head Ablation on Superficial Editing)

| Model | Method | Original Ans. Prob. (No Ablation) | Original Ans. Prob. (Ablated) | ΔP↓ | New Ans. Prob. (No Ablation) | New Ans. Prob. (Ablated) | ΔP↑ |
|------|------|---------------------|---------------------|-----|-------------------|-------------------|-----|
| LLaMA3-8B | ROME | 57.17 | 35.58 | 21.59 | 16.49 | 20.71 | 4.22 |
| LLaMA3-8B | MEMIT | 56.90 | 37.36 | 19.54 | 15.68 | 18.38 | 2.70 |
| Qwen2.5-7B | ROME | 57.83 | 36.52 | 21.31 | 11.84 | 17.57 | 5.73 |
| Qwen2.5-7B | MEMIT | 57.54 | 32.40 | 25.14 | 12.21 | 26.08 | 13.87 |
| Qwen2.5-14B | ROME | 55.71 | 39.99 | 15.72 | 13.99 | 21.40 | 7.41 |
| Qwen2.5-14B | MEMIT | 55.03 | 37.25 | 17.78 | 13.79 | 22.24 | 8.45 |

### Key Findings

1. **Prevalence of Superficial Editing**: All parameter editing methods suffer from superficial editing, which is not captured by traditional metrics.
2. **Residual Stream Reversal is Key**: In the late layers, the residual stream at the last position exhibits a "reversal" phenomenon where the probability of the original answer exceeds that of the new answer.
3. **MLP is Not the Culprit**: The outputs of MLP layers consistently decrease the probability of the original answer, showing no causal relationship with superficial editing.
4. **Attention Heads are Core**: A small number of attention heads (LOPH > 0.1) in late layers inject original knowledge into the last position; ablating them significantly alleviates superficial editing.
5. **SVD Reveals Micro-Mechanism**: Specific left singular vectors (Top-5% to 10%) of the attention head output matrices encode the original knowledge.
6. **Generalizable Method**: The same analytical framework observes consistent patterns in the "superficial unlearning" task.

## Highlights & Insights

- **Prominent Conceptual Contribution**: This work is the first to systematically define and quantify the phenomenon of "superficial editing", filling a blind spot in the evaluation of knowledge editing.
- **In-depth Analysis Levels**: The analysis progresses step-by-step from macro (residual stream) to micro (attention heads → SVD singular vectors).
- **Cross-task Generalization**: The analytical framework is successfully extended to the "superficial unlearning" task, demonstrating the generalizability of the methodology.
- **Revealing the Fundamental Limitation of Knowledge Editing**: Even when knowledge in the MLP is edited, the attention modules still retain the "memory" of the original knowledge.

## Limitations & Future Work

1. **Analysis Only, No Mitigation**: The paper deeply analyzes the mechanism of superficial editing but does not propose an effective mitigation solution.
2. **Limited Attack Types**: Only three types of attack prefixes are designed; more diverse triggers may exist in real-world scenarios.
3. **Restricted Model Scale**: The experiments concentrate on the 7B-14B scale, and whether larger models exhibit the same behavior remains to be verified.
4. **Potential Defense Directions**: Based on the discovered mechanism, future work could attempt joint editing or regularization on late-layer attention heads.

## Related Work & Insights

- Complementary to "locate-and-edit" methods like ROME/MEMIT: these methods only edit the MLP, whereas this paper reveals that the attention modules must also be considered.
- Inspires the development of "deep editing" methods: simultaneously editing MLPs and relevant attention heads to achieve more thorough knowledge updates.
- The logit lens and SVD analysis methodologies can be extended to other interpretability research scenarios.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The concept of "superficial editing" is novel, and the attack probe design is simple yet effective, revealing the fundamental flaws of knowledge editing at the mechanistic level.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — A comprehensive evaluation across three models, two datasets, and seven editing methods, with ablation studies drilling down to the level of attention heads and SVD vectors.
- **Writing Quality**: ⭐⭐⭐⭐ — Highly logical structure with a progressive layout from phenomenon to hypothesis to validation, supported by rich charts.
- **Value**: ⭐⭐⭐⭐ — Provides an important warning to the knowledge editing field, highlighting the deficiencies of current evaluation frameworks and the fundamental limitations of editing methods.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] SAKE: Steering Activations for Knowledge Editing](sake_steering_activations_for_knowledge_editing.md)
- [\[ACL 2025\] Efficient Knowledge Editing via Minimal Precomputation](efficient_knowledge_editing.md)
- [\[ACL 2025\] ScEdit: Script-based Assessment of Knowledge Editing](scedit_script-based_assessment_of_knowledge_editing.md)
- [\[ACL 2025\] Context-Robust Knowledge Editing for Language Models](context-robust_knowledge_editing_for_language_models.md)
- [\[ACL 2025\] CompKe: Complex Question Answering under Knowledge Editing](compke_complex_question_answering_under_knowledge_editing.md)

</div>

<!-- RELATED:END -->
