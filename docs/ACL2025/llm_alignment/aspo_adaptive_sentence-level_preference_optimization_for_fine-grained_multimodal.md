---
title: >-
  [Paper Note] ASPO: Adaptive Sentence-Level Preference Optimization for Fine-Grained Multimodal Reasoning
description: >-
  [ACL 2025][LLM Alignment][DPO] Refines the granularity of preference optimization in DPO from the response level to the sentence level. By dynamically computing adaptive reward weights for each sentence based on image-text similarity and textual perplexity, it achieves average improvements of 2.57/2.87/1.98 points on LLaVA-1.5-7B/13B and InstructBLIP-13B, respectively, while significantly reducing hallucination rates.
tags:
  - "ACL 2025"
  - "LLM Alignment"
  - "DPO"
  - "Sentence-Level Preference"
  - "Adaptive Reward"
  - "Multimodal Reasoning"
  - "Hallucination Mitigation"
date: 2026-05-08
content_hash: 8c2105cf5e923c59
---

# ASPO: Adaptive Sentence-Level Preference Optimization for Fine-Grained Multimodal Reasoning

**Conference**: ACL 2025  
**arXiv**: [2505.19100](https://arxiv.org/abs/2505.19100)  
**Code**: None  
**Area**: Multimodal LLMs / Preference Optimization  
**Keywords**: DPO, Sentence-Level Preference, Adaptive Reward, Multimodal Reasoning, Hallucination Mitigation

## TL;DR

Refines the granularity of preference optimization in DPO from the response level to the sentence level. By dynamically computing adaptive reward weights for each sentence based on image-text similarity and textual perplexity, it achieves average improvements of 2.57/2.87/1.98 points on LLaVA-1.5-7B/13B and InstructBLIP-13B, respectively, while significantly reducing hallucination rates.

## Background & Motivation

**Growing hallucination issues in multimodal large language models.** SFT-aligned MLLMs are prone to hallucinations in long responses, and increasing the probability of preferred outputs concurrently increases the probability of dis-preferred outputs. Although DPO is simple and efficient as a mainstream alignment method, its fundamental limitation is that it optimizes preferences over the entire response in a binary fashion—without distinguishing which sentences in the response are correct and which are incorrect.

**Noisy data exacerbates this issue.** In machine-generated preference data, sentences in the chosen response may be partially correct and partially incorrect. Conventional DPO treats all sentences with equal weights, assigning the same reward weight to both correct and incorrect sentences, which causes the model to converge to suboptimal solutions on noisy, semi-correct data.

**Existing fine-grained methods rely on external resources.** RLHF-V requires human annotations for paragraph-level correction, FiSAO demands an extra visual encoder for verification, and TLDR requires training an additional reward model. These methods are either costly or introduce extra parameters and models. **The core idea of ASPO is to leverage the model's own predictions to evaluate the quality of each sentence**, requiring no external models, APIs, or additional labeled data, thus achieving fine-grained preference optimization with zero extra cost.

## Method

### Overall Architecture

ASPO introduces a sentence-level adaptive reward mechanism on top of standard DPO. Training pipeline: For each preference pair $(x, y_c, y_r)$, the chosen response $y_c$ is split into a sequence of sentences $\{s_1, s_2, \ldots, s_n\}$. Two features (image-text similarity and textual perplexity) are computed for each sentence and fused into an adaptive weight $w_i$. Finally, the objective is optimized using weighted sentence-level implicit rewards instead of the original response-level implicit rewards.

### Key Designs

1. **Image-Text Similarity**:
    - **Function**: Measures the semantic relevance of each sentence to the input image
    - **Mechanism**: Uses CLIP to compute the cosine similarity between each sentence $s_i$ and the image $x$ as $S_i = \text{cosine}(s_i, x)$, followed by min-max normalization to obtain $S'_i \in [0, 1]$
    - **Design Motivation**: Sentences highly correlated with the image are more likely to be correct descriptions, while lowly correlated sentences are more likely to be hallucinations. Assigning larger reward weights to highly relevant sentences via image-text similarity mitigates the impact of hallucinations.

2. **Textual Perplexity**:
    - **Function**: Measures the model's prediction confidence for each sentence
    - **Mechanism**: Computes the conditional perplexity of each sentence as $PPL_i = \exp(-\frac{1}{N}\sum_{j=M+1}^{M+N}\log P(w_j|x, w_{<j}))$, negates it, and applies min-max normalization to obtain $PPL'_i$ (low perplexity = high confidence = high weight)
    - **Design Motivation**: Sentences with higher model confidence are more likely to be correct; perplexity reflects the model's internal certainty about the sentence.

3. **Adaptive Weight Fusion and Reward Scaling**:
    - **Function**: Fuses weights from the two dimensions and normalizes them
    - **Mechanism**: $w_i = \alpha S'_i + (1-\alpha) PPL'_i$, where $\alpha$ balances the contribution of the two metrics. The final adaptive implicit reward margin is defined as $\mathcal{M}^* = \frac{R_c}{R_c^*}\sum_{i=1}^{K}\beta(1+w_i)\log\frac{\pi_\theta(s_i^c|x)}{\pi_{ref}(s_i^c|x)} - \beta\log\frac{\pi_\theta(y_r|x)}{\pi_{ref}(y_r|x)}$
    - **Design Motivation**: The normalization factor $R_c/R_c^*$ prevents long responses from disproportionately benefiting from an increased total weight. When a response contains only one sentence, $w_i$ normalizes to 0, and ASPO degenerates to standard DPO.

### Loss & Training

The loss function is formulated as $\mathcal{L}_{ASPO} = -\mathbb{E}_\mathcal{D}[\log\sigma(\mathcal{M}^*)]$, which is structurally identical to DPO but replaces the original margin with the adaptive reward margin. Preference data is generated via the SeVa pipeline: for approximately 20K instructions sampled from LLaVA-Instruct-150K, diffusion noise is added to the images. The original and noisy images are used to generate chosen and rejected responses, respectively. Identical pairs are filtered out, resulting in about 16K preference pairs. The diffusion noise steps are set to 500.

## Key Experimental Results

### Main Results (LLaVA-1.5-7B Base)

| Benchmark | LLaVA-1.5 | +DPO | +ASPO | Gain (vs DPO) |
|------|-----------|------|-------|-------------|
| MMVet | 30.5 | 33.3 | **35.3** | +2.0 |
| MMB-D | 64.3 | 64.7 | **65.6** | +0.9 |
| LLaVA-W | 63.4 | 65.7 | **75.7** | +10.0 |
| SQA-I | 66.8 | 66.4 | **67.7** | +1.3 |
| POPE | 85.9 | 86.2 | **86.6** | +0.4 |
| SHR↓ | 36.7 | 40.1 | **33.9** | -6.2 |
| Average (excl. SHR) | 62.59 | 63.12 | **65.16** | +2.04 |

### Comparison with Other Preference Optimization Methods (LLaVA-1.5-7B)

| Method | Granularity | MMVet | MMB-D | LLaVA-W | SQA-I | POPE | Average |
|------|------|-------|-------|---------|-------|------|------|
| POVID | Response-level | 31.8 | 64.9 | 68.7 | 68.8 | 86.9 | 64.22 |
| CSR iter-3 | Sentence-level | 33.9 | 65.4 | 71.1 | 70.7 | 85.9 | 65.40 |
| FiSAO | Token-level | 30.7 | 64.8 | - | 69.3 | 85.7 | - |
| **ASPO** | **Sentence-level** | **35.3** | **65.6** | **75.7** | 67.7 | **86.6** | **66.18** |

### Ablation Study

| Configuration | Average Score | Description |
|------|--------|------|
| ASPO (Full) | 65.16 | Image-text similarity + Perplexity |
| ASPO-S (Similarity only) | 64.91 | Removes perplexity, -0.25 |
| ASPO-P (Perplexity only) | 65.00 | Removes similarity, -0.16 |
| DPO (Response-level) | 63.12 | No fine-grained weights, -2.04 |

### Key Findings

- ASPO achieves the largest improvement (+10.0) on LLaVA-W, which evaluates the quality of open-ended long responses—precisely the scenario where fine-grained optimization yields the greatest benefits.
- SHR (hallucination rate) drops from 40.1 to 33.9, showing that ASPO effectively reduces hallucinations.
- The two dimensions (similarity and perplexity) are complementary; removing either degrades performance.
- Significant improvements are also observed on LLaVA-1.5-13B (average 65.93 $\rightarrow$ 68.80), validating the scalability of ASPO.
- It is also effective on InstructBLIP-13B (Q-Former architecture) (43.86 $\rightarrow$ 45.84), indicating that it does not rely on a specific architecture.

## Highlights & Insights

- **Sentences are the "sweet spot" granularity for preference optimization**: more precise than response-level and semantically more complete than token-level. ASPO outperforms token-level FiSAO and most response-level methods.
- **Zero-cost self-supervised signal**: Image-text similarity leverages the existing CLIP model, and perplexity utilizes predictions from the model itself—eliminating the need to train extra models or call paid APIs.
- **Elegant length normalization design**: The $R_c/R_c^*$ factor ensures that long responses do not disproportionately receive larger total weights due to containing more sentences.
- **Graceful degradation properties**: Responses with a single sentence automatically degenerate to standard DPO, maintaining compatibility with the baseline method.

## Limitations & Future Work

- **High sensitivity to sentence segmentation quality**: Incorrect sentence boundaries lead to inaccurate reward allocation; the paper does not discuss the robustness of segmentation.
- **Only validated in multimodal scenarios**: Image-text similarity weight relies heavily on visual information; pure NLP tasks would require alternative formulations.
- **Sensitivity to the $\alpha$ hyperparameter**: How to optimally balance the weights of image-text similarity and perplexity remains under-explored.
- **No sentence-level weighting for rejected responses**: Adaptive weights are only assigned to sentences in the chosen response, while the rejected response is still treated as a single entity.

## Related Work & Insights

- **vs CSR (Iterative sentence-level)**: CSR requires iterative multi-turn candidate response generation and external evaluation; ASPO achieves superior performance within a single training run.
- **vs RLHF-V (Human paragraph-level correction)**: Relies on expensive human annotations, whereas ASPO is fully self-supervised.
- **vs FiSAO (Token-level)**: Token granularity is too fine and leads to semantic fragmentation; sentence-level optimization preserves semantic integrity.
- **vs MDPO (Response-level + Image-preferred)**: Resolves unconditional preferences but still operates at a coarse granularity; ASPO is superior in both granularity and signal quality.

## Rating

- Novelty: ⭐⭐⭐⭐ Sentence-level adaptive reward is an effective improvement over DPO, with a well-designed dual-dimensional approach of image-text similarity and perplexity.
- Experimental Thoroughness: ⭐⭐⭐⭐ 3 base model architectures + 10 benchmarks + comparisons with 12 methods + ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear system diagrams, rigorous mathematical derivations, and a solid analysis of academic degradation properties.
- Value: ⭐⭐⭐⭐ Provides practical improvements to the methodology of multimodal DPO, with its zero-additional-cost being a major advantage.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Fine-grained Video Dubbing Duration Alignment with Segment Supervised Preference Optimization](fine-grained_video_dubbing_duration_alignment_with_segment_supervised_preference.md)
- [\[ACL 2025\] PRMBench: A Fine-grained and Challenging Benchmark for Process-Level Reward Models](prmbench_a_fine-grained_and_challenging_benchmark_for_process-level_reward_model.md)
- [\[ACL 2025\] Probability-Consistent Preference Optimization for Enhanced LLM Reasoning](probability-consistent_preference_optimization_for_enhanced_llm_reasoning.md)
- [\[ACL 2025\] SDPO: Segment-Level Direct Preference Optimization for Social Agents](sdpo_segment-level_direct_preference_optimization_for_social_agents.md)
- [\[ACL 2025\] T-REG: Preference Optimization with Token-Level Reward Regularization](t-reg_preference_optimization_with_token-level_reward_regularization.md)

</div>

<!-- RELATED:END -->
