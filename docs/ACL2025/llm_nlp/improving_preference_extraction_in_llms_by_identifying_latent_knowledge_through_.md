---
title: >-
  [Paper Note] Improving Preference Extraction In LLMs By Identifying Latent Knowledge Through Classifying Probes
description: >-
  [ACL 2025][LLM (Other)][LLM-as-Judge] This paper proposes using linear classifying probes combined with contrast pairs to extract the latent preference judgments of LLMs. This approach consistently outperforms traditional generative evaluation methods on LLM-as-Judge tasks, and supervised probes even exceed fine-tuned evaluators while maintaining comparable computational costs.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "LLM-as-Judge"
  - "Linear Probes"
  - "Contrast Pairs"
  - "Latent Knowledge"
  - "Preference Extraction"
date: 2026-05-08
content_hash: fa1689c55ad3d2ff
---

# Improving Preference Extraction In LLMs By Identifying Latent Knowledge Through Classifying Probes

**Conference**: ACL 2025  
**arXiv**: [2503.17755](https://arxiv.org/abs/2503.17755)  
**Code**: None (the paper states it will be released)  
**Area**: LLM/NLP  
**Keywords**: LLM-as-Judge, Linear Probes, Contrast Pairs, Latent Knowledge, Preference Extraction

## TL;DR

This paper proposes using linear classifying probes combined with contrast pairs to extract the latent preference judgments of LLMs. This approach consistently outperforms traditional generative evaluation methods on LLM-as-Judge tasks, and supervised probes even exceed fine-tuned evaluators while maintaining comparable computational costs.

## Background & Motivation

LLM-as-Judge is currently the dominant paradigm for automated evaluation, replacing human evaluation by prompting LLMs to output numerical scores or pairwise comparisons. However, this generation-based evaluation method suffers from several issues:
- Constrained decoding may introduce artifacts
- Prompts may introduce unintended biases
- Lengthy reasoning may obscure the core judgment
- Black-box methods can produce untrustworthy or factually incorrect generations
- Biases learned during pre-training can affect judgments

A key empirical observation is that the direction of "belief" or "judgment" encoded in the latent space of LLMs is linearly correlated. For example, using Llama 3.1 70B on the MT-Bench dataset, the first principal component of the difference in contrast pair embeddings can roughly distinguish which model is human-preferred. This suggests that the internal representations of LLMs contain more accurate judgment information than their generated outputs.

## Method

### Overall Architecture

The core idea of the method is to directly extract judgment signals from the latent representations (activations) of the LLM instead of prompting the LLM to "verbalize" its judgment. The specific pipeline is as follows:
1. Construct contrast pairs: For each evaluation instance, append positive and negative contrast tokens respectively.
2. Retrieve the embedding vectors under both conditions.
3. Eliminate shared features through embedding differences, highlighting judgment-related features.
4. Fit a classifier on the difference vectors to extract preferences.

### Key Designs

1. **Contrast Pair Construction**: For each evaluation instance $s_i$, construct a pair of prompts $(x_i^+, x_i^-)$. For example: "Between Choice 1 and Choice 2, the more [task] [item] is Choice 1" and "...is Choice 2". The two prompts only differ in the final contrast token, thereby controlling confounding variables.

2. **Embedding Difference Analysis**: Extract the embedding vectors $\phi(x_i^+)$ and $\phi(x_i^-)$ of the contrast tokens after the last decoder block and before the final normalization layer of the LLM. Center the embeddings to eliminate syntactic differences $\Delta_{syntax}$, leaving the knowledge/judgment difference $\Delta_{knowledge}$ as the most prominent contrasting feature.

3. **Unsupervised Probe (PCA-based)**: Perform PCA on the centered embedding differences $\{\tilde{\phi}(x_i^+) - \tilde{\phi}(x_i^-)\}$, and use the first principal component as the classification direction. This method requires no labels, assuming that the "knowledge difference" is the most prominent contrasting feature.

4. **Supervised Probe (Logistic Regression)**: Given labels, fit a classifier using logistic regression: $\mathbb{P}(x^+ \text{ true}) = \sigma(\mathbf{w}^T(\tilde{\phi}(x_i^+) - \tilde{\phi}(x_i^-)))$. This requires training the parameters of only a single linear layer.

5. **Position Bias-Corrected Baseline**: For the generative baselines, eliminate position bias by swapping the order of the two choices and averaging the predicted probabilities. However, this requires running the model twice for each question.

### Loss & Training

- Supervised probe: Standard logistic regression loss, trained on 5,000 annotated samples.
- Unsupervised probe: PCA, requiring no labels.
- Contrastive fine-tuning baselines: LoRA (rank value not detailed) and full fine-tuning.
- Activation extraction location: Embedding of the final token after the last decoder block and before the final normalization layer.

## Key Experimental Results

### Main Results

F1 scores on MT-Bench (using the majority vote with 80% human agreement as the gold standard):

| Method | Llama 3.1 70B F1 | Notes |
|------|-----------------|------|
| Pairwise comparison prompting | ~0.65 | Generative baseline |
| Unsupervised probe | ~0.80 | No labels required |
| Supervised probe | ~0.80 | Requires few labels |

Unsupervised probe vs prompting across model families (aggregated results from 6 datasets):

| Model | Unsupervised Probe F1 | Prompting F1 | Gain |
|------|------------|--------|------|
| Gemma 2 27B | ~0.85 | ~0.70 | +15% |
| Llama 3.1 70B | ~0.88 | ~0.72 | +16% |
| Qwen 2.5 72B | ~0.87 | ~0.73 | +14% |
| Mistral Large 123B | ~0.84 | ~0.70 | +14% |

### Ablation Study

| Configuration | Key Result | Notes |
|------|---------|------|
| Supervised probe vs LoRA fine-tuning | Probe is superior | Under the same training data volume |
| Supervised probe vs Full fine-tuning | Probe is superior | Across the entire Gemma 2 model size spectrum |
| Cross-dataset generalization (Unsupervised) | F1 0.70-0.99 | Extremely strong transfer capability across different tasks |
| Cross-dataset generalization (Supervised) | F1 0.58-0.99 | Supervised probe generalizes slightly worse but remains good overall |
| LLMBar adversarial robustness | Probes are significantly more robust | Smaller performance drops under adversarial prompts |

### Key Findings

- Unsupervised probes outperform calibrated prompting methods across all 6 datasets and 4 model families (with a single exception: Qwen 2.5 0.5B).
- Important finding: The performance of small models using unsupervised probes is almost always superior to that of large models using prompting, indicating that the use of large models in existing LLM-as-Judge practices might be "wasteful".
- Supervised probes further outperform unsupervised probes, and they exceed LoRA and full fine-tuning under the same training data volume.
- The advantage of probes is more pronounced in subjective tasks (text quality evaluation), while fine-tuning is more competitive in objective tasks (commonsense reasoning).
- Unsupervised probes generalize exceptionally well across different datasets (cosine similarity > 0.7), suggesting that they capture general "belief/judgment" features.
- In LLMBar adversarial testing, probing methods show a smaller performance degradation and are more robust under adversarial prompts compared to prompting methods.

## Highlights & Insights

1. **Revealing the Capability Gap**: There is a significant gap between a model's ability to express things via generation and its actual encoded capability within its latent space. Probing methods directly extract judgments from the latent space, bypassing various biases and information losses during generation.
2. **Computational Efficiency**: Each sample requires only two forward passes (the same as position-bias-corrected prompting methods) but yields significantly higher accuracy. Training supervised probes only involves logistic regression, introducing almost zero computational overhead.
3. **Scaling Efficiency**: Small models + probes > large models + prompting, which holds significant practical value for resource-constrained application scenarios.
4. **Discovery of a Universal Judgment Direction**: The excellent performance of unsupervised probes in cross-task generalization implies the existence of a universal "judgment" or "belief" direction within LLMs, which is a phenomenon worthy of deeper investigation.

## Limitations & Future Work

- This work only focuses on pairwise comparison tasks; extending it to single-annotation tasks like direct scoring (Likert scales) would require multi-class probes.
- The performance saturation point of supervised probes has not been studied—fine-tuning might catch up under extremely large training volumes.
- Vulnerability to adversarial prompts (e.g., "You are a smart professor") remains, which represents a fundamental limitation of representation probing methods.
- The choice of embedding extraction layer affects the results; the last layer is used by default but has not been systematically optimized.
- The continuous updates of LLMs are not considered—as the instruction-following capabilities of new models improve, the gap between generative methods and probing may narrow.
- The sign of the unsupervised probes cannot be determined natively—the PCA direction might be inverted, still requiring a tiny number of labels for calibration in practice.

## Related Work & Insights

- Contrastive Consistent Search (CCS) proposed by Burns et al. (2023) serves as the direct inspiration for the unsupervised probes in this paper.
- Aligned with Marks & Tegmark (2024) concerning the linear representation of factual knowledge in LLMs, this work extends probing from knowledge detection to preference judgment.
- Poses a paradigmatic challenge to logit-weighted evaluation methods like G-Eval: directly extracting information from the latent space is more effective than doing so from output probabilities.
- Provides tools for AI safety and alignment research: if a model "knows" the correct answer but does not "verbalize" it, probes can be used to detect this inconsistency.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Applying representation probing methods to LLM-as-Judge is a completely new direction with profound insights.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Highly comprehensive, covering 4 model families (0.5B-123B), 6 datasets, along with generalization and adversarial tests.
- Writing Quality: ⭐⭐⭐⭐ The theoretical framework is clear and the comparative analysis is systematic, though some mathematical formulations could be more concise.
- Value: ⭐⭐⭐⭐⭐ Reveals the capability gap in LLM evaluation with a practical and efficient method, providing direct guidance for practitioners.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] GenKnowSub: Improving Modularity and Reusability of LLMs through General Knowledge Subtraction](genknowsub_improving_modularity_and_reusability_of_llms_through_general_knowledg.md)
- [\[ACL 2025\] Identifying Reliable Evaluation Metrics for Scientific Text Revision](reliable_eval_metrics_scientific.md)
- [\[ACL 2025\] Self-Tuning: Instructing LLMs to Effectively Acquire New Knowledge through Self-Teaching](self-tuning_instructing_llms_to_effectively_acquire_new_knowledge_through_self-t.md)
- [\[ACL 2025\] Analyzing LLMs' Knowledge Boundary Cognition Across Languages Through the Lens of Internal Representations](knowledge_boundary_crosslingual.md)
- [\[ACL 2025\] FEAT: A Preference Feedback Dataset through a Cost-Effective Auto-Generation and Labeling Framework for English AI Tutoring](feat_a_preference_feedback_dataset_through_a_cost-effective_auto-generation_and_.md)

</div>

<!-- RELATED:END -->
