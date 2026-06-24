---
title: >-
  [Paper Note] Contrastive Prompting Enhances Sentence Embeddings in LLMs through Inference-Time Steering
description: >-
  [ACL 2025][LLM (Other)][sentence embedding] This paper proposes Contrastive Prompting (CP), an inference-time method that constructs an auxiliary prompt to encode non-core information of a sentence. By performing "semantic subtraction" between the hidden layer activations of the normal prompt and the auxiliary prompt during inference, it filters out irrelevant semantics like stop words, focusing the LLM sentence embeddings more on core semantics. This plug-and-play approach c…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "sentence embedding"
  - "contrastive prompting"
  - "activation steering"
  - "inference-time"
  - "LLM"
date: 2026-05-08
content_hash: 8a9529f9ef46f9f0
---

# Contrastive Prompting Enhances Sentence Embeddings in LLMs through Inference-Time Steering

**Conference**: ACL 2025  
**arXiv**: [2505.12831](https://arxiv.org/abs/2505.12831)  
**Code**: [GitHub](https://github.com/zifengcheng/CP)  
**Area**: LLM/NLP  
**Keywords**: sentence embedding, contrastive prompting, activation steering, inference-time, LLM  

## TL;DR

This paper proposes Contrastive Prompting (CP), an inference-time method that constructs an auxiliary prompt to encode non-core information of a sentence. By performing "semantic subtraction" between the hidden layer activations of the normal prompt and the auxiliary prompt during inference, it filters out irrelevant semantics like stop words, focusing the LLM sentence embeddings more on core semantics. This plug-and-play approach consistently improves the performance of various prompting methods (such as PromptEOL, CoT, and Knowledge) on STS and classification tasks.

## Background & Motivation

**Background**: Directly extracting zero-shot sentence embeddings from LLMs (without fine-tuning or extra data) is a practical direction. Existing methods utilize prompt engineering to compress sentence semantics into the hidden state of the last token, such as PromptEOL ("This sentence: '[TEXT]' means in one word:"), MetaEOL (multi-task meta-prompting), Pretended CoT (Chain-of-Thought prompting), and Knowledge (knowledge-enhanced prompting).

**Limitations of Prior Work**: Even with carefully designed prompts, the last token still encodes a significant amount of non-core information. Experiments show that even when using Knowledge prompting to emphasize "subject and action," the decoded token with the highest probability is still a stop word like "a" instead of semantic keywords. Prompt engineering essentially only alters representations **indirectly** and cannot **directly** filter out non-core information.

**Key Challenge**: Existing methods all indirectly affect the representation of the last token by changing the prefix text, lacking a mechanism to directly strip away non-core semantics in the hidden space.

**Key Insight**: Inspired by activation steering but without relying on supervised positive-negative sample pairs, the proposed method uses an auxiliary prompt ("The irrelevant information of this sentence...") to adaptively capture the non-core information activation of each sentence. It then subtracts the auxiliary activation from the normal prompt activation to achieve sentence-by-sentence adaptive "semantic subtraction."

## Method

### Overall Architecture

A three-step process:  
(1) Wrap the text in the auxiliary prompt and perform forward propagation to the $\ell$-th layer, extracting the contextualized value vector of the last token, $\mathbf{v}^{\text{aux},(\ell)}$;  
(2) Wrap the text in the normal prompt and perform forward propagation to the $\ell$-th layer, computing the contrastive vector $\Delta\mathbf{v}^\ell = \mathbf{v}^{\text{nor},(\ell)} - \mathbf{v}^{\text{aux},(\ell)}$ and replacing the value vector of the last token;  
(3) Adjust the norm of the replaced vector, and continue forward propagation to an intermediate layer to extract the sentence embedding.

### Key Designs

1. **Auxiliary Prompt Construction**: The template "The irrelevant information of this sentence: '[TEXT]' means in one word:" is designed to guide the LLM to focus on non-core information in the sentence and encode it into the last token. The auxiliary prompt only needs to propagate to lower layers (layers 5\~7), resulting in minimal computational overhead. The paper also explores variants such as "redundant information," "background," and "descriptive term," indicating that as long as the semantics refer to "non-core information," performance consistently improves, showing that the method is insensitive to the exact phrasing of the auxiliary prompt.

2. **Contrastive Activation Steering**: At the multi-head attention of the $\ell$-th layer, the contextualized value vectors of the last tokens from both the normal and auxiliary prompts are extracted to compute the semantic activation vector $\Delta\mathbf{v}^\ell = \mathbf{v}_{N_\text{nor}}^{\text{nor},(\ell)} - \mathbf{v}_{N_\text{aux}}^{\text{aux},(\ell)}$. This vector is **sentence-adaptive** (different sentences generate different contrastive vectors) and does not require positive-negative sample pairs from extra supervised data. Only the value vector of the last token is intervened, keeping other tokens unchanged.

3. **Norm Adjustment & Intermediate Embedding**: Since the norm of the vector might change significantly after intervention, two adjustment strategies are proposed—**Norm Scaling (NS)**: $\hat{\mathbf{v}} = \alpha \cdot \Delta\mathbf{v}^\ell$, which controls the intervention intensity via a scaling factor $\alpha$ (optimal value 2\~3); and **Norm Recovering (NR)**: $\hat{\mathbf{v}} = \Delta\mathbf{v}^\ell \cdot \frac{\|\mathbf{v}^{\text{nor}}\|_2}{\|\Delta\mathbf{v}^\ell\|_2}$, which recovers the original norm to preserve model stability. In addition, the intermediate layer (instead of the last layer) output is used as the embedding to further improve quality and save computation.

### Plug-and-Play Characteristic

CP is a purely inference-time intervention that can be **seamlessly combined** with any prompting method, such as PromptEOL, Pretended CoT, Knowledge, and MetaEOL, without modifying model parameters or training pipelines. For multi-prompt methods (such as CK = CoT + Knowledge average), the auxiliary prompt only needs to be propagated once to optimize all normal prompts simultaneously.

## Key Experimental Results

### STS Benchmark (LLaMA2-7B, 7-task average Spearman×100)

| Method | Avg. (Original) | +CP-NS | +CP-NR | Gain |
|------|------------|--------|--------|---------|
| PromptEOL | 70.03 | **75.27** | 75.20 | **+5.24** |
| Pretended CoT | 76.86 | **77.45** | 77.45 | +0.59 |
| Knowledge | 77.14 | **77.56** | 77.40 | +0.42 |
| CK (CoT+Know) | 78.23 | **78.68** | 78.60 | +0.45 |

### Cross-Model Generalization (Pretended CoT + CP-NS)

| Backbone | Avg. (Original) | +CP-NS | Gain |
|----------|------------|--------|------|
| LLaMA2-7B | 76.86 | **77.45** | +0.59 |
| LLaMA2-13B | 73.34 | **73.91** | +0.57 |
| LLaMA3.1-8B | 74.07 | **75.22** | +1.15 |

### Downstream Classification Tasks (LLaMA2-7B, PromptEOL + CP-NS)

| Task | Original | +CP-NS | Change |
|------|------|--------|------|
| SUBJ | 96.32 | **96.97** | +0.65 |
| TREC | 95.40 | **97.00** | +1.60 |
| MRPC | 75.19 | **77.51** | +2.32 |
| SST2 | 95.00 | **95.94** | +0.94 |
| 7-Task Avg. | 90.94 | **91.73** | +0.79 |

### Computational Overhead (Forward propagation layers, LLaMA2-7B with 32 layers total)

| Method | Without CP | With CP | Extra Overhead |
|------|-------|-------|---------|
| PromptEOL | 27 Layers (1×) | 31 Layers (1.15×) | +15% |
| Knowledge | 31 Layers (1.15×) | 37 Layers (1.37×) | +19% |
| CK (Dual Prompts) | 54 Layers (2×) | 60 Layers (2.22×) | +11% |

### Key Ablation Findings

- **Intervention Position**: Attention Head > Transformer Layer Output > FFN Output, with Attention Head being optimal (STS-B dev 82.61 vs 81.93).
- **Intervention Layer**: Layer 5 is optimal for PromptEOL, and Layer 7 is optimal for CoT/Knowledge.
- **Scaling Factor $\alpha$**: Optimal $\alpha = 2$ for PromptEOL and $\alpha = 3$ for CoT/Knowledge; values too large or too small degrade performance.
- **Decoding Probability Verification**: After applying CP, the top-1 predicted token changes from stop words (e.g., "It", "Don") to semantic keywords (e.g., "Dec", "Throw").

## Highlights & Insights

- The "semantic subtraction" concept is simple and elegant—the auxiliary prompt captures noise activations, and subtracting noise from the normal prompt yields clean semantic signals.
- It adaptively generates contrastive vectors sentence-by-sentence, which is more flexible than traditional activation steering (which requires global positive and negative sample pairs).
- PromptEOL achieves the largest gain (+5.24), as its original prompt is the simplest and contains the most non-core information; the gains of CP are proportional to the "weakness" of the baseline method.
- The auxiliary prompt is only propagated to lower layers (layers 5\~7), incurring an extra computational overhead of only 11%\~19%.

## Limitations & Future Work

- The method design assumes that the auxiliary prompt is capable of capturing "non-core information," but there is no theoretical guarantee of what "core" represents. The performance relies on the semantic complementarity between the auxiliary and normal prompts.
- Evaluation is conducted only on English STS and classification tasks; cross-lingual efficacy remains unverified.
- Although robust, the design of the auxiliary prompt still retains some subjectivity; prompts pointing to "sentiment" or "entities" conversely degrade performance.
- The optimal intervention layer and scaling factor require a grid search on the validation set, as different prompts lead to different optimal hyperparameters.
- No equal-scale comparison with fine-tuning methods (such as SimCSE-LLM, etc.).
- The potential of combining CP with fine-tuning methods is not explored—whether CP, as an inference-time intervention, can stack cumulative gains on fine-tuned models remains an open question.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The "semantic subtraction" concept using auxiliary prompts + activation contrast is simple and novel, extending activation steering from requiring supervised data to unsupervised sentence-by-sentence adaptiveness.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers 4 baseline methods, 3 LLM backbones, dual-dimensional evaluations on STS and classification, and comprehensive ablations on intervention position, layers, scaling factors, and auxiliary prompts.
- **Writing Quality**: ⭐⭐⭐⭐ The motivation is intuitively demonstrated through decoding probability experiments. The three-step process of the method is clear, with rich figures and tables.
- **Value**: ⭐⭐⭐⭐ Plug-and-play without training, with extremely small overhead (+15%). It can be directly integrated into sentence embedding pipelines of production systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Token Prepending: A Training-Free Approach for Eliciting Better Sentence Embeddings from LLMs](token_prepending_training_free.md)
- [\[ACL 2025\] Nudging: Inference-time Alignment of LLMs via Guided Decoding](nudging_inference_time_alignment.md)
- [\[ACL 2025\] Enhancing Input-Label Mapping in In-Context Learning with Contrastive Decoding](enhancing_input-label_mapping_in_in-context_learning_with_contrastive_decoding.md)
- [\[ACL 2025\] Better Embeddings with Coupled Adam](better_embeddings_with_coupled_adam.md)
- [\[ACL 2025\] Steering off Course: Reliability Challenges in Steering Language Models](steering_off_course_reliability_challenges_in_steering_language_models.md)

</div>

<!-- RELATED:END -->
