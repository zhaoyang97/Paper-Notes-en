---
title: >-
  [Paper Note] Towards Better Code Understanding in Decoder-Only Models with Contrastive Learning
description: >-
  [AAAI 2026][Contrastive Learning] This paper proposes CL4D, a contrastive learning framework that continues pre-training decoder-only code generation models, enabling them to extract effective code representations and achieve performance on par with or superior to encoder-only models of comparable scale on code understanding tasks such as code search and clone detection.
tags:
  - AAAI 2026
  - Contrastive Learning
  - Decoder-Only Models
  - Code Understanding
  - Code Search
  - Clone Detection
date: 2026-05-08
content_hash: 010e2de68f6e0f26
---

# Towards Better Code Understanding in Decoder-Only Models with Contrastive Learning

**Conference**: AAAI 2026
**arXiv**: [2406.12326](https://arxiv.org/abs/2406.12326)
**Code**: [https://github.com/JiayiLin1024/CL4D](https://github.com/JiayiLin1024/CL4D)
**Area**: Self-Supervised Learning / Code Understanding
**Keywords**: Contrastive Learning, Decoder-Only Models, Code Understanding, Code Search, Clone Detection

## TL;DR
This paper proposes CL4D, a contrastive learning framework that continues pre-training decoder-only code generation models, enabling them to extract effective code representations and achieve performance on par with or superior to encoder-only models of comparable scale on code understanding tasks such as code search and clone detection.

## Background & Motivation
Large-scale decoder-only code generation models (e.g., CodeGen, StarCoder, Code Llama) have achieved remarkable results on code generation tasks, with model sizes reaching up to 70B parameters. However, these models fall notably short of specialized encoder-only models (e.g., CodeBERT, UniXcoder, CodeSage — the largest being only 1.5B parameters) on code understanding tasks such as code search and clone detection.

The root cause lies in the following tension: decoder-only models possess more parameters and have been trained on more code data, yet their autoregressive training objective and unidirectional attention mechanism constrain their representational capacity on understanding tasks. At the same time, training an encoder-only model of equivalent scale from scratch is computationally prohibitive.

The paper's starting point is: **Can the knowledge of existing decoder-only code generation models be reused — without retraining — through efficient continued pre-training to enhance code understanding ability?** The core idea is to leverage contrastive learning (CL4D) to compensate for the representational deficiencies of the decoder-only architecture.

## Method

### Overall Architecture
CL4D is a contrastive learning continued pre-training framework for decoder-only models. The overall pipeline consists of: (1) constructing cross-lingual (query, code) training pairs from The Stack dataset; (2) encoding inputs with a dual-encoder architecture (two decoder modules sharing weights); and (3) training the model to learn discriminative representations via contrastive learning with in-batch negatives and hard negatives.

### Key Designs
1. **Data Construction**:

    - Function: Extracts code from six programming languages (Python, Java, Go, PHP, JavaScript, Ruby) from The Stack dataset.
    - Mechanism: Uses Tree-Sitter to extract the first line of a function's docstring as the query and the corresponding function body as the code, forming bimodal (query, code) pairs.
    - Design Motivation: Enhances the model's ability to distinguish natural language from programming language; leverages existing code knowledge from pre-trained models; requires only millions of samples for continued pre-training.

2. **Code Representation Extraction**:

    - Function: Identifies the optimal strategy for extracting code representations from decoder-only models.
    - Mechanism: Investigates two approaches — (1) using the embedding of the last token; (2) using the mean of all token embeddings — along with the effect of left padding versus right padding.
    - Design Motivation: Unlike encoder-only models that use a [CLS] token, the unidirectional attention in decoder-only models means only the last token aggregates full context. Experiments show that **right padding + mean pooling over all token embeddings** is the optimal strategy.

3. **Contrastive Learning Training**:

    - Function: Enhances the representational capacity of decoder-only models through contrastive learning.
    - Mechanism: Randomly samples $n$ (query, code) pairs to form a batch, treating the paired code as the positive sample and all other codes in the batch as negatives. Hard negatives — code snippets that are close to the query in representation space but semantically different — are additionally incorporated, pre-filtered using UniXcoder.
    - Design Motivation: The next-token prediction objective inherently limits representational power. Contrastive learning pulls semantically similar samples closer and pushes dissimilar ones apart, while allowing mixed multi-language batches to learn a unified semantic space.

### Loss & Training
The loss function is an InfoNCE variant incorporating in-batch negatives and hard negatives:

$$\mathcal{L} = -\log\frac{\exp(s(q,c^+)/\tau)}{\sum_{i=1}^n \exp(s(q,c_i)/\tau) + \exp(s(q,c^h)/\tau)}$$

where the temperature coefficient is $\tau = 0.05$ and $s(q,c)$ denotes cosine similarity. Training uses the AdamW optimizer with a learning rate of 2e-5, batch size of 64, on 8×A100 GPUs for 2 epochs. The longest training run (phi-1) takes approximately 3 days.

## Key Experimental Results

### Main Results

| Model | CSN (MRR) | CoSQA (MRR) | POJ-104 (MAP) |
|-------|-----------|-------------|---------------|
| UniXcoder (125M, encoder) | 74.40 | 70.1 | 89.56 |
| CodeSage (1.3B, encoder) | 75.80 | 68.0 | 87.70 |
| CodeGPT (125M) + CL4D | 70.20 | 69.0 | 87.96 |
| CodeGen (350M) + CL4D | 73.30 | 71.5 | 89.68 |
| phi-1 (1.3B) + CL4D | 75.18 | 72.8 | 92.72 |
| DeepSeek-Coder (1.3B) + CL4D | **77.57** | **71.9** | 89.71 |

CL4D enables decoder-only models to surpass encoder-only models of comparable scale by approximately 2% on most tasks.

### Ablation Study

| Configuration | CSN (MRR) | CoSQA (MRR) | POJ-104 (MAP) |
|---------------|-----------|-------------|---------------|
| CL4D (full) | 72.00 | 51.20 | 45.84 |
| − Hard Negative | 70.80 | 50.40 | 44.65 |
| − In-Batch Negative | 1.42 | 0.45 | 13.20 |

Removing in-batch negatives causes a dramatic performance collapse, confirming that contrastive learning is the critical component. Hard negatives contribute an additional ~1.5% improvement.

### Key Findings
- **Substantial zero-shot improvement**: CL4D improves the zero-shot performance of decoder-only models by 40%–76% (maximum gain: 75.90%), even matching fine-tuned encoder-only results on CSN.
- **Model scale effect**: Larger decoder-only models achieve better understanding performance after CL4D, suggesting that scaling laws apply to code understanding as well.
- **Representation space visualization**: t-SNE visualizations demonstrate that CL4D significantly improves the clustering of semantically similar code in representation space, whereas the original decoder-only model produces highly scattered representations.

## Highlights & Insights
- This work is the first systematic investigation into adapting decoder-only code generation models for code understanding tasks, bridging the gap between generation and understanding.
- The finding that right padding + average pooling is the optimal representation extraction strategy for decoder-only models has broad applicability.
- The results demonstrate that the decoder-only architecture has the potential to unify code understanding and generation, eliminating the need to maintain two separate model families.
- The approach is low-cost: continued pre-training on a small-scale code corpus for only a few days yields substantial improvements in understanding capability.

## Limitations & Future Work
- Evaluation is limited to code search and clone detection; additional understanding tasks such as code summarization, defect detection, and code translation are not assessed.
- Hard negative construction relies on UniXcoder for pre-computation, introducing additional preprocessing overhead, and the quality of hard negatives is bounded by UniXcoder's own representational capacity.
- More parameter-efficient fine-tuning methods (e.g., LoRA, Adapters) are not explored; full-parameter continued pre-training still requires several days on 8×A100 GPUs.
- Validation on larger-scale decoder-only models (e.g., 7B, 13B, 70B) is absent, leaving the continuation of the scaling trend unconfirmed.
- Training data covers only six programming languages; generalization to low-resource languages is not evaluated.
- The effect of continued pre-training on the model's generation performance is not studied — it remains unclear whether understanding gains come at the cost of generation quality.

## Related Work & Insights
- Similar to CodeSage in using contrastive learning to improve code representations, this work focuses specifically on adapting the decoder-only architecture.
- The methodology is generalizable to other domains: any setting where a powerful generative model exists but understanding ability is insufficient (e.g., natural language text, protein sequences) could benefit from a similar contrastive learning adaptation strategy.
- This work provides a viable path toward building a unified large code model that supports both generation and understanding.
- It complements the scaling law study of CoLSBERT: while CoLSBERT demonstrates the effectiveness of scaling encoder-only models, CL4D shows that existing large-scale decoder-only models can be directly repurposed for understanding tasks.

## Rating
- Novelty: ⭐⭐⭐⭐ (The idea is clear but the approach is relatively straightforward; contrastive learning itself is not novel.)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Multiple models, tasks, and ablations, though the range of task types is limited.)
- Writing Quality: ⭐⭐⭐⭐ (Well-structured, thorough argumentation, rich figures and tables.)
- Value: ⭐⭐⭐⭐ (Paves the way for unifying generation and understanding in decoder-only models; strong practical utility.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] ReCode: Updating Code API Knowledge with Reinforcement Learning](recode_updating_code_api_knowledge_with_reinforcement_learning.md)
- [\[AAAI 2026\] EquaCode: A Multi-Strategy Jailbreak Approach for Large Language Models via Equation Solving and Code Completion](equacode_a_multi-strategy_jailbreak_approach_for_large_language_models_via_equat.md)
- [\[AAAI 2026\] SPAN: Benchmarking and Improving Cross-Calendar Temporal Reasoning of Large Language Models](span_benchmarking_and_improving_cross-calendar_temporal_reasoning_of_large_langu.md)
- [\[AAAI 2026\] DiffBench Meets DiffAgent: End-to-End LLM-Driven Diffusion Acceleration Code Generation](diffbench_meets_diffagent_end-to-end_llm-driven_diffusion_ac.md)
- [\[AAAI 2026\] Extracting Events Like Code: A Multi-Agent Programming Framework for Zero-Shot Event Extraction](extracting_events_like_code_a_multi-agent_programming_framework_for_zero-shot_ev.md)

</div>

<!-- RELATED:END -->
