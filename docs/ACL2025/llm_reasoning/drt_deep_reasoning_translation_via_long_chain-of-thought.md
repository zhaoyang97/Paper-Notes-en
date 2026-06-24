---
title: >-
  [Paper Note] DRT: Deep Reasoning Translation via Long Chain-of-Thought
description: >-
  [ACL 2025][Reasoning][Machine Translation] This work introduces long CoT reasoning into machine translation by establishing a multi-agent framework (Translator $\to$ Advisor $\to$ Evaluator) to iteratively refine literary translations containing metaphors and similes. It synthesizes a 22K long-thought translation training dataset, and the resulting DRT-14B model outperforms large models such as QwQ-32B and DeepSeek-R1-Distill-32B in literary translation.
tags:
  - "ACL 2025"
  - "Reasoning"
  - "Machine Translation"
  - "Chain-of-Thought"
  - "Literary Translation"
  - "Multi-Agent"
  - "Metaphor Translation"
date: 2026-05-08
content_hash: 8d30ce301c03bdf7
---

# DRT: Deep Reasoning Translation via Long Chain-of-Thought

**Conference**: ACL 2025  
**arXiv**: [2412.17498](https://arxiv.org/abs/2412.17498)  
**Code**: [https://github.com/krystalan/DRT](https://github.com/krystalan/DRT) (Yes, including data and models)  
**Area**: LLM Reasoning  
**Keywords**: Machine Translation, Chain-of-Thought, Literary Translation, Multi-Agent, Metaphor Translation

## TL;DR
This work introduces long CoT reasoning into machine translation by establishing a multi-agent framework (Translator $\to$ Advisor $\to$ Evaluator) to iteratively refine literary translations containing metaphors and similes. It synthesizes a 22K long-thought translation training dataset, and the resulting DRT-14B model outperforms large models such as QwQ-32B and DeepSeek-R1-Distill-32B in literary translation.

## Background & Motivation

**Background**: While o1-like models have achieved breakthroughs in math and coding reasoning through long CoT, the value of long-thought reasoning has not been systematically explored in the field of translation. Existing translation models mostly rely on single-pass generation, lacking an iterative refinement process.

**Limitations of Prior Work**: (a) Literary sentences containing metaphors and similes cannot be properly translated via literal translation, requiring instead a deep understanding of rhetorical intent followed by free translation; (b) General DeepSeek-R1 distilled models suffer from incompatible reasoning capabilities when applied to translation tasks (e.g., DeepSeek-R1-Distill-Qwen-7B achieves a GEA score of only 43.66); (c) There is a lack of large-scale, long-thought translation training data.

**Key Challenge**: Literary translation requires deep semantic understanding and multi-step deliberation, but the one-step generation paradigm of existing models cannot capture this iterative refinement process.

**Goal**: (a) To construct a long-thought training dataset for literary translation scenarios; (b) To train a specialized DRT model, enabling small models to achieve high-quality literary translation.

**Key Insight**: Translating metaphors and similes in literature naturally requires multi-step reasoning: "comprehend $\to$ attempt $\to$ receive criticism $\to$ improve", which aligns perfectly with the essence of long-thought reasoning.

**Core Idea**: Generate long-thought training data for literary translation using multi-agent iterative refinement, and then fine-tune LLMs to internalize this iterative deliberation process.

## Method

### Overall Architecture
A three-step pipeline: (1) **Literary Mining**: Filtering 63K sentences featuring metaphors/similes that are poorly translated literally from 400 English books in Project Gutenberg; (2) **Multi-Agent Iterative Refinement**: Collaboration among a Translator, Advisor, and Evaluator to progressively improve translations until they meet quality standards; (3) **Long-thought Reorganization**: Eliminating invalid iterations and utilizing GPT-4o to restructure multi-agent dialogues into coherent self-reflection narratives, ultimately yielding 22,264 training samples.

### Key Designs

1. **Literary Sentence Filtering (Two-Stage Filtering)**:

    - **Function**: Filtering sentences requiring deep translation from a corpus of 577.6K sentences.
    - **Mechanism**: Utilizing Qwen2.5-72B-Instruct to make two judgments—Q1: Does it contain metaphors/similes? (Keep "yes") Q2: Can a literal translation satisfy a native speaker? (Keep "no").
    - **Design Motivation**: Only sentences that truly require deliberation are worth applying long-thought reasoning to, whereas ordinary sentences can be translated literally.

2. **Multi-Agent Iterative Refinement**:

    - **Function**: Three agents collaborate to iteratively improve translations.
    - **Mechanism**: Phase 1 – The Translator performs keyword-level translation (decomposing sub-problems); Phase 2 – Generating the initial full translation $t^0$; Phase 3 – Iterative loop: the Advisor evaluates $t^{k-1}$ and provides feedback $f^{k-1}$, the Evaluator assigns a score $s^{k-1}$, and the Translator generates a refined translation $t^k$ based on the feedback and score. This loop stops when the score threshold or maximum iteration count is reached.
    - **Design Motivation**: Simulating the deliberation process of human translators—initial translation $\to$ review $\to$ revision $\to$ re-review.

3. **Long-thought Reorganization**:

    - **Function**: Converting multi-agent dialogues into a coherent, long-thought format that a single model can learn.
    - **Mechanism**: Eliminating invalid iteration rounds where the score does not improve, filtering out samples with fewer than 3 valid refinement steps, and using GPT-4o to rewrite multi-agent dialogues into coherent self-reflection narratives. The final output is chosen as the translation with the highest score (not necessarily the last round).
    - **Design Motivation**: The multi-agent dialogue format is unsuitable for SFT of a single model, requiring standardization into a "thought process $\to$ final translation" format.

### Dataset Statistics
- 22,264 samples (19,264 training / 1,000 validation / 2,000 testing)
- Average thought tokens: 527.64, average refinement steps: 4–5 steps
- 73.22% of samples contain at least 3 refinement steps

### Loss & Training
- Full-parameter SFT based on Qwen2.5-7-7B/14B and LLaMA-3.1-8B.
- Utilizing the LLaMA-Factory framework, with vLLM accelerating inference.

## Key Experimental Results

### Main Results (English $\to$ Chinese Literary Translation)

| Model | GEA ↑ | GRF ↑ | CometKiwi ↑ | BLEU ↑ |
|------|-------|-------|-------------|--------|
| Qwen2.5-14B-Instruct | 70.86 | 84.74 | 72.01 | 30.23 |
| QwQ-32B-Preview | 75.50 | 86.31 | 71.48 | 27.46 |
| DeepSeek-R1-Distill-Qwen-32B | 71.88 | 84.78 | 71.93 | 29.36 |
| Qwen2.5-14B-SFT (w/o CoT) | 74.53 | 85.66 | 72.08 | **37.63** |
| **DRT-14B** | **77.41** | **87.19** | **72.11** | 36.46 |

DRT-14B significantly outperforms 32B-scale models on reference-free metrics (GEA, GRF).

### Human Evaluation (200 samples, Best-Worst Scaling)

| Model | Fluency | Semantic Accuracy | Literariness |
|------|--------|-----------|--------|
| Qwen2.5-14B-Instruct | -0.353 | -0.363 | -0.442 |
| QwQ-32B-Preview | -0.063 | 0.022 | -0.007 |
| Qwen2.5-14B-SFT | 0.103 | 0.108 | 0.087 |
| **DRT-14B** | **0.313** | **0.233** | **0.362** |

DRT-14B leads substantially in literariness (0.362 vs. 0.087/0.007).

### Ablation Study

| Configuration | GEA | Description |
|------|-----|------|
| DRT-7B | 75.05 | Full model |
| Qwen2.5-7B-SFT | 72.29 | w/o long-thought, drop 2.76 |
| DRT-14B | 77.41 | Full model |
| Qwen2.5-14B-SFT | 74.53 | w/o long-thought, drop 2.88 |

### Key Findings
- **Long-thought reasoning significantly improves reference-free metrics but may lower BLEU**: DRT-14B achieves GEA +2.88 but BLEU -1.17, because reasoning leads to a more liberal translation that deviates from the reference translation.
- **Evaluator accuracy of 92.5%**: Far exceeding CometKiwi (56%), demonstrating the effectiveness of LLM-as-evaluator in literary translation evaluation.
- **Inference cost: 12× slower**: Long-thought translation is 11.9–13.9 times slower than standard translation, making it suitable only for scenarios with high-quality demands.
- **Diminishing returns in refinement steps**: The largest modification occurs from Step 0 $\to$ 1 (21.44 characters), with subsequent steps showing progressively smaller changes.

## Highlights & Insights
- **The o1 Paradigm in Translation**: This work is the first to systematically introduce long CoT reasoning into translation, demonstrating that reasoning is indeed beneficial in literary translation scenarios that require deep semantic understanding.
- **Multi-agent Dialogue to Single-model Long-thought Data Conversion**: Generating high-quality refinement processes using multi-agents first, and then reorganizing them into formats trainable for a single model. This data-synthesis paradigm can be transferred to other tasks requiring iterative refinement (such as academic writing and code review).
- **14B Model Outperforming 32B**: DRT-14B exceeds QwQ-32B and DeepSeek-R1-Distill-32B across multiple metrics, indicating that domain-specific long-thought data is more crucial than general reasoning capabilities.

## Limitations & Future Work
- **English-to-Chinese only**: Other language pairs have not been evaluated.
- **Applicable only to literary translation**: Ordinary translation does not require long-thought reasoning, making the 12× inference cost impractical.
- **Unreliable automatic evaluation**: BLEU and COMET exhibit low correlation for literary translation, leading to a heavy reliance on human evaluation.
- **High data-synthesis cost**: Involves multi-agent iteration combined with GPT-4o restructuring.
- Potential improvements: (a) Expanding to other translation scenarios requiring deliberation (e.g., legal or medical); (b) Training a lightweight evaluator to replace the 72B model.

## Related Work & Insights
- **vs. Marco-O1**: Marco-O1 is a general o1-style reasoning model and performs moderately on translation (GEA 64.24); in contrast, DRT employs specialized data and training tailored for translation.
- **vs. DeepSeek-R1 Distillation**: General-purpose reasoning distillation models even underperform compared to the baselines on translation tasks (e.g., DeepSeek-R1-Distill-Qwen-7B achieves a GEA of only 43.66), indicating that reasoning capabilities cannot be directly transferred to translation.
- **vs. GPT-4o**: GPT-4o (GEA 71.88) < DRT-14B (77.41), demonstrating that domain-specialized smaller models can outperform general large models.

## Rating
- Novelty: ⭐⭐⭐⭐ First work to systematically apply long CoT reasoning to translation, featuring a cleverly designed data-synthesis pipeline.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-dimensional validation combining automatic metrics, human evaluation, ablation studies, and comparisons with commercial models.
- Writing Quality: ⭐⭐⭐⭐ Clear data-synthesis process and comprehensive evaluation.
- Value: ⭐⭐⭐⭐ Practical contributions to both literary translation and the long-thought paradigm, with both data and models open-sourced.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Can Large Language Models Detect Errors in Long Chain-of-Thought Reasoning?](can_large_language_models_detect_errors_in_long_chain-of-thought_reasoning.md)
- [\[ACL 2025\] Unlocking General Long Chain-of-Thought Reasoning Capabilities of Large Language Models via Representation Engineering](glore_long_cot_representation.md)
- [\[ACL 2025\] Improving Chain-of-Thought Reasoning via Quasi-Symbolic Abstractions](improving_chain-of-thought_reasoning_via_quasi-symbolic_abstractions.md)
- [\[ACL 2025\] Unveiling the Key Factors for Distilling Chain-of-Thought Reasoning](unveiling_the_key_factors_for_distilling_chain-of-thought_reasoning.md)
- [\[ACL 2025\] Enhancing Chain-of-Thought Reasoning with Critical Representation Fine-tuning](enhancing_chain-of-thought_reasoning_with_critical_representation_fine-tuning.md)

</div>

<!-- RELATED:END -->
