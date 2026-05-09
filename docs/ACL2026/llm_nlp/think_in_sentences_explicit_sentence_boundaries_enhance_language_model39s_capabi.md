---
title: >-
  [Paper Note] Think in Sentences: Explicit Sentence Boundaries Enhance Language Model's Capabilities
description: >-
  [ACL 2026][LLM/NLP][sentence boundaries] This paper proposes inserting delimiter tokens at sentence boundaries in LLM inputs to implement a "think-in-sentences" reasoning paradigm via both ICL and SFT. The approach yields consistent improvements across models ranging from 7B to 600B parameters (GSM8k +7.7%, DROP +12.5%) with negligible additional computational overhead.
tags:
  - ACL 2026
  - LLM/NLP
  - sentence boundaries
  - delimiters
  - in-context learning
  - supervised fine-tuning
  - free lunch
date: 2026-05-08
content_hash: 11f3ca62d819e1ec
---

# Think in Sentences: Explicit Sentence Boundaries Enhance Language Model's Capabilities

**Conference**: ACL 2026
**arXiv**: [2604.10135](https://arxiv.org/abs/2604.10135)
**Code**: [GitHub](https://github.com/CLCS-SUSTech/think-in-sentence)
**Area**: LLM/NLP
**Keywords**: sentence boundaries, delimiters, in-context learning, supervised fine-tuning, free lunch

## TL;DR

This paper proposes inserting delimiter tokens at sentence boundaries in LLM inputs to implement a "think-in-sentences" reasoning paradigm via both ICL and SFT. The approach yields consistent improvements across models ranging from 7B to 600B parameters (GSM8k +7.7%, DROP +12.5%) with negligible additional computational overhead.

## Background & Motivation

**Background**: Sentence-level structure was central to early neural language models — Skip-thought trained on reconstructing adjacent sentences, and BERT's next-sentence prediction task encoded inter-sentence coherence. With the rise of LLMs, however, sentence boundaries have been demoted to ordinary tokens, and models have entirely disregarded sentence structure in their token-by-token processing pipelines.

**Limitations of Prior Work**: Mainstream approaches to enhancing LLM capabilities either require substantial training costs (training-time scaling) or introduce inference latency (test-time scaling such as CoT). Goyal et al. (2024) proposed inserting "pause" tokens as a free-lunch solution, but with significant limitations: (1) pause token placement lacks linguistic priors and requires manual, task-specific tuning of insertion counts; (2) validation has not been extended to 7B+ models; (3) robustness and generalizability remain insufficient.

**Key Challenge**: Human language production relies on an incremental, sentence-by-sentence cognitive process, yet LLMs are trained on the continuous text produced by this process, resulting in an inherent misalignment between human cognitive mechanisms and model input processing.

**Goal**: Design a strategy that exploits sentence-level linguistic priors to enhance LLM performance in a robust and low-overhead manner.

**Key Insight**: The authors observe that sentences constitute the most natural "cognitive chunks" in natural language. Inserting structural delimiters at sentence boundaries can trigger a "context integration → next-step planning" cycle that simulates the human post-sentence reflection process.

**Core Idea**: Insert task-agnostic delimiter tokens at sentence boundaries so that LLMs implicitly perform sentence-by-sentence reasoning. This is realized via two paradigms: ICL (demonstrating delimiter patterns in prompts) and SFT (fine-tuning on delimiter-augmented data).

## Method

### Overall Architecture

Given a text sequence $T = [t_1, t_2, ..., t_n]$, sentence boundaries are identified using a sentence segmentation tool (SaT-12L-sm), and a delimiter $x_{seg}$ is inserted at the end of each sentence, yielding the structured sequence $S = [s_1, x_{seg}, s_2, x_{seg}, ..., s_n, x_{seg}]$. The model's objective is not only to predict the next token but also to learn when to generate the delimiter, thereby performing implicit sentence segmentation.

### Key Designs

1. **ICL Method (Sentence-Aware Prompting)**:

    - Function: Guides the model to adopt a sentence-by-sentence generation style during inference by demonstrating sentence delimiter patterns in few-shot examples.
    - Mechanism: Each sentence in the few-shot examples within the prompt is explicitly terminated with a `<seg>` marker. Through analogical learning, the model automatically continues this structured, sentence-by-sentence generation pattern during inference. No modification of model weights is required; standard autoregressive inference suffices.
    - Design Motivation: ICL is a lightweight inference-time method well suited to long-context scenarios, but it is constrained by context length and has limited effectiveness in zero-shot or context-restricted settings.

2. **SFT Method (Internalizing Sentence Structure)**:

    - Function: Directly internalizes sentence-level structural priors into model parameters through supervised fine-tuning.
    - Mechanism: Delimiters are systematically inserted at all sentence boundaries in the TULU3 dataset, and the model is then fine-tuned with a standard causal language modeling loss. The delimiter is added as a new special token to the tokenizer, and the corresponding embedding and LM head weights are learned during training. After training, the model can natively generate delimiter-annotated text.
    - Design Motivation: Overcomes the context-dependence limitation of ICL, enabling the model to function effectively in zero-shot scenarios and making it more suitable for practical deployment.

3. **Delimiter Selection Strategy**:

    - Function: Identifies the most effective delimiter form.
    - Mechanism: Multiple delimiter types are evaluated — structured tokens (`<seg>`, `<and>`, `####`), semantic words ("seg", "and"), punctuation ("\n", "."), and arbitrary symbols. Structured tokens consistently outperform all other types and are the only category that surpasses the baseline across all tasks.
    - Design Motivation: An ideal delimiter should be a purely structural marker, semantically unrelated to the text content. Semantic delimiters introduce ambiguity (the model must distinguish marker function from content), whereas structured tokens provide an unambiguous sentence boundary signal.

### Loss & Training

SFT employs a standard causal language modeling loss: $\mathcal{L}_{SFT}(\theta) = \sum_{s' \in S} \sum_{i=1}^{|s'|} \log P(t_i | t_{<i}; \theta)$, where $s' = [s, x_{seg}]$ and the final token $t_{|s'|} = x_{seg}$. Full-parameter fine-tuning is conducted on 8×L40 GPUs.

## Key Experimental Results

### Main Results (ICL)

| Model | GSM8k Δ | DROP Δ | MMLU Δ | MATH Δ |
|------|---------|--------|--------|--------|
| Qwen2-7B-Inst | +7.73% | +12.50% | +5.53% | +0.97% |
| Llama3-8B-Inst | +2.50% | +6.77% | +4.39% | -0.34% |
| Qwen2.5-72B-Inst | +1.82% | +1.64% | -0.24% | +2.74% |
| DeepSeek-V3 | +0.30% | +4.00% | +0.78% | +1.20% |

### SFT Results (Llama3-8B-Base)

| Method | MMLU | GSM8k | DROP | MMLU-Pro | HumanEval |
|------|------|-------|------|----------|-----------|
| Std-FT | 59.02 | 72.48 | 48.50 | 34.25 | 56.71 |
| Pause-FT | 56.11 | 75.44 | 55.97 | 35.71 | - |
| **Seg-FT** | **60.13** | 74.91 | 54.26 | **40.71** | **62.80** |

### Key Findings
- Smaller models benefit the most (7B-scale gains are pronounced), while larger models show smaller but still consistent improvements.
- DROP (reading comprehension requiring cross-sentence reasoning) exhibits the most substantial gains, indicating that sentence delimiters help models better process sentence-encoded facts and their relations.
- Seg-FT outperforms Std-FT on all 7 benchmarks, whereas Pause-FT degrades on knowledge-intensive tasks (MMLU, GPQA).
- Sentence-awareness generalizes to code generation (HumanEval +6.09%), with the model learning to insert delimiters within code as well.
- Analysis using probability-based vs. CoT-based evaluation reveals that delimiters do not improve knowledge retrieval but enhance multi-step reasoning processes.

## Highlights & Insights
- The insight that **sentences are "natural cognitive chunks"** is particularly compelling: the performance of fixed $n$-token chunking follows an inverted-U curve, with the optimal range $n \in [32, 64]$ corresponding precisely to typical sentence lengths, confirming that the sentence level is the optimal granularity for information processing. This parallels the concept of cognitive chunking in humans.
- **A key improvement over the "free lunch" methodology**: compared to the indiscriminate insertion of pause tokens, exploiting linguistic priors (sentence boundaries) makes the approach more robust and general, eliminating the need for task-specific hyperparameter tuning.
- **The unexpected generalization of SFT to code** is thought-provoking: sentence segmentation patterns in natural language transfer to line structure in code, suggesting that both share some form of structural prior.

## Limitations & Future Work
- ICL depends on sufficient context length to accommodate few-shot examples, limiting its applicability in zero-shot or context-restricted scenarios.
- SFT is validated only on Llama3-8B-Base; SFT experiments on larger models are absent.
- Sentence segmentation relies on an external tool (SaT-12L-sm), which may introduce segmentation errors.
- The authors do not explore adaptive selection of delimiter insertion positions during training (e.g., inserting only at critical sentence boundaries).
- Gains are relatively modest for highly structured tasks such as mathematical reasoning, with slight degradation on MATH for certain models.

## Related Work & Insights
- **vs. Pause Token (Goyal et al. 2024)**: Pause Token inserts markers indiscriminately and requires task-specific tuning, whereas this work exploits sentence boundaries as a linguistic prior, yielding greater robustness and generalizability. The SFT experiments directly demonstrate that Seg-FT outperforms Pause-FT overall.
- **vs. CoT Reasoning**: CoT enhances capability through explicit generation of reasoning steps but increases token consumption, whereas the proposed approach improves reasoning via implicit sentence delimiting at near-zero additional overhead. Ablation experiments show that the two approaches act synergistically.

## Rating
- Novelty: ⭐⭐⭐⭐ The sentence-boundary delimiter idea is simple yet effective, with a clear and well-grounded intuition rooted in cognitive science.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers multiple models and tasks with rich ablation analyses (delimiter selection, granularity, mechanistic analysis).
- Writing Quality: ⭐⭐⭐⭐ Motivation and experimental logic are clear; analysis is thorough.
- Value: ⭐⭐⭐⭐ Provides a practical free-lunch method, though gains are more limited on larger models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Do Large Language Models Think Like the Brain? Sentence-Level Evidences from Layer-Wise Embeddings and fMRI](../../AAAI2026/llm_nlp/do_large_language_models_think_like_the_brain_sentence-level_evidences_from_laye.md)
- [\[ACL 2026\] The Model Agreed, But Didn't Learn: Diagnosing Surface Compliance in Large Language Models](the_model_agreed_but_didn39t_learn_diagnosing_surface_compliance_in_large_langua.md)
- [\[ACL 2026\] FastDiSS: Few-step Match Many-step Diffusion Language Model on Sequence-to-Sequence Generation](fastdiss_few-step_match_many-step_diffusion_language_model_on_sequence-to-sequen.md)
- [\[AAAI 2026\] IROTE: Human-like Traits Elicitation of Large Language Model via In-Context Self-Reflective Optimization](../../AAAI2026/llm_nlp/irote_human-like_traits_elicitation_of_large_language_model_via_in-context_self-.md)
- [\[AAAI 2026\] An Invariant Latent Space Perspective on Language Model Inversion](../../AAAI2026/llm_nlp/an_invariant_latent_space_perspective_on_language_model_inve.md)

</div>

<!-- RELATED:END -->
