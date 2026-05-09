---
title: >-
  [Paper Note] Generating Effective CoT Traces for Mitigating Causal Hallucination
description: >-
  [ACL 2026][LLM Reasoning][Causal Hallucination] This paper first proposes the Causal Hallucination Rate (CHR) metric to quantify small LLMs' tendency to over-predict causal relations in event causality identification, then identifies two key criteria for effective CoT data through systematic experiments (sufficiently long semantic explanations + distribution alignment with the target model), and designs a low-cost CoT data generation pipeline that reduces Qwen2.5-1.5B's CHR from 83.54% to 6.26% while improving average accuracy to 66.00%.
tags:
  - ACL 2026
  - LLM Reasoning
  - Causal Hallucination
  - Chain-of-Thought
  - Event Causality Identification
  - Small Model Fine-tuning
  - Data Generation
date: 2026-05-08
content_hash: f659f25fbcfe37bd
---
# Generating Effective CoT Traces for Mitigating Causal Hallucination

**Conference**: ACL 2026
**arXiv**: [2604.12748](https://arxiv.org/abs/2604.12748)
**Code**: None
**Area**: LLM Reasoning
**Keywords**: causal hallucination, chain-of-thought, event causal identification, small model fine-tuning, data generation

## TL;DR
This paper first proposes the Causal Hallucination Rate (CHR) metric to quantify the tendency of small LLMs to over-predict causal relations in event causal identification (ECI). Through systematic experiments, two key criteria for effective CoT data are identified—sufficiently long semantic explanations paired with a distribution aligned to the target model—and a low-cost CoT data generation pipeline is designed accordingly. The pipeline reduces CHR of Qwen2.5-1.5B from 83.54% to 6.26% while improving mean accuracy to 66.00%.

## Background & Motivation

**State of the Field**: Large language models excel at complex reasoning tasks such as mathematics and coding, yet suffer from severe "causal hallucination" in ECI—models tend to predict a causal relation regardless of whether one actually exists between an event pair. This problem is especially pronounced in small models (≤1.5B parameters); for instance, Qwen2.5-1.5B achieves high accuracy on causal event pairs but near-zero accuracy on non-causal ones.

**Limitations of Prior Work**: Existing ECI research focuses primarily on inference-time prompt design (e.g., causal prompts in Dr.ECI, multi-agent debate in MRBalance), none of which mitigates causal hallucination in small models. Current ECI datasets contain only binary labels without intermediate reasoning steps, making them unsuitable for CoT fine-tuning. Moreover, existing CoT data construction guidelines—prioritizing low perplexity, preferring shorter traces, and rewriting to reduce distribution gap—were derived from mathematical reasoning tasks and may not transfer to ECI.

**Root Cause**: Due to limited parameter capacity, small LLMs cannot learn fine-grained causal discrimination from binary labels or brief prompts alone; rich intermediate reasoning steps are needed to teach the model to distinguish causal from non-causal relations. However, what constitutes an effective CoT trace for ECI has not been systematically studied.

**Paper Goals**: (1) Define a metric to quantify causal hallucination; (2) systematically investigate the criteria for effective CoT traces; (3) design a low-cost CoT data generation pipeline to mitigate causal hallucination in small models.

**Starting Point**: Rather than directly adopting CoT construction guidelines from mathematical reasoning, the paper conducts controlled experiments over three factors—perplexity, trace length, and distribution gap—and finds that ECI has its own distinct criteria: longer traces are actually better, and perplexity is not a reliable selection criterion.

**Core Idea**: Effective ECI CoT traces must satisfy two criteria—containing sufficiently long semantic explanations and reasoning steps (Criterion I), and maintaining a small distribution gap with the target model without increasing perplexity (Criterion II)—upon which a two-step generation pipeline is designed.

## Method

### Overall Architecture
A two-step CoT trace generation pipeline: in the first step, few-shot examples are constructed with Qwen3-235B-A22B (Thinking) to prompt Llama3.1-8B to generate CoT traces rich in semantic explanations and reasoning steps, retaining only traces that yield correct answers; in the second step, the target model itself rewrites these traces to reduce the distribution gap, with verification that rewriting does not increase perplexity. The resulting traces are used to fine-tune the target small model via LoRA.

### Key Designs

1. **Causal Hallucination Rate (CHR) Metric**:

    - Function: Quantifies the degree of causal hallucination in ECI.
    - Mechanism: $\text{CHR} = \text{Acc}_{\text{causal}} - \text{Acc}_{\text{non-causal}}$, i.e., the difference between accuracy on causal event pairs and accuracy on non-causal event pairs. CHR > 0 indicates causal hallucination (larger values indicate greater severity); CHR < 0 indicates a tendency to over-predict non-causal relations. For example, the original CHR of Qwen2.5-1.5B is 83.54%, meaning the model predicts nearly all event pairs as causal.
    - Design Motivation: Conventional metrics such as overall accuracy or F1 fail to capture this systematic bias—a model that predicts all samples as causal may achieve 50% overall accuracy yet have a CHR of 100%.

2. **Empirical Findings on Effective CoT Trace Criteria**:

    - Function: Determines what kind of CoT traces most effectively mitigates causal hallucination.
    - Mechanism: Three sets of controlled experiments yield conclusions that diverge from mathematical reasoning: (1) Perplexity is not a reliable selection criterion—traces selected by low perplexity yield CHR of 39.26%, whereas longer Llama traces (with higher perplexity) yield CHR of only 34.12%, because longer traces contain richer semantic explanations. (2) Small models can learn from longer CoT traces—CHR decreases monotonically as trace length increases (242 tokens: 59.79% → 317 tokens: 34.68% → 482 tokens: 30.60%). (3) The rewriting strategy is effective only when it does not increase perplexity—rewriting medium-length traces actually increases both perplexity and CHR.
    - Design Motivation: Three "common wisdoms" transferred from mathematical reasoning are refuted, establishing task-specific data construction criteria for ECI.

3. **Two-Step CoT Generation Pipeline**:

    - Function: Generates high-quality CoT training data satisfying both criteria at low cost.
    - Mechanism: Step 1—two few-shot examples (one causal, one non-causal) are constructed using Qwen3-235B-A22B (Thinking) and used to prompt Llama3.1-8B to generate long CoT traces rich in semantic explanations and reasoning steps; only correct traces are retained. Step 2—the target model (e.g., Qwen2.5-1.5B) rewrites the Step 1 traces; rewritten traces are kept only if perplexity does not increase and the answer remains correct, otherwise the original trace is retained. The pipeline relies primarily on Llama3.1-8B, keeping costs low.
    - Design Motivation: Balances trace quality (guided by a large model) with distribution alignment (via target model rewriting), while a fallback mechanism that preserves original traces ensures data quality.

### Loss & Training
LoRA fine-tuning is performed using the SFTTrainer from the TRL framework: batch size 1, gradient accumulation over 8 steps, 1 training epoch, learning rate $2 \times 10^{-4}$, cosine annealing schedule. LoRA rank=8, scaling factor=16, dropout=0.05. Decoding temperature is fixed at 0 for reproducibility.

## Key Experimental Results

### Main Results

| Method | CHR (↓) | mAcc (↑) |
|--------|---------|----------|
| GPT-4 | 53.30 | 51.40 |
| Llama3.1-8B | 60.59 | 58.97 |
| Qwen2.5-1.5B (original) | 83.54 | 52.97 |
| Qwen2.5-1.5B (CoT prompting) | 69.77 | 51.48 |
| Qwen2.5-1.5B (binary label fine-tuning) | 66.67 | 56.74 |
| **Qwen2.5-1.5B (Ours)** | **6.26** | **66.00** |
| Llama3.2-1B (original) | 76.43 | 55.58 |
| **Llama3.2-1B (Ours)** | **9.14** | **63.44** |

### Ablation Study

| Configuration | CHR | mAcc | Note |
|---------------|-----|------|------|
| Qwen2.5-1.5B original | 83.54 | 52.97 | Baseline |
| w/o rewriting | 23.39 | 56.51 | Step 1 only |
| w/ rewriting | 6.26 | 66.00 | Full pipeline |
| Llama3.2-1B original | 76.43 | 55.58 | Baseline |
| w/o rewriting | 17.13 | 55.51 | Step 1 only |
| w/ rewriting | 9.14 | 63.44 | Full pipeline |

### Key Findings
- The fine-tuned 1.5B model exhibits less causal hallucination than GPT-4 (CHR 6.26% vs. 53.30%) and Llama3.1-8B (6.26% vs. 60.59%), demonstrating the high quality of CoT data generated by the pipeline.
- Strong cross-dataset generalization: training on EventStoryLine reduces CHR to 11.37% and 11.13% on Causal-TimeBank and MAVEN-ERE, respectively.
- Strong cross-difficulty generalization: sentence-level training data generalizes to document-level ECI (harder cross-sentence event pairs), reducing CHR from 54.52% to 1.41%.
- Robustness: model accuracy remains largely unchanged after injecting erroneous intervention prompts, indicating that the model has genuinely learned causal reasoning rather than simply following instructions.

## Highlights & Insights
- The most significant contribution of this paper is refuting three "common wisdoms" about CoT construction transferred from mathematical reasoning. In particular, the finding that "small models can learn from long CoT traces" contradicts the conclusion of Li et al. (2025), yet the paper provides a convincing explanation through rigorous controlled experiments—the semantic explanations within longer traces, rather than length per se, are the critical factor.
- The CHR metric is elegantly simple yet strikes at the core issue: conventional evaluation metrics obscure causal hallucination (a model with CHR=100% still achieves 50% overall accuracy), whereas CHR directly exposes the model's systematic bias. This metric is generalizable to other tasks suffering from label-imbalance preference.
- The two-step pipeline embodies a refined balance between "using large models to guide quality" and "using small models to adapt the distribution," while a fallback mechanism that retains original traces prevents quality degradation from rewriting.

## Limitations & Future Work
- The first step of the pipeline relies on Qwen3-235B to construct few-shot examples; although only two examples are required, access to a large-scale model remains necessary.
- The condition for the rewriting strategy to be effective (no increase in perplexity) requires prior validation, and the behavior is inconsistent across different trace lengths.
- Experiments are conducted solely on English ECI datasets such as EventStoryLine; applicability to Chinese or other languages remains unverified.

## Related Work & Insights
- **vs. Dr.ECI (Cai et al., 2025)**: Dr.ECI constructs prompts using causal reasoning principles, but achieves CHR of 100% on Qwen2.5-1.5B (all predictions are causal), whereas the proposed pipeline reduces CHR to 6.26% via CoT fine-tuning.
- **vs. the perplexity-based selection strategy of Zhang et al. (2025)**: This strategy is effective for mathematical reasoning but does not hold for ECI—traces with the lowest perplexity do not yield the lowest CHR; trace length and richness of semantic explanations are the dominant factors.

## Rating
- Novelty: ⭐⭐⭐⭐ Proposes the CHR metric and refutes three CoT construction conventions, making a distinctive contribution to the ECI field.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Controlled experiments are rigorous, with comprehensive cross-dataset, cross-difficulty, and robustness evaluations.
- Writing Quality: ⭐⭐⭐⭐ The experimental analysis progresses in a layered fashion, with clear logic from findings to criteria to pipeline.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] Process Reward Models Meet Planning: Generating Precise and Scalable Datasets for Step-Level Rewards](process_reward_models_meet_planning_generating_precise_and_scalable_datasets_for.md)
- [\[ACL 2026\] Towards Effective In-context Cross-domain Knowledge Transfer via Domain-invariant-neurons-based Retrieval](towards_effective_in-context_cross-domain_knowledge_transfer_via_domain-invarian.md)
- [\[ICLR 2026\] Segment-Level Attribution for Selective Learning of Long Reasoning Traces](../../ICLR2026/llm_reasoning/segment-level_attribution_for_selective_learning_of_long_reasoning_traces.md)
- [\[CVPR 2026\] Understanding and Mitigating Hallucinations in Multimodal Chain-of-Thought Models](../../CVPR2026/llm_reasoning/understanding_and_mitigating_hallucinations_in_multimodal_chain-of-thought_model.md)
- [\[ICLR 2026\] Dynamics Within Latent Chain-of-Thought: An Empirical Study of Causal Structure](../../ICLR2026/llm_reasoning/dynamics_within_latent_chain-of-thought_an_empirical_study_of_causal_structure.md)

<!-- RELATED:END -->
