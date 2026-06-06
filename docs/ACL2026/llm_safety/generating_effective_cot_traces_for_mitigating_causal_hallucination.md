---
title: >-
  [Paper Note] Generating Effective CoT Traces for Mitigating Causal Hallucination
description: >-
  [ACL 2026][LLM Safety][Causal Hallucination] This paper proposes the Causal Hallucination Rate (CHR) metric to quantify the tendency of small LLMs to over-predict causal relationships in Event Causality Identification (E…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "Causal Hallucination"
  - "Chain-of-Thought"
  - "Event Causality Identification"
  - "Small Model Fine-tuning"
  - "Data Generation"
date: 2026-05-08
content_hash: a7f320e5d944109c
---

# Generating Effective CoT Traces for Mitigating Causal Hallucination

**Conference**: ACL 2026  
**arXiv**: [2604.12748](https://arxiv.org/abs/2604.12748)  
**Code**: None  
**Area**: LLM Reasoning  
**Keywords**: Causal Hallucination, Chain-of-Thought, Event Causality Identification, Small Model Fine-tuning, Data Generation

## TL;DR
This paper proposes the Causal Hallucination Rate (CHR) metric to quantify the tendency of small LLMs to over-predict causal relationships in Event Causality Identification (ECI). Through systematic experiments, two key criteria for effective CoT data are identified (sufficiently long semantic explanations + distribution alignment with the target model). A low-cost CoT data generation pipeline is designed, reducing the CHR of Qwen2.5-1.5B from 83.54% to 6.26% while improving average accuracy to 66.00%.

## Background & Motivation

**Background**: Large language models perform exceptionally well on complex reasoning tasks such as mathematics and programming, but they suffer from severe "causal hallucination" in Event Causality Identification (ECI) tasks—models tend to predict a causal relationship regardless of whether one truly exists between event pairs. This issue is particularly acute in small models (≤1.5B parameters); for instance, Qwen2.5-1.5B achieves high accuracy on causal event pairs but near-zero accuracy on non-causal pairs.

**Limitations of Prior Work**: Existing ECI research primarily focuses on inference-time prompting (e.g., causal prompting in Dr.ECI, multi-agent debate in MRBalance), but these methods fail to alleviate causal hallucination in small models. Current ECI datasets only contain binary labels and lack intermediate reasoning steps, making them unsuitable for CoT fine-tuning. Furthermore, existing CoT data construction guidelines—such as prioritizing low perplexity, assuming shorter traces are easier to learn, or rewriting to bridge distribution gaps—were derived from mathematical reasoning tasks and may not apply to ECI.

**Key Challenge**: Due to limited parameter counts, small LLMs struggle to learn fine-grained causal discrimination from binary labels or short prompts alone. They require rich intermediate reasoning steps to "teach" the model how to distinguish between causal and non-causal relationships. However, what constitutes an "effective" CoT trace in the ECI domain has not been systematically studied.

**Goal**: (1) Define a metric to quantify causal hallucination; (2) Systematically investigate criteria for effective CoT traces; (3) Design a low-cost CoT data generation pipeline to mitigate causal hallucination in small models.

**Key Insight**: Instead of directly adopting CoT construction guidelines from mathematical reasoning, the authors conduct controlled experiments on perplexity, trace length, and distribution gaps. They discover unique criteria for the ECI domain: longer traces are actually superior, and perplexity is not a reliable selection criterion.

**Core Idea**: Effective ECI CoT traces must satisfy two criteria: they must contain sufficiently long semantic explanations and reasoning steps (Criterion I), and they must maintain a small distribution gap with the target model without increasing perplexity (Criterion II). A two-step generation pipeline is designed based on these criteria.

## Method

### Overall Architecture
A two-step CoT trace generation pipeline: The first step uses Qwen3-235B-A22B (Thinking) to construct few-shot examples, prompting Llama3.1-8B to generate CoT traces containing rich semantic explanations and reasoning steps, retaining only those that yield the correct answer. The second step uses the target model itself to rewrite these traces to reduce the distribution gap, verifying that perplexity does not increase after rewriting. Finally, the target small model is fine-tuned using these traces via LoRA.

### Key Designs

1.  **Causal Hallucination Rate (CHR) Metric**:
    *   **Function**: Quantifies the degree of causal hallucination in models during event causality identification.
    *   **Mechanism**: $\text{CHR} = \text{Acc}_{\text{causal}} - \text{Acc}_{\text{non-causal}}$, representing the difference between accuracy on causal event pairs and non-causal event pairs. $\text{CHR} > 0$ indicates causal hallucination (higher values denote more severity), while $\text{CHR} < 0$ indicates a tendency to over-predict non-causal relationships. For example, the original CHR of Qwen2.5-1.5B is 83.54%, meaning the model predicts almost all event pairs as causal.
    *   **Design Motivation**: Traditional overall accuracy or F1 scores fail to capture this systematic bias—a model predicting all samples as causal might have 50% overall accuracy but a CHR of 100%.

2.  **Empirical Findings on Effective CoT Trace Criteria**:
    *   **Function**: Determines which type of CoT trace best alleviates causal hallucination.
    *   **Mechanism**: Three sets of controlled experiments yield conclusions different from mathematical reasoning: (1) Perplexity is not a reliable selection criterion—traces chosen for low perplexity resulted in a CHR of 39.26%, while longer Llama traces (with higher perplexity) yielded a CHR of only 34.12% due to richer semantic explanations; (2) Small models can actually learn from longer CoT traces—as trace length increases, CHR consistently decreases (242 tokens: 59.79% → 317 tokens: 34.68% → 482 tokens: 30.60%); (3) The rewriting strategy is only effective if it does not increase perplexity—rewriting medium-length traces actually increased both perplexity and CHR.
    *   **Design Motivation**: Overturns three "common sense" rules transferred from the mathematical reasoning domain, establishing specific data construction guidelines for ECI tasks.

3.  **Two-step CoT Generation Pipeline**:
    *   **Function**: Generates high-quality CoT training data satisfying both criteria at a low cost.
    *   **Mechanism**: Step 1—Use Qwen3-235B-A22B (Thinking) to construct two few-shot examples (one causal, one non-causal) and prompt Llama3.1-8B to generate long CoT traces with rich semantic explanations, keeping correct ones. Step 2—Use the target model (e.g., Qwen2.5-1.5B) to rewrite the traces from the first step, verifying that perplexity has not increased and the answer remains correct; otherwise, the original trace is kept. The pipeline primarily relies on Llama3.1-8B, ensuring low cost.
    *   **Design Motivation**: Balances trace quality (guided by a larger model) with distribution alignment (rewritten by the target model), while ensuring data quality via a fallback mechanism that preserves original traces.

### Loss & Training
LoRA fine-tuning is performed using the `SFTTrainer` from the TRL framework: batch size of 1, 8 gradient accumulation steps, 1 epoch, and a learning rate of $2 \times 10^{-4}$ with a cosine annealing scheduler. LoRA parameters: rank=8, scaling factor=16, dropout=0.05. The decoding temperature is fixed at 0 to ensure reproducibility.

## Key Experimental Results

### Main Results

| Method | CHR (↓) | mAcc (↑) |
| :--- | :--- | :--- |
| GPT-4 | 53.30 | 51.40 |
| Llama3.1-8B | 60.59 | 58.97 |
| Qwen2.5-1.5B (Original) | 83.54 | 52.97 |
| Qwen2.5-1.5B (CoT Prompt) | 69.77 | 51.48 |
| Qwen2.5-1.5B (Binary Label FT) | 66.67 | 56.74 |
| **Qwen2.5-1.5B (Ours)** | **6.26** | **66.00** |
| Llama3.2-1B (Original) | 76.43 | 55.58 |
| **Llama3.2-1B (Ours)** | **9.14** | **63.44** |

### Ablation Study

| Configuration | CHR | mAcc | Description |
| :--- | :--- | :--- | :--- |
| Qwen2.5-1.5B Original | 83.54 | 52.97 | Baseline |
| w/o Rewriting | 23.39 | 56.51 | Step 1 only |
| w/ Rewriting | 6.26 | 66.00 | Full pipeline |
| Llama3.2-1B Original | 76.43 | 55.58 | Baseline |
| w/o Rewriting | 17.13 | 55.51 | Step 1 only |
| w/ Rewriting | 9.14 | 63.44 | Full pipeline |

### Key Findings
- The fine-tuned 1.5B model exhibits lower causal hallucination than GPT-4 (CHR 6.26% vs. 53.30%) and Llama3.1-8B (6.26% vs. 60.59%), indicating the extremely high quality of the CoT data generated by the pipeline.
- Strong cross-dataset generalization: Models trained on EventStoryLine achieved CHR reductions to 11.37% and 11.13% on Causal-TimeBank and MAVEN-ERE, respectively.
- Strong cross-difficulty generalization: Sentence-level training data generalizes to document-level ECI (more difficult cross-sentence event pairs), with CHR dropping from 54.52% to 1.41%.
- Robustness: Accuracy remains stable even after injecting incorrect intervention prompts, suggesting the model has truly learned causal reasoning rather than simply following instructions.

## Highlights & Insights
- Overturning three "common sense" rules of CoT construction transferred from mathematical reasoning is the most significant contribution. Specifically, the discovery that "small models can learn from long CoT traces" contradicts the findings of Li et al. (2025); however, rigorous controlled experiments provide a convincing explanation—semantic explanations within long traces are the critical factor, rather than the length of the trace itself.
- The design of the CHR metric is simple yet strikes at the heart of the problem: traditional evaluation metrics mask causal hallucination (a model with CHR=100% still maintains 50% overall accuracy), whereas CHR directly exposes the model's systematic bias. This metric could be generalized to other tasks with label imbalance preferences.
- The two-step pipeline reflects a refined balance of "using large models to guide quality and small models to adapt distribution," while avoiding potential quality degradation from rewriting via a fallback mechanism.

## Limitations & Future Work
- The first step of the pipeline relies on Qwen3-235B to construct few-shot examples; while only two examples are needed, access to a large model is still required.
- The effectiveness condition of the rewriting strategy (no increase in perplexity) requires pre-verification, and behavior is inconsistent across different trace lengths.
- Experiments only covered English ECI datasets like EventStoryLine; applicability to Chinese or other languages has not been verified.

## Related Work & Insights
- **vs. Dr.ECI (Cai et al., 2025)**: Dr.ECI uses causal reasoning principles to construct prompts, but CHR reaches 100% on Qwen2.5-1.5B (predicting all as causal), whereas this paper reduces CHR to 6.26% through CoT fine-tuning.
- **vs. Perplexity selection strategy by Zhang et al. (2025)**: This method is effective in mathematical reasoning but does not hold for ECI—traces with the lowest perplexity do not yield the lowest CHR; length and semantic richness are the dominant factors.

## Rating
- Novelty: ⭐⭐⭐⭐ Proposes the CHR metric and overturns three CoT construction commonalities, offering a unique contribution to the ECI field.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Rigorous controlled variable experiments, with comprehensive cross-dataset/cross-difficulty/robustness testing.
- Writing Quality: ⭐⭐⭐⭐ Logical progression from findings to criteria to the pipeline is clear.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Mitigating Object Hallucination in LVLMs via Attention Imbalance Rectification](../../CVPR2026/llm_safety/mitigating_object_hallucinations_in_lvlms_via_attention_imbalance_rectification.md)
- [\[ACL 2026\] ForgeryTalker: Generating Attribution Reports for Manipulated Facial Images](generating_attribution_reports_for_manipulated_facial_images_a_dataset_and_basel.md)
- [\[ACL 2026\] Multi-component Causal Tracing in Large Language Models](multi-component_causal_tracing_in_large_language_models.md)
- [\[ACL 2026\] CausalDetox: Causal Head Selection and Intervention for Language Model Detoxification](causaldetox_causal_head_selection_and_intervention_for_language_model_detoxifica.md)
- [\[ICCV 2025\] ChartCap: Mitigating Hallucination of Dense Chart Captioning](../../ICCV2025/llm_safety/chartcap_mitigating_hallucination_of_dense_chart_captioning.md)

</div>

<!-- RELATED:END -->
