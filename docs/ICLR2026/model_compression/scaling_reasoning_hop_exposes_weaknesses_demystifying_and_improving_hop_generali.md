---
title: >-
  [Paper Note] Scaling Reasoning Hop Exposes Weaknesses: Demystifying and Improving Hop Generalization in Large Language Models
description: >-
  [ICLR 2026][Model Compression][Reasoning hop generalization] This paper systematically uncovers the internal mechanism underlying LLM failures in reasoning hop generalization — namely, attention head competition between correct and erroneous reasoning trajectories — and proposes TCR (Test-time Correction of Reasoning), which dynamically identifies and deactivates erroneous processing heads (ep heads) at inference time to correct reasoning errors, achieving an average accuracy improvement of 5–7%.
tags:
  - ICLR 2026
  - Model Compression
  - Reasoning hop generalization
  - Chain-of-Thought
  - Attention head competition mechanism
  - Erroneous processing heads
  - Test-time intervention
date: 2026-05-08
content_hash: fd66ca72c4a28cc1
---

# Scaling Reasoning Hop Exposes Weaknesses: Demystifying and Improving Hop Generalization in Large Language Models

**Conference**: ICLR 2026
**arXiv**: [2601.21214](https://arxiv.org/abs/2601.21214)
**Area**: Model Compression / LLM Interpretability
**Keywords**: Reasoning hop generalization, Chain-of-Thought, Attention head competition mechanism, Erroneous processing heads, Test-time intervention

## TL;DR

This paper systematically uncovers the internal mechanism underlying LLM failures in reasoning hop generalization — namely, attention head competition between correct and erroneous reasoning trajectories — and proposes TCR (Test-time Correction of Reasoning), which dynamically identifies and deactivates erroneous processing heads (ep heads) at inference time to correct reasoning errors, achieving an average accuracy improvement of 5–7%.

## Background & Motivation

- **Background**: Chain-of-Thought (CoT) reasoning has become the standard paradigm for LLMs to solve complex problems, yet performance degrades sharply when the number of reasoning steps required at test time exceeds the training distribution (reasoning hop generalization).
- **Limitations of Prior Work**: For example, 3×8-digit multiplication and 2×2-digit multiplication require the same multiplication skill, yet performance degrades significantly in the multi-hop version. Existing methods either require post-training on downstream data (Hu et al., 2025) or architectural modifications (Fan et al., 2025, looped transformer), and cannot compatibly enhance the reasoning capabilities of off-the-shelf LLMs.
- **Key Challenge**: The internal mechanisms underlying reasoning hop generalization failures are poorly understood — existing interpretability tools are primarily designed for simple local prediction tasks (e.g., factual recall, simple arithmetic) and cannot be directly applied to long-chain CoT reasoning involving hundreds of tokens.
- **Key Insight**: The paper adopts an error-centric perspective, first systematically identifying critical error types and their corresponding token positions, then employing mechanistic analysis tools (Logit Lens, Knockout, circuit analysis) to investigate internal mechanisms.
- **Core Idea**: LLMs simultaneously harbor correct and erroneous reasoning trajectories, driven by distinct sets of attention heads. Erroneous processing heads (ep heads) cause reasoning failures by amplifying erroneous signals and suppressing correct ones; deactivating these ep heads restores correct predictions.

## Method

### Overall Architecture

The paper comprises two major components: **mechanistic analysis** (Sections 3–4) and **the TCR intervention** (Section 5).

1. **Mechanistic Analysis**: Decomposes CoT reasoning into hop-by-hop analysis, identifies critical error types, and reveals competition mechanisms at the attention head level.
2. **TCR**: Based on mechanistic insights, designs a lightweight test-time intervention — an entropy threshold detector identifies error positions, while a trained head selector selects ep heads to deactivate.

### Key Designs

#### 1. Systematic Decomposition of Reasoning Errors

- **Function**: Decomposes CoT responses into fine-grained reasoning hops, locating the token position at which the first error occurs.
- **Mechanism**: For an $n$-hop problem $x \to r_1 \to \cdots \to r_n \to y$, the overall CoT accuracy is decomposed as a product of hop-wise conditional probabilities: $p(r_1, \ldots, r_n, y | x) = \prod_{i=1}^{n} p(r_i | x, r_1, \ldots, r_{i-1}) \cdot p(y | x, r_1, \ldots, r_n)$
- **Key Findings**: Each task exhibits only 1–2 critical error types that account for ≥30% of all errors. For example, in the Parity-NL 50-hop task, 78.6% of errors stem from a single error type: "recalling the wrong name."
- **Design Motivation**: Error concentration in a small number of patterns implies the existence of coherent underlying mechanisms, making mechanistic analysis tractable.

#### 2. Discovery of Competition Mechanism Among Attention Heads

The paper identifies three functionally distinct categories of attention heads in LLM reasoning circuits:

- **Answer-Writing Heads (aw heads)**: Located in the middle-to-deep layers (e.g., layers 20–26), they directly write answer information into the residual stream. Using an improved localization metric $s_{\text{aw-head}}(\mathbf{a}_i^l)$ (Equation 4), it is found that correct and erroneous predictions share approximately 60% of aw heads, and these heads simultaneously encode signals for both correct and incorrect tokens.
- **Processing Heads**: Located in the shallow-to-middle layers, they support reasoning through indirect information processing. They are divided into two groups:
  - **Correct Processing Heads (cp heads, $\mathcal{H}_{cp}$)**: Drive correct reasoning trajectories.
  - **Erroneous Processing Heads (ep heads, $\mathcal{H}_{ep}$)**: Drive erroneous reasoning trajectories.
  - Key finding: $\mathcal{H}_{cp}$ and $\mathcal{H}_{ep}$ are nearly disjoint.
- **Basic Heads ($\mathcal{H}_{basic}$)**: Extract basic input information and are indispensable for both correct and erroneous predictions.

**Competition Mechanism**: Correct and erroneous reasoning trajectories coexist within the LLM. At critical error positions, ep heads amplify spurious signals and suppress correct ones, causing the probability of erroneous candidate tokens in aw heads to exceed that of correct candidates, ultimately producing incorrect outputs. After deactivating a single ep head, the reasoning mechanism of correct processing heads is restored (93.3% of cp heads align with the original correct predictions).

#### 3. TCR: Test-time Correction of Reasoning

TCR consists of three components:

**(a) Candidate ep Head Set Construction**: Ep heads are localized across five representative tasks; heads shared across tasks and error types are selected, yielding a compact candidate set $\mathbf{H}$ (8 heads for Qwen2.5-7B, 9 for Phi-3 and Qwen3-8B, 10 for LLaMA3-8B).

**(b) Head Selector Training**: A classifier $f_\theta(\cdot)$ fine-tuned via Qwen2.5-0.5B + LoRA selects which ep heads to deactivate given the input context. Trained with multi-label Softmax loss, Hit@1 accuracy reaches 75–87% in-distribution and 35–82% out-of-distribution.

**(c) Entropy-based Detector**: Monitors prediction entropy at each generated token; when entropy exceeds threshold $\tau$, intervention is triggered. At each trigger, the top-3 heads predicted by the classifier are selected, each is deactivated individually, and the final corrected output is determined by majority voting.

### Theoretical Analysis

The paper provides two explanations for why more reasoning hops exacerbate errors:
1. **Expanded Search Space**: Longer reasoning chains entail larger input sizes and more intermediate states to track, greatly increasing the difficulty of retrieving the correct reasoning trajectory $\mathcal{H}_{cp}$.
2. **Out-of-Distribution Generalization Failure**: When the required number of hops substantially exceeds the training distribution, correct reasoning trajectories are more frequently overridden by $\mathcal{H}_{ep}$, which may capture only local patterns and thus lead to shortcut reasoning.

## Key Experimental Results

### Main Results: TCR Performance Across 7 Tasks × 4 LLMs

| Method | Parity-NL | MDM | LLC | CLF | MOAS | ObjC | NumS | Avg. |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-7B Base | 48.3% | 43.0% | 11.7% | 56.8% | 39.2% | 52.0% | 41.1% | 41.7% |
| +DoLa | 58.1% | 38.5% | 8.0% | 52.3% | 40.0% | 52.3% | 48.7% | 42.6% |
| **+TCR** | **60.4%** | **48.2%** | **16.2%** | **66.6%** | **46.0%** | **56.0%** | **46.0%** | **48.5% (+6.8%)** |
| **+TCR-gold** | **81.2%** | **58.3%** | **23.0%** | **71.3%** | **62.0%** | **76.0%** | **54.5%** | **61.3% (+19.6%)** |
| LLaMA3-8B Base | 70.0% | 0.0% | 81.0% | 15.2% | 22.9% | 68.8% | 4.5% | 37.5% |
| **+TCR** | **82.0%** | 0.0% | **82.3%** | **28.2%** | **39.4%** | 67.8% | **7.8%** | **43.9% (+6.4%)** |
| **+TCR-gold** | **88.0%** | 0.0% | **90.7%** | **32.7%** | **47.0%** | **76.4%** | **10.1%** | **49.3% (+11.8%)** |

### Head Selector Generalization Performance (Hit@1 Accuracy)

| Model | In-Distribution | Out-of-Distribution |
|---|---|---|
| Qwen2.5-7B-Instruct | 79.6% | 53.4% |
| Phi-3-Instruct | 75.2% | 58.2% |
| LLaMA3-8B-Instruct | 80.8% | 35.5% |
| Qwen3-8B-Instruct | 87.2% | 82.2% |

### Key Findings

1. TCR consistently improves reasoning hop generalization across all four models, with an average gain of 5–7%; TCR-gold demonstrates the correction upper bound (nearly 20% improvement on Qwen2.5).
2. DoLa (a hallucination mitigation method based on contrastive decoding) yields only marginal or even negative effects in reasoning settings, indicating that reasoning errors are fundamentally different from factual hallucinations.
3. Qwen3-8B approaches saturation on some tasks (e.g., 98.7% on Parity-NL), yet TCR-gold still yields a 22.4% improvement on the challenging MDM task.
4. After deactivating ep heads, the internal mechanism of corrected predictions closely aligns with that of originally correct predictions (93.3% cp head overlap), confirming that correct reasoning circuits genuinely exist but are suppressed.

## Highlights & Insights

1. **Significance of Core Finding**: LLMs simultaneously run correct and erroneous reasoning trajectories in parallel internally; which one prevails depends on the "competition" outcome among a small number of attention heads. This finding provides an entirely new perspective for understanding LLM reasoning failures.
2. **Methodological Innovation**: An improved answer-writing head localization metric (Equation 4) is proposed, which resolves cross-layer probability scale discrepancies through knockout effect normalization, achieving greater accuracy than pure Logit Lens approaches.
3. **Cross-task Shared ep Heads**: Ep heads exhibit substantial overlap across different tasks and error types, enabling a single compact candidate set (8–10 heads) to cover all scenarios.
4. **Implications of TCR-gold**: Under an oracle detector, Qwen2.5 jumps from 41.7% to 61.3%, demonstrating that LLMs internally harbor correct reasoning capabilities far exceeding their current performance, which are merely suppressed by erroneous mechanisms.

## Limitations & Future Work

1. **Overly Simple Entropy Threshold Detector**: The fixed threshold $\tau$ produces numerous false positives (normal high-entropy tokens misidentified as errors), which is the primary reason for the large gap between TCR and TCR-gold (6.8% vs. 19.6%).
2. **Limited Out-of-Distribution Generalization of Head Selector**: Out-of-distribution Hit@1 on LLaMA3 is only 35.5%, indicating that ep head activation patterns still differ substantially across tasks.
3. **Manual Involvement in Candidate Set Construction**: Mechanistic analysis must first be performed separately across multiple tasks to localize ep heads, followed by manual selection of the intersection — a relatively heavy pipeline.
4. **Validation Limited to Symbolic Reasoning / Math / Programming Tasks**: Effectiveness on more open-ended reasoning tasks such as natural language inference, commonsense reasoning, and multi-step planning remains unknown.
5. **Additional Computation from Majority Voting**: Each trigger requires 3 knockout passes plus regeneration, reducing inference efficiency.
6. **Compatibility with Reasoning Models (e.g., o1/R1) Not Verified**: The reasoning mechanisms of these models may differ from standard CoT.

## Related Work & Insights

- **Reasoning Hop Generalization**: Dziri et al. (2023) attribute the problem to single-hop error accumulation; Hu et al. (2025) propose rule-verbalization fine-tuning; Fan et al. (2025) reuse computation via looped transformers — this paper is the first to explain the problem through the lens of attention head competition mechanisms.
- **LLM Mechanistic Analysis**: Circuit analysis (Wang et al., 2023) and causal indirect effects (Meng et al., 2022) — this paper extends these tools from simple tasks to long-chain CoT reasoning.
- **Test-time Intervention**: DoLa (Chuang et al., 2024) mitigates hallucinations via inter-layer contrastive decoding but is unsuitable for reasoning settings; the knockout intervention proposed here is more direct and effective.
- **Insights**: The cross-task sharing of ep heads suggests that LLMs may harbor universal "erroneous reasoning modules," motivating future exploration of more systematic reasoning circuit editing approaches.

## Rating

⭐⭐⭐⭐ (4/5)

- **Novelty**: ⭐⭐⭐⭐⭐ First to reveal the attention head competition mechanism in reasoning hop generalization; findings are highly insightful.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Coverage across 7 tasks × 4 models is comprehensive; mechanistic analysis is rigorous.
- **Writing Quality**: ⭐⭐⭐⭐ Research questions are clearly articulated, analytical logic is tight, and the circuit diagram in Figure 1 is highly intuitive.
- **Value**: ⭐⭐⭐ TCR requires pre-training a head selector, and the simple detector limits practical gains (large gap between TCR and TCR-gold).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] A Fano-Style Accuracy Upper Bound for LLM Single-Pass Reasoning in Multi-Hop QA](a_fano-style_accuracy_upper_bound_for_llm_single-pass_reasoning_in_multi-hop_qa.md)
- [\[ICLR 2026\] Landscape of Thoughts: Visualizing the Reasoning Process of Large Language Models](landscape_of_thoughts_visualizing_the_reasoning_process_of_large_language_models.md)
- [\[ICLR 2026\] InftyThink: Breaking the Length Limits of Long-Context Reasoning in Large Language Models](inftythink_breaking_the_length_limits_of_long-context_reasoning_in_large_languag.md)
- [\[ICLR 2026\] BeyondBench: Contamination-Resistant Evaluation of Reasoning in Language Models](beyondbench_contamination-resistant_evaluation_of_reasoning_in_language_models.md)
- [\[ACL 2026\] SeLaR: Selective Latent Reasoning in Large Language Models](../../ACL2026/model_compression/selar_selective_latent_reasoning_in_large_language_models.md)

</div>

<!-- RELATED:END -->
