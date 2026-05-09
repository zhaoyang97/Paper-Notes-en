---
title: >-
  [Paper Note] Does Less Hallucination Mean Less Creativity? An Empirical Investigation in LLMs
description: >-
  [AAAI 2026][Hallucination Mitigation] This paper systematically investigates how three hallucination mitigation methods (CoVe, DoLa, RAG) affect LLM creativity, finding that they exert diametrically opposite effects on divergent creativity—CoVe enhances it, DoLa suppresses it, and RAG has no significant impact—while convergent creativity remains largely unaffected. These patterns hold consistently across model families and parameter scales.
tags:
  - AAAI 2026
  - Hallucination Mitigation
  - Creativity
  - LLM
  - Divergent Thinking
  - Convergent Thinking
date: 2026-05-08
content_hash: b74355fa47b354fc
---

# Does Less Hallucination Mean Less Creativity? An Empirical Investigation in LLMs

**Conference**: AAAI 2026
**arXiv**: [2512.11509](https://arxiv.org/abs/2512.11509)
**Code**: None
**Area**: Information Retrieval
**Keywords**: Hallucination Mitigation, Creativity, LLM, Divergent Thinking, Convergent Thinking

## TL;DR
This paper systematically investigates how three hallucination mitigation methods (CoVe, DoLa, RAG) affect LLM creativity, finding that they exert diametrically opposite effects on divergent creativity—CoVe enhances it, DoLa suppresses it, and RAG has no significant impact—while convergent creativity remains largely unaffected. These patterns hold consistently across model families and parameter scales.

## Background & Motivation

Large language models have demonstrated remarkable capabilities in natural language understanding and reasoning, yet have long suffered from hallucination. The research community has devoted considerable effort to developing hallucination mitigation methods, particularly for high-stakes domains such as AI-assisted scientific discovery. However, a critical question has been entirely overlooked:

**Does suppressing hallucination simultaneously impair a model's creativity?**

This question is significant because **creativity typically involves forming unconventional associations**. Scientists sometimes generate useful hypotheses by connecting seemingly unrelated concepts—a type of associative leap that may resemble what is flagged as "hallucination" in models. Following Guilford's (1950) psychological framework, creativity is decomposed into:

- **Convergent thinking**: Correctly solving problems within given constraints
- **Divergent thinking**: Generating diverse and varied ideas

If hallucination mitigation methods indiscriminately suppress a model's capacity for "free association," then the pursuit of factual accuracy may come at the cost of the creative hypothesis generation essential to scientific discovery. This has profound implications for AI4Science.

**Core hypothesis**: The authors initially hypothesized that hallucination mitigation methods would universally suppress creativity; however, experimental results revealed a more nuanced pattern—different methods exert opposing effects on creativity.

## Method

### Overall Architecture

This paper adopts an experiment-driven research framework (as shown in Figure 1):

1. **Three representative hallucination mitigation methods selected**: CoVe, DoLa, RAG
2. **Evaluated on two creativity benchmarks**: NeoCoder (programming creativity) and CS4 (story creativity)
3. **Tested across model families and scales**: LLaMA (1B/8B/70B), Qwen-Coder 7B, Mistral 7B
4. **Convergent and divergent creativity measured separately**

### Key Designs

#### 1. **Three Hallucination Mitigation Methods**

**(a) CoVe (Chain of Verification)**:
- A structured multi-stage reasoning method consisting of four steps: drafting an initial answer → generating verification questions → answering verification questions → producing a refined answer
- Improves factual consistency by prompting the model to critically evaluate its own outputs
- Implemented using the AutoGen multi-agent framework

**(b) DoLa (Decoding by Contrasting Layers)**:
- A decoding method that contrasts predictions from higher and lower layers
- Dynamically selects the early layer with the greatest divergence from the final layer using Jensen-Shannon divergence
- Subtracts lower-layer logits from higher-layer logits, amplifying tokens learned across layers while suppressing unreliable tokens from earlier layers

**(c) RAG (Retrieval-Augmented Generation)**:
- Retrieves relevant information from an external knowledge base prior to generation
- Uses ColBERTv2 as the retrieval backbone, retrieving documents from CodeRAG-bench
- Applied exclusively to the NeoCoder benchmark (CS4 lacks a retrieval corpus)

#### 2. **Creativity Evaluation Benchmarks**

**(a) NeoCoder**:
- 199 CodeForces programming problems, each with approximately 30 human solutions
- Constraints are incrementally added (up to 5), simulating fixed laws in scientific experiments
- **Convergent**: Proportion of responses that correctly solve the problem while satisfying all constraints
- **Divergent**: Proportion of responses employing novel edit techniques beyond those observed in human solutions

$$\text{NeoCoder-Divergent}(\text{LLM}, t) = \frac{1}{|\mathcal{Y}_t|}\sum_{y_i^t \in \mathcal{Y}_t} \frac{|\mathcal{T}_t^i \setminus \widehat{\mathcal{T}}^i|}{|\mathcal{T}_t^i|}$$

**(b) CS4**:
- 50 instructions, each expanded to 39 cumulative constraints, yielding 250 unique prompts
- Evaluation dimensions: constraint satisfaction, consistency, diversity (Dist-N), and QUC quality score

#### 3. **In-depth Analysis of DoLa: Linear Probe Analysis**

To understand why DoLa suppresses creativity, a linear probe analysis inspired by ITI (Inference-Time Intervention) is applied to examine the correlation between individual layers and creativity:
- Linear probes are trained on each attention head to predict whether the model will produce divergent creative output
- Head-level correlations are aggregated to the layer level
- **Finding**: Early layers exhibit significantly stronger correlations with creativity than later layers (as shown in Figure 4)

Based on this finding, an **inverted DoLa creativity enhancement method** is proposed: amplifying creativity-correlated layers (first 5 layers) and suppressing anti-correlated layers (last 5 layers), successfully improving divergent creativity without degrading convergent creativity.

### Loss & Training

This paper is a purely empirical study involving no model training. The impact of each method is quantified using percentage improvement:

$$\text{Difference from Base}(\%) = \frac{M_{\text{method}} - M_{\text{baseline}}}{M_{\text{baseline}}} \times 100$$

All generations are run independently three times and averaged. GPT-4o-mini is used as the LLM-as-a-Judge where required.

## Key Experimental Results

### Main Results

**Percentage change in divergent creativity relative to baseline**:

| Method | LLaMA-1B | LLaMA-8B | LLaMA-70B | Qwen-Coder-7B | Mistral-7B |
|--------|----------|----------|-----------|---------------|------------|
| CoVe (NeoCoder) | +12.5% | +2~4% | +2~4% | +2~4% | -3%~+2% |
| DoLa (NeoCoder) | -2.5%~-1% | -1%~-0.5% | -1%~-0.5% | -1%~-0.5% | -2.5%~+3% |
| RAG (NeoCoder) | -3%~+1.5% | ~-5% | ~+3% | -0.5%~+1.5% | -2.5%~+2.5% |
| CoVe (CS4) | +5~8% | +5~8% | N/A | N/A | ~+2% |
| DoLa (CS4) | ~-8% | -3%~-2% | N/A | N/A | -3%~-2% |

**Key trends**:
- CoVe: **consistently enhances** divergent creativity (most models and settings)
- DoLa: **consistently suppresses** divergent creativity (all models and settings)
- RAG: **negligible impact** (positive and negative fluctuations cancel across models)

### Ablation Study

**Layer-level creativity correlation analysis** (Figure 4):

| Layer Position | Correlation with Creativity | Notes |
|----------------|----------------------------|-------|
| Early layers (1–8) | High correlation | Creativity-relevant layers concentrated here |
| Middle layers (9–20) | Moderate correlation | Transition region |
| Later layers (21–32) | Low / negative correlation | Anti-creativity layers concentrated here |

**Effect of inverted DoLa** (Figure 5):

| Configuration | Divergent Creativity | Convergent Creativity | Notes |
|---------------|---------------------|-----------------------|-------|
| Baseline | Reference | Reference | No intervention |
| DoLa (original) | Decreased | Unchanged | Contrasting early layers suppresses creativity |
| Inverted DoLa (amplifying creativity layers) | **Increased** | **Unchanged** | Divergent and convergent creativity can be decoupled |

### Key Findings

1. **CoVe enhances divergent creativity**: The verification process encourages broader exploration of the solution space, analogous to the cognitive mechanism by which questioning enhances human creativity.
2. **DoLa suppresses divergent creativity**: Because the early layers it contrasts encode more exploratory and divergent representations, subtracting them effectively suppresses creativity.
3. **RAG has no significant impact**: Likely attributable to retrieval quality issues—a semantic mismatch between CodeForces narrative scenarios and technical tutorial documents.
4. **Convergent creativity is largely unaffected**: None of the three methods significantly alters the model's ability to correctly solve constrained problems.
5. **These trends hold consistently across model families (LLaMA/Qwen/Mistral) and scales (1B–70B)**: suggesting that the creativity–hallucination relationship is an intrinsic property of LLMs.
6. **Divergent and convergent creativity appear to be decoupled**: one can be enhanced without degrading the other.

## Highlights & Insights

1. **Highly insightful research question**: The relationship between hallucination and creativity is a critically overlooked yet consequential issue, especially for AI4Science.
2. **Rigorous experimental design**: Comprehensive evaluation across models, scales, and benchmarks, with three independent runs averaged.
3. **The linear probe analysis** offers a new perspective on the layer-wise distribution of creativity in LLMs, identifying early layers as particularly critical.
4. **The inverted DoLa experiment** elegantly demonstrates that divergent and convergent creativity can be decoupled, with clear practical implications.
5. **Concrete guidance for AI4Science**: Prefer CoVe over DoLa to preserve creativity.

## Limitations & Future Work

1. Programming and story generation serve only as proxy tasks for scientific hypothesis generation; dedicated evaluation frameworks for scientific creativity are needed in future work.
2. No ablation study is conducted on why CoVe enhances creativity; the underlying mechanism remains unclear.
3. The null effect of RAG may stem from poor retrieval quality rather than an inherent limitation of RAG itself; alternative retrieval strategies were not examined.
4. The creativity enhancement via inverted DoLa is presented only as a preliminary result, lacking in-depth analysis and large-scale validation.
5. Framing hallucination and creativity as entirely opposed may be an oversimplification—more nuanced interactions between the two may exist.

## Related Work & Insights

- **CoVe** (Dhuliawala et al., 2023): Chain-of-Verification method, unexpectedly found to enhance creativity in this work.
- **DoLa** (Chuang et al., 2024): Layer-contrastive decoding, effective for factuality but detrimental to creativity.
- **ITI** (Li et al., 2024): Inference-Time Intervention; this paper borrows its linear probe methodology to analyze layer-level properties.
- **NeoCoder** (Lu et al., 2025): Constraint-driven programming creativity benchmark.
- Insight: "Creativity" and "factuality" in LLMs may be distributed across different network layers, providing a basis for designing more fine-grained decoding strategies; future hallucination mitigation methods should explicitly consider creativity preservation.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (First systematic study of hallucination mitigation's impact on creativity)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Comprehensive cross-model/scale/benchmark evaluation, but lacking deeper ablation)
- Writing Quality: ⭐⭐⭐⭐⭐ (Clear logic, excellent experimental organization)
- Value: ⭐⭐⭐⭐ (Important guidance for AI4Science, though scope of applicability may be limited)

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] "As Eastern Powers, I Will Veto." : An Investigation of Nation-Level Bias of Large Language Models in International Relations](as_eastern_powers_i_will_veto_an_investigation_of_nation-level_bias_of_large_lan.md)
- [\[ICLR 2026\] Token-Guard: Towards Token-Level Hallucination Control via Self-Checking Decoding](../../ICLR2026/information_retrieval/token-guard_towards_token-level_hallucination_control_via_self-checking_decoding.md)
- [\[AAAI 2026\] N2N-GQA: Noise-to-Narrative for Graph-Based Table-Text Question Answering Using LLMs](n2n-gqa_noise-to-narrative_for_graph-based_table-text_question_answering_using_l.md)
- [\[AAAI 2026\] SR-KI: Scalable and Real-Time Knowledge Integration into LLMs via Supervised Attention](sr-ki_scalable_and_real-time_knowledge_integration_into_llms_via_supervised_atte.md)
- [\[ACL 2026\] ChAIRO: Contextual Hierarchical Analogical Induction and Reasoning Optimization for LLMs](../../ACL2026/information_retrieval/chairo_contextual_hierarchical_analogical_induction_and_reasoning_optimization_f.md)

<!-- RELATED:END -->
