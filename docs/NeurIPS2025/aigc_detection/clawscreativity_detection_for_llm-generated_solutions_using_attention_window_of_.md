---
title: >-
  [Paper Note] CLAWS: Creativity Detection for LLM-Generated Solutions Using Attention Window of Sections
description: >-
  [NeurIPS 2025][AIGC Detection][LLM creativity detection] This paper proposes CLAWS, a method that analyzes the attention weight distribution of LLMs across different prompt sections during mathematical solution generation to classify outputs as "creative," "typical," or "hallucinated," without requiring human evaluation.
tags:
  - "NeurIPS 2025"
  - "AIGC Detection"
  - "LLM creativity detection"
  - "attention analysis"
  - "mathematical reasoning"
  - "hallucination detection"
  - "white-box method"
date: 2026-05-08
content_hash: ecff49e63bfa6fd9
---

# CLAWS: Creativity Detection for LLM-Generated Solutions Using Attention Window of Sections

**Conference**: NeurIPS 2025
**arXiv**: [2510.17921](https://arxiv.org/abs/2510.17921)  
**Code**: [GitHub](https://github.com/kkt94/CLAWS)  
**Area**: AIGC Detection
**Keywords**: LLM creativity detection, attention analysis, mathematical reasoning, hallucination detection, white-box method

## TL;DR

This paper proposes CLAWS, a method that analyzes the attention weight distribution of LLMs across different prompt sections during mathematical solution generation to classify outputs as "creative," "typical," or "hallucinated," without requiring human evaluation.

## Background & Motivation

Recent advances in reinforcement learning–trained reasoning language models (RLMs) have yielded significant progress in mathematical problem solving, yet the evaluation of their creativity remains largely overlooked. Existing research on creativity assessment focuses predominantly on writing tasks (e.g., TTCW), with little attention to creativity in mathematical reasoning.

This gap arises from two core challenges:

**Difficulty in defining creativity**: There is no consensus on what constitutes a "creative" mathematical solution.

**Reliance on human evaluation**: Assessing mathematical creativity requires high-level domain experts, making large-scale annotation prohibitively expensive.

Furthermore, overly suppressing LLM outputs to avoid hallucinations may simultaneously stifle creative responses. Distinguishing "creative solutions" from "hallucinated solutions" is therefore critical to maximizing the utility and diversity of LLM outputs.

## Method

### Overall Architecture

The CLAWS framework consists of four stages:
1. **Generation**: An RLM generates mathematical solutions given a structured prompt.
2. **Feature Extraction**: Features are extracted from attention weights during generation.
3. **LLM Evaluator Labeling**: Two evaluators—GPT-o4-mini and Gemini-1.5-Pro—assign ground-truth labels.
4. **Detection Method Evaluation**: Classification performance of various methods is compared based on the extracted features.

### Key Designs

**Structured Prompt Segmentation**: The input prompt $X = G|P|S|I$ is divided into five semantic sections:
- Guideline ($G$): Criteria for creativity evaluation
- Problem ($P$): The mathematical problem to be solved
- Reference Solutions ($S$): One or more typical reference solutions
- Instruction ($I$): Directive to generate a novel solution
- Response ($R$): The model-generated answer

**Attention Weight Matrix Construction**: For each attention head $h$ of the last layer $L$, at decoding timestep $t$, the attention vector is extracted as:

$$A_{t,h}^{(L)} = [a_{1,h}^{(L)} \; a_{2,h}^{(L)} \cdots a_{k+t,h}^{(L)}], \quad k = \text{len}(X)$$

All timesteps are stacked and padded to a uniform dimension: $A_h^{(L)} \in \mathbb{R}^{T \times (\text{len}(X) + T)}$

**Average Attention per Section** (AVGA):

$$\text{AVGA}_{\mathcal{U}} = \frac{1}{H \cdot T \cdot |\mathcal{I}_{\mathcal{U}}|} \sum_{h=1}^{H} \sum_{t=1}^{T} \sum_{i \in \mathcal{I}_{\mathcal{U}}} A_h^{(L)}[t, i]$$

where $\mathcal{U} \in \{G, P, S, I, R\}$ and $\mathcal{I}_{\mathcal{U}}$ denotes the set of token indices for each section.

**Normalized Attention Ratio** (CLAWS feature):

$$\text{CLAWS}_{\mathcal{U}} = \frac{\text{AVGA}_{\mathcal{U}}}{\sum_{\mathcal{U}' \in \{G,P,S,I,R\}} \text{AVGA}_{\mathcal{U}'}}$$

### Loss & Training

CLAWS itself is a training-free feature extraction method. When combined with downstream classifiers (XGBoost / MLP / TabM), standard cross-entropy loss is used for three-class training.

**Three-class label definitions**:
- **Hallucinated solution**: Neither evaluator judges the solution as "correct."
- **Creative solution**: Both evaluators judge the solution as "correct," and at least one judges it as "creative."
- **Typical solution**: Both evaluators judge the solution as "correct," and neither judges it as "creative."

## Key Experimental Results

### Main Results

CLAWS is evaluated against five baseline methods across five RLMs (7–8B parameters) using the Prototype strategy:

| Model | Method | TEST F1w | AMC F1w | AIME F1w | A(J)HSME F1w |
|-------|--------|----------|---------|----------|--------------|
| DeepSeek | Perplexity | 48.09 | 44.56 | 55.93 | 42.34 |
| DeepSeek | **CLAWS** | **58.66** | **46.71** | **56.90** | **38.82** |
| Mathstral | Hidden Score | 49.86 | 37.37 | 65.96 | 33.42 |
| Mathstral | **CLAWS** | **63.20** | **51.47** | **65.25** | **49.13** |
| OpenMath2 | Window Entropy | 40.89 | 43.44 | 40.55 | 42.45 |
| OpenMath2 | **CLAWS** | Outperforms all baselines | — | — | — |

### Ablation Study

- Visualization of CLAWS's five-dimensional features reveals distinct attention distribution patterns across the three solution categories.
- Creative solutions attend more to the Reference Solutions section, whereas hallucinated solutions exhibit anomalously high self-attention on the Response section.
- Performance improves further when CLAWS is combined with the TabM classifier.

### Key Findings

1. CLAWS outperforms all five baselines on F1w, F1m, APm, and AUROC across most models and datasets.
2. CLAWS is the only method that consistently achieves genuine three-class classification rather than degenerating into binary classification.
3. High-creativity models (e.g., Qwen) exhibit systematic differences in attention allocation to the Guideline and Solutions sections compared to low-creativity models.
4. Performance improves further when CLAWS is paired with the TabM classifier.

## Highlights & Insights

- **Core Insight**: Creative generation depends on differential attention to different parts of the prompt—creative solutions attend more to reference solutions and the problem statement, while hallucinated solutions exhibit anomalously high attention to their own responses.
- **Practical Value**: CLAWS requires only a single inference pass for detection, with no additional model calls or multiple generations needed.
- **Three-Class Capability**: CLAWS is the first method to simultaneously detect "creative / typical / hallucinated" outputs, rather than performing binary hallucination classification alone.
- **Generalizability**: Consistent advantages are maintained across five RLMs and multiple mathematical competition datasets.

## Limitations & Future Work

1. Validation is limited to mathematical reasoning tasks; extension to other reasoning domains such as code generation or scientific reasoning has not been explored.
2. Creativity labels rely on the evaluation quality of GPT-o4-mini and Gemini-1.5-Pro rather than true human annotation.
3. Only models with 7–8B parameters are tested; attention pattern differences in larger models remain unknown.
4. The Creative class is substantially underrepresented relative to the other two classes (class imbalance), which negatively affects detection performance.
5. Only the last-layer attention weights are utilized; aggregating across multiple layers may yield richer features.

## Related Work & Insights

- **Relation to Hallucination Detection**: Conventional methods (e.g., SelfCheckGPT, INSIDE) require multiple generations or external models, whereas CLAWS enables single-pass white-box detection.
- **Connection to RL-Based Reasoning**: Creativity differences exhibited by RL-trained reasoning models can be quantified through attention patterns.
- **Future Directions**: The section-level attention analysis of CLAWS could be generalized to creativity evaluation in code generation, or applied to filter high-quality creative solutions as training data during RL training.

## Rating

- ⭐ Novelty: 4/5 — First systematic treatment of creativity definition and detection in mathematical reasoning from an attention-segmentation perspective.
- ⭐ Value: 4/5 — A white-box detection scheme with zero additional computational overhead that is straightforward to integrate in practice.
- ⭐ Experimental Thoroughness: 4/5 — Comprehensive evaluation across 5 models, 4 datasets, and 5 evaluation strategy combinations.
- ⭐ Writing Quality: 3/5 — The framework is clearly presented, but the heavy notation load means some experimental details require consulting the appendix.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Death of the Novel(ty): Beyond n-Gram Novelty as a Metric for Textual Creativity](../../ICLR2026/aigc_detection/death_of_the_novelty_beyond_n-gram_novelty_as_a_metric_for_textual_creativity.md)
- [\[ACL 2025\] Learning to Rewrite: Generalized LLM-Generated Text Detection](../../ACL2025/aigc_detection/learning_to_rewrite_generalized_llm-generated_text_detection.md)
- [\[ICLR 2026\] RelayFormer: A Unified Local-Global Attention Framework for Scalable Image and Video Manipulation Localization](../../ICLR2026/aigc_detection/relayformer_a_unified_local-global_attention_framework_for_scalable_image_and_vi.md)
- [\[NeurIPS 2025\] Classical Planning with LLM-Generated Heuristics: Challenging the State of the Art with Python Code](classical_planning_with_llm-generated_heuristics_challenging_the_state_of_the_ar.md)
- [\[NeurIPS 2025\] DuoLens: A Framework for Robust Detection of Machine-Generated Multilingual Text and Code](duolens_a_framework_for_robust_detection_of_machine-generated_multilingual_text_.md)

</div>

<!-- RELATED:END -->
