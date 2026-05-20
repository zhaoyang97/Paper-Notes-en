---
title: >-
  [Paper Note] KTAE: A Model-Free Algorithm to Key-Tokens Advantage Estimation in Mathematical Reasoning
description: >-
  [NeurIPS 2025][Model Compression][token-level advantage estimation] KTAE proposes a model-free token-level advantage estimation algorithm that quantifies the statistical association between each token and correct reasoni…
tags:
  - "NeurIPS 2025"
  - "Model Compression"
  - "token-level advantage estimation"
  - "GRPO"
  - "DAPO"
  - "mathematical reasoning"
  - "reinforcement learning"
date: 2026-05-08
content_hash: b91f629d414ec851
---

# KTAE: A Model-Free Algorithm to Key-Tokens Advantage Estimation in Mathematical Reasoning

**Conference**: NeurIPS 2025
**arXiv**: [2505.16826](https://arxiv.org/abs/2505.16826)  
**Code**: [GitHub](https://github.com/ZNLP/KTAE)  
**Area**: Model Compression
**Keywords**: token-level advantage estimation, GRPO, DAPO, mathematical reasoning, reinforcement learning

## TL;DR
KTAE proposes a model-free token-level advantage estimation algorithm that quantifies the statistical association between each token and correct reasoning outcomes via Fisher's exact test and information gain. The resulting fine-grained token importance is superimposed on the rollout-level advantage of GRPO/DAPO, achieving superior performance on five mathematical reasoning benchmarks while significantly reducing generation length.

## Background & Motivation

**Background**: GRPO is currently the dominant reinforcement learning algorithm for LLMs, eliminating the dependency on a Critic model through group-relative policy optimization. DAPO further extends GRPO with Clip-Higher, dynamic sampling, and other enhancements.

**Limitations of Prior Work**: GRPO/DAPO compute rollout-level advantages, assigning an identical advantage value $\hat{A}_{i,t} = \frac{R_i - \text{mean}(\mathbf{R})}{\text{std}(\mathbf{R})}$ to every token within the same reasoning chain. However, in mathematical reasoning, different tokens contribute very differently — an erroneous reasoning chain may deviate from the correct path only in its final steps.

**Key Challenge**: Uniform advantage prevents the model from precisely identifying which tokens constitute critical reasoning steps and which are irrelevant (e.g., "First", "denote"), thereby impeding effective learning.

**Goal**: Estimate fine-grained per-token advantage values without introducing any additional model.

**Key Insight**: By leveraging the correct/incorrect labels of multiple rollouts, a contingency table is constructed for each token, and statistical methods are applied to quantify the strength and direction of its association with correct outcomes.

**Core Idea**: The statistical association between token occurrence and rollout correctness — quantified via Fisher's exact test, information gain, and BM25-based frequency analysis — is converted into token-level advantages and added to the GRPO advantage.

## Method

### Overall Architecture
Given a question $q$, $G$ rollouts $\{o_1, \dots, o_G\}$ are sampled from the policy model, each associated with a rule-based reward $\{R_1, \dots, R_G\}$. KTAE constructs the association between each token $o_{ij}$ and correct rollouts, ultimately producing a token-level advantage $\hat{A}_{o_{ij}}^{KTAE}$ that replaces the uniform GRPO advantage in the policy gradient update.

### Key Designs

1. **Token-Level Contingency Table Construction**:

    - Function: Quantifies the occurrence pattern of each token across correct and incorrect rollouts.
    - Mechanism: The $G$ rollouts are partitioned into a correct set $x_T$ and an incorrect set $x_F$. For each token $o_{ij}$, the following counts are recorded: appearances in correct rollouts ($a$), appearances in incorrect rollouts ($b$), non-appearances in correct rollouts ($c$), and non-appearances in incorrect rollouts ($d$), forming a $2 \times 2$ contingency table.
    - Design Motivation: Contingency tables are a classical tool for statistical association analysis, naturally suited to characterizing the relationship between token occurrence and rollout correctness.

2. **Association Strength Quantification — Fisher's Exact Test**:

    - Function: Tests whether a significant association exists between token occurrence and rollout correctness.
    - Core Formula: $Fisher(o_{ij}) = \frac{(a+b)!(c+d)!(a+c)!(b+d)!}{a!b!c!d!N!}$, computed in log space to handle large factorials.
    - Transformation: $\mathcal{F}(o_{ij}) = e^{-2 \cdot Fisher(o_{ij})}$ (equals 0 when $p=1$, approaches 1 as $p \to 0$).
    - Rationale for Fisher over Chi-squared: When the sample size $G$ is small (e.g., 8 or 16), Fisher's exact test provides exact probabilities, whereas the chi-squared approximation is unreliable.

3. **Association Strength Quantification — Information Gain**:

    - Function: Provides a complementary information-theoretic quantification of association.
    - Core Formula: $IG(o_{ij}) = H(Y) - H(Y|X_{o_{ij}})$, where $H(Y)$ is the entropy of rollout correctness and $H(Y|X_{o_{ij}})$ is the conditional entropy given whether the token appears.
    - Combined Score: $h_1 \cdot \mathcal{F}(o_{ij}) + h_2 \cdot IG(o_{ij})$ serves as the overall association strength.

4. **Association Direction Quantification**:

    - Function: Determines whether a token is positively or negatively correlated with correct outcomes.
    - Mechanism: Inspired by BM25 term frequency, normalized token frequencies $TF_{T/F}(o_{ij}) = \frac{(k_1+1) \cdot tf_{T/F}(o_{ij})}{k_1(1-b+b \times \frac{len_{T/F}}{len_{avg}})+tf_{T/F}(o_{ij})}$ are computed over the concatenated correct and incorrect rollout sequences.
    - Direction Score: $D(o_{ij}) = (\arcsin\sqrt{\frac{a}{a+c}} - \arcsin\sqrt{\frac{b}{b+d}}) + h_3(\frac{TF_T}{TF_F} - \frac{TF_F}{TF_T})$, combining Cohen's $h$ effect size with frequency ratios.
    - Design Motivation: High-frequency generic tokens are distinguished by proportion differences, while low-frequency key tokens are distinguished by frequency ratios.

5. **Final Token-Level Advantage**:

    - The association strength multiplied by the direction score yields the key-token-value, which is sigmoid-normalized and added to the GRPO advantage:
    - $\hat{A}_{o_{ij}}^{KTAE} = \hat{A}_{o_i}^{GRPO} + \sigma\big((h_1 \cdot \mathcal{F}(o_{ij}) + h_2 \cdot IG(o_{ij})) \cdot D_{o_{ij}}\big) - 0.5$

### Loss & Training
- KTAE serves as a plug-and-play module that directly replaces the advantage computation step in GRPO/DAPO.
- No additional models are introduced; computational overhead stems primarily from contingency table statistics, which depend only on the number of tokens and not on model size.
- KTAE is orthogonal to DAPO's Clip-Higher, dynamic sampling, and other enhancements, and can be combined with them.

## Key Experimental Results

### Main Results (7B model, 5 mathematical benchmarks, zero-shot greedy pass@1)

| Method | AIME24 | MATH-500 | AMC | Minerva | OlympiadBench | Avg |
|--------|--------|----------|-----|---------|---------------|-----|
| GRPO-7B | 36.7 | 81.0 | 57.8 | 32.7 | 43.2 | 50.3 |
| **GRPO+KTAE-7B** | 33.3 | **82.4** | **65.1** | **33.8** | **43.7** | **51.7** |
| DAPO-7B | 36.7 | 81.8 | 60.2 | 34.5 | 45.3 | 51.7 |
| **DAPO+KTAE-7B** | 36.7 | **83.2** | **63.9** | **35.3** | 43.7 | **52.5** |
| R1-Distill-Qwen-1.5B | 20.0 | 77.4 | 49.4 | 25.0 | 35.8 | 41.5 |
| DAPO+KTAE-1.5B | 20.0 | 77.6 | 50.6 | 29.0 | 40.0 | **43.4** |

DAPO+KTAE-7B achieves the highest average score of 52.5, and KTAE-1.5B surpasses R1-Distill-1.5B trained on the same base model.

### Generation Length Comparison (average response length, 7B model)

| Method | AIME24 | MATH-500 | AMC | Minerva | OlympiadBench | Avg |
|--------|--------|----------|-----|---------|---------------|-----|
| GRPO-7B | 989 | 606 | 806 | 641 | 813 | 771.0 |
| **GRPO+KTAE-7B** | 941 | **563** | **741** | **577** | **771** | **718.6** |
| DAPO-7B | 1155 | 676 | 969 | 700 | 986 | 897.2 |
| **DAPO+KTAE-7B** | 1013 | **604** | **864** | **607** | **798** | **777.2** |

Without any length-penalty reward, KTAE significantly reduces generation length (GRPO+KTAE averages 52 fewer tokens), achieving greater reasoning efficiency.

### Ablation Study

| Configuration | Effect |
|---------------|--------|
| Remove IG | Largest accuracy drop; generates shortest sequences |
| Remove $\mathcal{F}$ (Fisher) | Accuracy drops; length slightly increases |
| Remove tf (frequency analysis) | Accuracy drops; length increases; initial entropy rises markedly |
| GRPO baseline | Entropy collapse observed |
| KTAE (full) | Entropy collapse avoided; all components are indispensable |

### Key Findings
- IG contributes most to accuracy and is the core component of KTAE.
- KTAE effectively prevents the entropy collapse seen in GRPO; entropy in DAPO+KTAE continues to increase throughout training.
- Visualizations demonstrate that KTAE accurately distinguishes key reasoning tokens such as "complement" and "ratio" from irrelevant tokens such as "First" and "denote".
- In rollouts with incorrect annotations (correct answers but failed format parsing), KTAE still correctly highlights positively contributing tokens.

## Highlights & Insights
- **Fine-grained signals with zero additional model cost**: Unlike process reward models (PRMs), which require extra training, KTAE obtains token-level signals purely through statistical analysis — lightweight and less susceptible to reward hacking.
- **Accuracy↑ + Length↓ simultaneously**: Achieving shorter reasoning chains without any length penalty demonstrates that token-level advantages genuinely guide the model to focus on critical reasoning steps and reduce redundant generation.
- **Elegant combination of statistical methods**: Fisher's exact test (precise association testing), information gain (information-theoretic complement), and BM25-style term frequency (direction estimation) each play a distinct role, yielding strong theoretical interpretability.
- **Transferable framework**: The contingency table and statistical association paradigm is not limited to mathematical reasoning and could in principle be extended to domains with verifiable rewards, such as code generation and logical reasoning.

## Limitations & Future Work
- Validation is currently limited to mathematical reasoning; applicability to other CoT-dependent domains (code generation, logical reasoning) remains unexplored.
- KTAE's computation currently utilizes less than 1% GPU capacity (primarily serial CPU execution), leaving substantial room for engineering optimization.
- The impact of $G$ (number of rollouts) is not thoroughly analyzed; statistical power of Fisher's exact test may be insufficient for small $G$.
- Hyperparameters $h_1, h_2, h_3$ require tuning, and no sensitivity analysis is provided.
- Experiments are conducted exclusively on the Qwen2.5-Math model family; generalization across model families remains to be verified.

## Related Work & Insights
- **vs. GRPO/DAPO**: The original GRPO/DAPO assign uniform rollout-level advantages; KTAE refines these to the token level via statistical analysis, constituting an orthogonal enhancement.
- **vs. Process Reward Models (PRMs)**: PRMs require training an additional reward model, incurring high cost and susceptibility to reward hacking; KTAE is model-free and statistics-based, making it more lightweight and controllable.
- **vs. DeepSeek R1**: R1 induces long CoT and self-reflection through pure RL; KTAE can serve as a complementary source of finer-grained training signals.

## Rating
- Novelty: ⭐⭐⭐⭐ Introducing statistical hypothesis testing into RL advantage estimation is a novel perspective, though contingency tables and Fisher's exact test are themselves classical methods.
- Experimental Thoroughness: ⭐⭐⭐⭐ Five benchmarks, 1.5B and 7B models, ablations, and visualizations are all included, but cross-family model experiments are absent.
- Writing Quality: ⭐⭐⭐⭐ Method derivation is clear and figures are intuitive, though notation is occasionally dense.
- Value: ⭐⭐⭐⭐ A plug-and-play improvement with strong practical utility and meaningful impact on RL training efficiency.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] DeltaFlow: An Efficient Multi-frame Scene Flow Estimation Method](deltaflow_an_efficient_multi-frame_scene_flow_estimation_method.md)
- [\[ICCV 2025\] B-VLLM: A Vision Large Language Model with Balanced Spatio-Temporal Tokens](../../ICCV2025/model_compression/b_vllm_a_vision_large_language_model_with_balanced_spatio_temporal_tokens.md)
- [\[NeurIPS 2025\] QSVD: Efficient Low-Rank Approximation for Unified Query-Key-Value Weight Compression](qsvd_efficient_low-rank_approximation_for_unified_query-key-value_weight_compres.md)
- [\[NeurIPS 2025\] Reject Only Critical Tokens: Pivot-Aware Speculative Decoding](reject_only_critical_tokens_pivot-aware_speculative_decoding.md)
- [\[NeurIPS 2025\] C-LoRA: Contextual Low-Rank Adaptation for Uncertainty Estimation in Large Language Models](c-lora_contextual_low-rank_adaptation_for_uncertainty_estimation_in_large_langua.md)

</div>

<!-- RELATED:END -->
