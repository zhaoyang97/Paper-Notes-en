---
title: >-
  [Paper Note] Hyperband-based Bayesian Optimization for Black-box Prompt Selection
description: >-
  [ICML 2025][LLM Evaluation][prompt selection] This work proposes the HbBoPs method, which combines a structure-aware deep kernel Gaussian process (separately encoding instructions and few-shot exemplars) with a Hyperband multi-fidelity scheduler. It achieves both sample efficiency and query efficiency in black-box LLM prompt selection, outperforming all SOTA methods across ten benchmarks and three LLMs.
tags:
  - "ICML 2025"
  - "LLM Evaluation"
  - "prompt selection"
  - "Bayesian optimization"
  - "Hyperband"
  - "multi-fidelity optimization"
  - "Gaussian process"
  - "deep kernel"
  - "black-box LLM"
date: 2026-05-08
content_hash: 2f942e38fa9bffc6
---

# Hyperband-based Bayesian Optimization for Black-box Prompt Selection

**Conference**: ICML 2025  
**arXiv**: [2412.07820](https://arxiv.org/abs/2412.07820)  
**Code**: Not released  
**Area**: LLM Evaluation  
**Keywords**: prompt selection, Bayesian optimization, Hyperband, multi-fidelity optimization, Gaussian process, deep kernel, black-box LLM

## TL;DR

This work proposes the HbBoPs method, which combines a structure-aware deep kernel Gaussian process (separately encoding instructions and few-shot exemplars) with a Hyperband multi-fidelity scheduler. It achieves both sample efficiency and query efficiency in black-box LLM prompt selection, outperforming all SOTA methods across ten benchmarks and three LLMs.

## Background & Motivation

LLMs are highly sensitive to input prompts, making optimal prompt selection crucial for downstream task performance. Under the black-box setting (accessing the LLM only via APIs without access to parameters, gradients, or token probabilities), prompt selection faces three major challenges:

**Huge search space**: Combinatorial explosion of instructions and few-shot exemplars (e.g., 5 instructions $\times$ 50 exemplars = 250 candidates)

**No gradient information**: Optimization via backpropagation is impossible

**High evaluation cost**: Each evaluation requires querying the LLM over the entire validation set

Key limitations of prior work:
- **EASE, TRIPLE-SH/GSE** are not designed for the joint selection of instructions and exemplars
- **No existing method simultaneously achieves both sample efficiency** (using a surrogate model to reduce the number of evaluated prompts) **and query efficiency** (using multi-fidelity to reduce the number of LLM calls per evaluation)

## Method

### Overall Architecture

HbBoPs consists of three core components:

1. **Structure-Aware Deep Kernel Gaussian Process**: Learns low-dimensional representations of prompts to predict performance
2. **Hyperband Multi-Fidelity Scheduler**: Controls the number of validation instances on which each prompt is evaluated
3. **Bayesian Optimization Proposal**: Uses the GP to guide candidate prompt selection within the Hyperband framework

### Structure-Aware Deep Kernel GP

Traditional methods encode the entire prompt as a single text block, ignoring the combinatorial structure of the prompt. This paper proposes to separately encode instructions and exemplars, leveraging structural information to enhance the prediction capability of the surrogate model.

Architecture of the feature extractor $\phi$:

- **Instruction encoder** $\phi_{\text{enc}(i)}$: Lin(d, 64) → ReLU → Lin(64, 32) → ReLU
- **Exemplar encoder** $\phi_{\text{enc}(e)}$: Same architecture but with independent parameters
- **Fusion network** $\phi_{(\phi_{\text{enc}(i)}, \phi_{\text{enc}(e)})}$: Lin(64, 32) → ReLU → Lin(32, 10)

Using an ARD Matérn 5/2 kernel, the kernel parameters $\theta$ and feature extractor parameters $\mathbf{w}$ are jointly optimized by maximizing the log marginal likelihood:

$$\hat{\theta}, \hat{\mathbf{w}} = \arg\max_{\theta, \mathbf{w}} -\mathbf{v}^\intercal \mathbf{K}_t(\theta, \mathbf{w})^{-1}\mathbf{v} - \log|\mathbf{K}_t(\theta, \mathbf{w})|$$

### Hyperband Multi-Fidelity Scheduling

The number of validation instances is treated as the fidelity parameter. Compared to Successive Halving (SH), Hyperband runs multiple brackets (different combinations of starting prompt numbers and starting budgets) to hedge against the "budget vs. configuration" dilemma.

Key design decisions:
- Higher-stage validation instances contain the lower-stage instances (expansion rather than resampling)
- Return the best prompt evaluated on the full validation set
- Lower bound $b_{\min} = 10$ validation instances, halving parameter $\eta = 2.0$

### Bayesian Optimization Proposal

In the first round of each bracket, the Expected Improvement (EI) acquisition function of GP is used instead of random sampling:

$$\alpha_{\text{EI}}(p|\mathcal{D}_{t|b}) = \mathbb{E}[\max\{v_{\min,b} - f(\mathbf{z}_p), 0\}]$$

The GP is trained online—accumulating training data step-by-step during the selection process without relying on pre-trained surrogate models. The GP is trained using the highest fidelity level that has at least 4 observations.

## Key Experimental Results

### Experimental Setup

- **Benchmarks**: 10 tasks (ARC, GSM8K, 8 BIG-bench/Instruction Induction tasks)
- **LLMs**: Claude 3 Haiku, Llama3 8B Instruct, Mistral 7B Instruct
- **Search Space**: 5 instructions $\times$ 50 exemplars = 250 candidate prompts
- **Budget**: LLM calls equivalent to 25 full-fidelity evaluations

### Main Results

Average normalized test error under the full budget:

| Method | Test Error | Type |
|------|----------|------|
| RS (Random Search) | 0.214 | Baseline |
| HDBO | 0.185 | Full-fidelity SOTA |
| TRIPLE-GSE | 0.158 | Multi-fidelity SOTA |
| TRIPLE-SH | 0.159 | Multi-fidelity |
| **HbBoPs** | **0.150** | **Ours** |

### Analysis by LLM (Median Relative Improvement of HbBoPs vs. TRIPLE-SH)

| Budget Ratio | Claude 3 Haiku | Llama3 8B | Mistral 7B |
|----------|----------------|-----------|------------|
| 0.25 | 6.6% (test) | 3.6% (test) | 3.9% (test) |
| 0.50 | 2.7% (test) | 1.0% (test) | 1.6% (test) |
| 1.00 | -0.6% (test) | 0.0% (test) | 0.5% (test) |

### Key Findings

- **More significant advantage under low budgets**: Under 0.25 budget, HbBoPs outperforms HDBO by ~35% and TRIPLE-SH by ~24%
- **Optimal anytime performance**: Consistently leads throughout the entire optimization process
- Structure-aware encoding (separately encoding instructions and exemplars) significantly outperforms monolithic encoding

### Ablation Study

- Removing deep kernel → Performance degradation (vanilla GP cannot handle high-dimensional embeddings)
- Removing Hyperband (switching to full fidelity) → Significant drop in query efficiency
- Replacing the encoder (BERT → other models) → The method is robust to the choice of encoder

## Highlights & Insights

1. **Simultaneously resolving two efficiency bottlenecks**: For the first time, sample efficiency (surrogate model evaluates fewer prompts) and query efficiency (multi-fidelity calls the LLM fewer times) are unified in prompt selection.

2. Crucial role of **structure-aware representation**: Treating the prompt as a combinatorial structure of instruction + exemplar rather than a monolithic text block significantly enhances the surrogate model's prediction capability.

3. **High practicality**: The method is trained completely online and does not require pre-trained surrogate models; it is applicable to any black-box LLM.

4. **Hyperband outperforms SH**: The multi-bracket strategy effectively hedges against the uncertainty of "exploration vs. exploitation".

## Limitations & Future Work

- The search space is fixed at 250 candidate prompts; a larger search space has not been tested
- Only the 5-shot setting is considered; the impact of different shot numbers has not been explored
- The code is not publicly released, which remains to be verified for reproducibility
- No direct comparison with LLM-based prompt generation methods

## Related Work

- **Prompt Optimization/Selection**: APE (Zhou 2023), DSPy/MIPROv2 (Opsahl-Ong 2024), EASE (Wu 2024), TRIPLE (Shi 2024)
- **Bayesian Optimization**: Deep Kernel GP (Wilson 2016), High-dimensional BO (Hvarfner 2024)
- **Multi-fidelity Optimization**: Hyperband (Li 2018), BOHB (Falkner 2018)

## Rating

⭐⭐⭐⭐ — Elegant method design that systematically integrates mature techniques from the HPO domain (Hyperband + BO + deep kernel GP) into the prompt selection problem, backed by thorough experimentation. The core contribution lies in framework integration rather than a single technical breakthrough.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Predicting the Performance of Black-Box LLMs through Follow-Up Queries](../../NeurIPS2025/llm_evaluation/predicting_the_performance_of_black-box_llms_through_follow-up_queries.md)
- [\[ICML 2025\] Sample Efficient Demonstration Selection for In-Context Learning](sample_efficient_demonstration_selection_for_in-context_learning.md)
- [\[ICML 2026\] Investigating Advanced Reasoning of Large Language Models via Black-Box Environment Interaction](../../ICML2026/llm_evaluation/investigating_advanced_reasoning_of_large_language_models_via_black-box_environm.md)
- [\[ICLR 2026\] Prompt and Parameter Co-Optimization for Large Language Models](../../ICLR2026/llm_evaluation/prompt_and_parameter_co-optimization_for_large_language_models.md)
- [\[ICML 2025\] The Best of Both Worlds: Bridging Quality and Diversity in Data Selection with Bipartite Graph](the_best_of_both_worlds_bridging_quality_and_diversity_in_data_selection_with_bi.md)

</div>

<!-- RELATED:END -->
