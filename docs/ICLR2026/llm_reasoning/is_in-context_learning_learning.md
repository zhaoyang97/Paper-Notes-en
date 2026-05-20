---
title: >-
  [Paper Note] Is In-Context Learning Learning?
description: >-
  [ICLR 2026][LLM Reasoning][in-context learning] This paper systematically investigates whether ICL constitutes genuine "learning" through large-scale controlled experiments. It demonstrates that ICL satisfies the formal…
tags:
  - "ICLR 2026"
  - "LLM Reasoning"
  - "in-context learning"
  - "ICL"
  - "memorisation"
  - "distributional shift"
  - "generalization"
  - "autoregressive models"
date: 2026-05-08
content_hash: af10d9e8362bd48b
---

# Is In-Context Learning Learning?

**Conference**: ICLR 2026
**arXiv**: [2509.10414](https://arxiv.org/abs/2509.10414)  
**Code**: Not open-sourced  
**Area**: LLM Reasoning
**Keywords**: in-context learning, ICL, memorisation, distributional shift, generalization, autoregressive models

## TL;DR

This paper systematically investigates whether ICL constitutes genuine "learning" through large-scale controlled experiments. It demonstrates that ICL satisfies the formal mathematical definition of learning, yet empirical evidence reveals its generalization capacity to be limited — models primarily exploit structural regularities within the prompt via deduction rather than acquiring new capabilities from the provided demonstrations.

## Background & Motivation

**Background**: In-context learning (ICL) enables autoregressive language models to solve downstream tasks via next-token prediction without parameter updates, requiring only a small number of exemplars in the prompt. This capability has sparked extensive debate on whether LLMs can "learn" unseen tasks from a handful of demonstrations.

**Limitations of Prior Work**:

- Existing research conflates "deduction" with "learning" — the two are not equivalent
- ICL does not explicitly encode the provided observations, but instead relies on the model's prior knowledge and the prompt exemplars
- Prior studies lack systematic control over confounding factors such as memorisation, pretraining data leakage, and distributional shift
- It remains unclear whether strong ICL performance stems from genuinely learning from demonstrations or from prior knowledge retrieval combined with pattern matching

**Key Challenge**: Although ICL formally resembles learning (inferring task rules from in-prompt exemplars), it is uncertain whether the autoregressive encoding mechanism provides sufficient inductive bias to support robust generalization and genuine knowledge acquisition — a question with direct implications for how LLMs should be understood and deployed.

**Goal**: The paper addresses the question "Is ICL learning?" at both theoretical and empirical levels. It first formally proves that ICL satisfies the mathematical definition of learning, then conducts large-scale controlled ablation studies to characterize the practical boundaries of ICL's learning capacity, systematically eliminating confounds including memorisation, pretraining leakage, distributional shift, and prompt style.

## Method

### Overall Architecture

The paper adopts a dual-track strategy combining theoretical analysis and large-scale empirical investigation. The theoretical component argues that ICL meets the formal criteria for learning (analogous to the PAC learning framework); the empirical component systematically ablates or controls multiple confounding factors to reveal the true boundaries of ICL's learning capacity. Experiments span diverse model architectures, prompt styles, task types, and data distribution settings, constituting one of the largest controlled studies of ICL behavior to date.

### Key Design 1: Disentangling Memorisation from Pretraining Effects

The central goal is to separate the contribution of pretraining memorisation from genuine learning attributable to the prompt exemplars. Specific approaches include:

- **Benchmark Contamination Detection**: Multiple methods are applied to assess whether pretraining data already encodes test task information, quantifying the contribution of memorisation to ICL performance
- **Zero-shot vs. Few-shot Delta Analysis**: Zero-shot performance reflects pure prior knowledge; only the increment over zero-shot can potentially be attributed to "learning from demonstrations"
- **Counterfactual Label Experiments**: Random or inverted labels are used to test whether the model genuinely exploits the input–output mapping within the demonstrations

Formally, let $f$ denote the target function and $\hat{f}_{\text{ICL}}$ the ICL prediction function. The learning gain of ICL is defined as:

$$\Delta_{\text{learn}} = \mathbb{E}[\mathcal{L}(\hat{f}_{\text{zero-shot}})] - \mathbb{E}[\mathcal{L}(\hat{f}_{\text{few-shot}})]$$

where $\mathcal{L}$ is the task loss. After adequately controlling for memorisation, $\Delta_{\text{learn}}$ is found to decrease substantially, indicating that a large portion of ICL performance derives from prior knowledge rather than learning from exemplars.

### Key Design 2: Distributional Shift and Exemplar Scaling Behavior Analysis

The core experimental methodology involves systematically varying the following factors and observing the resulting scaling behavior of ICL accuracy:

- **Number of exemplars $k$**: from $k=0$ (zero-shot) to many-shot, tracking accuracy as a function of $k$
- **Exemplar distribution**: varying class balance, sample difficulty, selection strategy, and presentation order
- **Prompt style**: standard few-shot, chain-of-thought (CoT), and diverse template formats and phrasings
- **Model selection**: autoregressive models of multiple scales and architectures

A key finding is that as $k$ increases, accuracy converges to a limit that is largely independent of the specific configuration. Formally:

$$\lim_{k \to \infty} \text{Acc}(k; \mathcal{D}, \mathcal{M}, \mathcal{S}) \approx C_{\text{task}}$$

where $\mathcal{D}$ denotes the exemplar distribution, $\mathcal{M}$ the model, and $\mathcal{S}$ the prompt style. This finding contrasts with the expectation under genuine learning, whereby performance should continue to improve with more or better data.

### Key Design 3: Distributional Sensitivity of Chain-of-Thought

The paper specifically analyzes ICL behavior under CoT prompting. Although CoT substantially improves accuracy on certain tasks, it exhibits greater sensitivity to the distributional properties and format of the prompt. This suggests that CoT improvements do not arise from deeper task learning, but rather from exploiting the structural regularity of reasoning chains to perform pattern deduction more efficiently. Key observations include:

- Standard few-shot performance is relatively stable but subject to a low accuracy ceiling
- CoT performance is more variable and highly dependent on the format and structure of the reasoning chain
- On tasks that are formally similar yet semantically distinct, CoT accuracy diverges substantially
- ICL essentially extracts patterns from the statistical regularity of the prompt rather than encoding new knowledge

## Key Experimental Results

### Main Results: Systematic Evaluation of ICL Learning Capacity

| Controlled Variable | Core Finding | Key Evidence |
|:--------------------|:-------------|:-------------|
| Memorisation | Pretraining memorisation substantially contributes to ICL performance | Performance drops markedly on contamination-free benchmarks |
| Number of exemplars | Accuracy saturates rapidly | Gains become negligible beyond $k > 16$; logarithmic saturation curve |
| Exemplar distribution | Distribution-insensitive at the limit | Convergence values are similar across different class/difficulty distributions |
| Model selection | Inter-model differences diminish at the limit | Comparison across multiple architectures and scales |
| Prompt style | Standard few-shot is insensitive; CoT is sensitive | Format variation experiments show larger CoT fluctuations |
| Input linguistic features | Surface features have limited impact on asymptotic performance | Paraphrasing and format changes produce minimal accuracy differences |

### Ablation Study: Analysis of ICL Information Utilization Mechanisms

| Experimental Condition | Accuracy | Core Implication |
|:-----------------------|:---------|:-----------------|
| Correct labels | Baseline (highest) | Standard ICL performance |
| Random labels | Marginal drop | Model does not fully rely on the input–output mapping in demonstrations |
| Counterfactual labels | Moderate drop | Model partially uses label information but it is not the primary cue |
| No labels (input format only) | Approaches few-shot | Structural features of the prompt matter more than label content |
| Shuffled exemplar order | Negligible change | Model is insensitive to exemplar presentation order |

## Rating

**Rating**: ⭐⭐⭐⭐

**Strengths**:

- Raises an important question about the nature of ICL, with both theoretical depth and empirical breadth
- Rigorous experimental design: systematically eliminates memorisation, distributional shift, prompt style, and other confounds
- The core finding — that accuracy saturates with more exemplars and is insensitive to hyperparameter choices — carries significant theoretical and practical implications
- The distributional sensitivity analysis of CoT offers a novel perspective for understanding CoT mechanisms

**Weaknesses**:

- The paper arrives primarily at negative conclusions (limited ICL learning capacity) without offering constructive improvements or solutions
- Some experiments may be subject to selection bias in the benchmark task set; generalizability to more complex reasoning and generation tasks requires further validation
- "Learning" admits multiple formulations (PAC learning, Bayesian learning, generalization-theoretic learning), and the robustness of the conclusions across these definitions is not fully discussed
- A systematic scaling law analysis of how ICL behavior varies across model sizes is absent

**Key Distinctions from Related Work**:

- Unlike work that explains ICL through Bayesian inference (Xie et al., 2021), this paper systematically challenges the hypothesis that ICL constitutes a robust learning mechanism from an empirical learning-theoretic perspective
- Unlike single-factor analyses, this paper simultaneously controls for memorisation, distribution, model, and prompt style, yielding more comprehensive and reliable negative conclusions
- Unlike purely theoretical analyses, this paper combines mathematical argumentation with large-scale experiments, substantially strengthening the persuasiveness of its conclusions

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Unlabeled Data Can Provably Enhance In-Context Learning of Transformers](../../NeurIPS2025/llm_reasoning/unlabeled_data_can_provably_enhance_in-context_learning_of_transformers.md)
- [\[AAAI 2026\] LLMs for Game Theory: Entropy-Guided In-Context Learning and Adaptive CoT Reasoning](../../AAAI2026/llm_reasoning/llms_for_game_theory_entropy-guided_in-context_learning_and_adaptive_cot_reasoni.md)
- [\[ICLR 2026\] Adaptive Social Learning via Mode Policy Optimization for Language Agents](adaptive_social_learning_via_mode_policy_optimization_for_language_agents.md)
- [\[ICLR 2026\] Temperature as a Meta-Policy: Adaptive Temperature in LLM Reinforcement Learning](temperature_as_a_meta-policy_adaptive_temperature_in_llm_reinforcement_learning.md)
- [\[ICLR 2026\] FastGRPO: Accelerating Policy Optimization via Concurrency-aware Speculative Decoding and Online Draft Learning](fastgrpo_accelerating_policy_optimization_via_concurrency-aware_speculative_deco.md)

</div>

<!-- RELATED:END -->
