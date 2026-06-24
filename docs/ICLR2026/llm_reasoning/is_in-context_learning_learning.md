---
title: >-
  [Paper Note] Is In-Context Learning Learning?
description: >-
  [ICLR2026][Reasoning][in-context learning] Through large-scale controlled experiments, this paper systematically analyzes whether ICL constitutes "learning". It finds that while ICL satisfies the mathematical definition of learning, empirical evidence shows limited generalization—models primarily rely on structural patterns in prompts for deduction rather than truly acquiring new capabilities from examples.
tags:
  - "ICLR2026"
  - "Reasoning"
  - "in-context learning"
  - "ICL"
  - "memorisation"
  - "distributional shift"
  - "generalization"
  - "autoregressive models"
date: 2026-05-08
content_hash: 4665d5777f5511b1
---

# Is In-Context Learning Learning?

**Conference**: ICLR2026  
**arXiv**: [2509.10414](https://arxiv.org/abs/2509.10414)  
**Code**: Not open sourced  
**Area**: LLM Reasoning  
**Keywords**: in-context learning, ICL, memorisation, distributional shift, generalization, autoregressive models  

## TL;DR

Through large-scale controlled experiments, this paper systematically analyzes whether ICL constitutes "learning". It finds that while ICL satisfies the mathematical definition of learning, empirical evidence shows limited generalization—models primarily rely on structural patterns in prompts for deduction rather than truly acquiring new capabilities from examples.

## Background & Motivation

**Background**: In-context learning (ICL) enables autoregressive language models to solve downstream tasks via next-token prediction without parameter updates, requiring only a few exemplars in the prompt. This capability has sparked extensive debate over whether LLMs truly "learn" unseen tasks from a few examples.

**Limitations of Prior Work**:

- Research on ICL often blurs the boundary between "deduction" and "learning"; deduction does not equate to learning.
- ICL does not explicitly encode given observational data but depends on the model's prior knowledge and the exemplars in the prompt.
- Existing studies lack systematic control over confounding factors such as memorisation effects, pre-training data leakage, and distributional shifts.
- It is difficult to determine whether the superior performance of ICL stems from "truly learning from examples" or "prior knowledge retrieval + pattern matching."

**Key Challenge**: ICL is formally similar to learning (inferring task rules from exemplars in the prompt), but does its autoregressive encoding mechanism provide sufficient inductive bias to support robust generalization and genuine knowledge acquisition? The answer directly affects how LLM capabilities are perceived and utilized.

**Goal**: To answer "is ICL learning" from both theoretical and empirical perspectives—first by arguing mathematically that ICL meets the formal definition of learning, and then by evaluating the actual boundaries of ICL learning through large-scale ablation studies to systematically eliminate confounding factors like memorisation, pre-training leakage, distributional shifts, and prompt styles.

## Method

### Overall Architecture

This paper answers "is ICL learning" through a dual-track approach of theory and empiricism. It first demonstrates mathematically that ICL satisfies the criteria of "learning" at the formal definition level (similar to the PAC learning framework). Then, it uses a large-scale controlled variable experiment covering various model architectures, prompt styles, task types, and data distributions to test the limits of this "formal learning." The mechanism is to decompose ICL performance into several confounding sources—pre-training memorisation, exemplar distribution, and prompt format—eliminating or perturbing them one by one to see how much of the "genuine learning from examples" remains.

### Key Designs

**1. Separating "Prior Retrieval" from "Learning from Examples"**

The high accuracy of ICL stems from two mixed sources: first, encountering similar tasks during pre-training (memorisation/leakage), and second, truly learning input-output mappings from prompt exemplars. Failing to separate these leads to an overestimation of ICL's learning ability. This paper isolates them using three approaches: first, quantifying pre-training data coverage of test tasks using various contamination detections and re-testing with contamination-free benchmarks; second, using zero-shot performance to measure pure prior levels, treating only the few-shot Gain over zero-shot as possible "learning," defined as the learning gain:

$$\Delta_{\text{learn}} = \mathbb{E}[\mathcal{L}(\hat{f}_{\text{zero-shot}})] - \mathbb{E}[\mathcal{L}(\hat{f}_{\text{few-shot}})]$$

where $\mathcal{L}$ is the task loss; finally, using random labels and counterfactual (label reversal) prompts to detect whether the model truly utilizes the mappings in the examples. Results show that $\Delta_{\text{learn}}$ significantly shrinks once memorisation is controlled, and random/counterfactual labels cause only minor decreases, indicating much of the ICL performance comes from prior knowledge retrieval.

**2. Observing Continuous Improvement with Increasing Exemplars**

Genuine learning should show continuous improvement with more and better data. Thus, the paper systematically scans the number of exemplars $k$ (from $k=0$ zero-shot to many-shot), exemplar distribution (class balance, sample difficulty, selection strategy, and order), prompt styles (standard few-shot, CoT, various templates), and models of different scales and architectures. The key finding is that as $k$ increases, accuracy tends toward a limit almost independent of the configuration:

$$\lim_{k \to \infty} \text{Acc}(k; \mathcal{D}, \mathcal{M}, \mathcal{S}) \approx C_{\text{task}}$$

where $\mathcal{D}$ is the exemplar distribution, $\mathcal{M}$ is the model, and $\mathcal{S}$ is the prompt style. Accuracy saturates rapidly (Gain is negligible after $k>16$) rather than rising continuously with data—this saturation curve is the core evidence for the negative conclusion regarding "genuine learning."

**3. CoT Gain comes from Structural Patterns rather than Deeper Learning**

While Chain-of-Thought (CoT) significantly boosts accuracy on certain tasks—often interpreted as "improved reasoning"—this paper finds it highly sensitive to the distributional features and format of the prompt. Standard few-shot performance is relatively stable but has a lower ceiling, whereas CoT shows higher variance and depends heavily on the format and structure of the reasoning chain. In tasks with similar forms but different semantics, accuracy varies significantly. This high variance suggests CoT gains do not come from deeper task learning but from utilizing structural regularities in the reasoning chain for more efficient pattern deduction—ICL essentially extracts patterns from the statistical regularities of the prompt rather than encoding new knowledge.

## Key Experimental Results

### Main Results: Systematic Evaluation of ICL Learning Ability

| Control Variable | Core Conclusion | Key Evidence |
|:---------|:---------|:---------|
| Memorisation | Pre-training memory significantly contributes to ICL | Performance drops significantly on contamination-free benchmarks |
| Exemplar Count | Accuracy saturates rapidly | Gain is negligible for $k > 16$; logarithmic saturation curve |
| Exemplar Distribution | Insensitive to distribution in the limit | Convergence values are similar across different class/difficulty distributions |
| Model Selection | Differences across models narrow in the limit | Comparison across various architectures and scales |
| Prompt Style | Sensitive for CoT, insensitive for standard few-shot | Format variation experiments show higher fluctuations for CoT |
| Linguistic Features | Surface features have limited impact on limit performance | Paraphrasing/formatting changes have weak effects on accuracy |

### Ablation Study: Analysis of ICL Information Utilization Mechanisms

| Experimental Condition | Accuracy Performance | Core Meaning |
|:---------|:----------|:---------|
| Correct Labels | Baseline (Highest) | Standard ICL performance |
| Random Labels | Minor decrease | Model does not fully rely on exemplar mappings |
| Counterfactual Labels | Moderate decrease | Model partially uses label info but not as the core basis |
| No Labels (Input only) | Near few-shot | Structural features of the prompt are more important than label content |
| Shuffled Exemplars | Nearly unchanged | Model is insensitive to the order of exemplars |

## Rating

**Rating**: ⭐⭐⭐⭐

**Highlights & Insights**:

- Addresses a fundamental question regarding the nature of ICL with both theoretical depth and empirical breadth.
- Highly rigorous experimental design: systematically eliminates confounding factors like memory, distributional shift, and style.
- The core finding—that accuracy saturates as exemplars increase and is insensitive to hyperparameters—holds significant theoretical and practical value.
- The analysis of CoT's distributional sensitivity provides a new perspective on CoT mechanisms.

**Limitations & Future Work**:

- Primarily yields negative conclusions (limited ICL learning ability) without providing many constructive directions or solutions.
- Some experiments might be limited by selection bias in specific benchmark task sets; generalization to complex reasoning and generation tasks needs verification.
- The definition of "learning" itself has multiple interpretations (PAC vs. Bayesian vs. generalization); the robustness of conclusions under different definitions is not fully discussed.
- Lacks a systematic scaling law analysis of ICL behavior differences across different model sizes.

**Related Work & Insights**:

- Unlike works interpreting ICL from a Bayesian inference perspective (Xie et al., 2021), this paper systematically questions the hypothesis of ICL as a robust learning mechanism from an empirical learning theory standpoint.
- Unlike single-factor analyses, this paper simultaneously controls memory, distribution, model, and style, reaching more comprehensive and reliable negative conclusions.
- Unlike purely theoretical analyses, this paper combines mathematical proof with large-scale experiments, enhancing the persuasiveness of its conclusions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] PERK: Long-Context Reasoning as Parameter-Efficient Test-Time Learning](perk_long-context_reasoning_as_parameter-efficient_test-time_learning.md)
- [\[NeurIPS 2025\] Unlabeled Data Can Provably Enhance In-Context Learning of Transformers](../../NeurIPS2025/llm_reasoning/unlabeled_data_can_provably_enhance_in-context_learning_of_transformers.md)
- [\[ICML 2026\] Many-Shot CoT-ICL: Making In-Context Learning Truly Learn](../../ICML2026/llm_reasoning/many-shot_cot-icl_making_in-context_learning_truly_learn.md)
- [\[ICLR 2026\] Learning to Reason over Continuous Tokens with Reinforcement Learning (HyRea)](learning_to_reason_over_continuous_tokens_with_reinforcement_learning.md)
- [\[AAAI 2026\] LLMs for Game Theory: Entropy-Guided In-Context Learning and Adaptive CoT Reasoning](../../AAAI2026/llm_reasoning/llms_for_game_theory_entropy-guided_in-context_learning_and_adaptive_cot_reasoni.md)

</div>

<!-- RELATED:END -->
