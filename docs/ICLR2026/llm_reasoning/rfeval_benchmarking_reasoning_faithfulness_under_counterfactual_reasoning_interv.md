---
title: >-
  [Paper Note] RFEval: Benchmarking Reasoning Faithfulness under Counterfactual Reasoning Intervention in Large Reasoning Models
description: >-
  [ICLR 2026][LLM Reasoning][Reasoning Faithfulness] This paper proposes a formal definition of Reasoning Faithfulness (RF) decomposed into stance consistency and causal influence…
tags:
  - "ICLR 2026"
  - "LLM Reasoning"
  - "Reasoning Faithfulness"
  - "Counterfactual Intervention"
  - "Large Reasoning Models"
  - "Stance Consistency"
  - "Causal Influence"
date: 2026-05-08
content_hash: 9304f4ce7615a491
---

# RFEval: Benchmarking Reasoning Faithfulness under Counterfactual Reasoning Intervention in Large Reasoning Models

**Conference**: ICLR 2026
**arXiv**: [2602.17053](https://arxiv.org/abs/2602.17053)  
**Code**: [https://github.com/AIDASLab/RFEval](https://github.com/AIDASLab/RFEval)  
**Area**: LLM Reasoning
**Keywords**: Reasoning Faithfulness, Counterfactual Intervention, Large Reasoning Models, Stance Consistency, Causal Influence

## TL;DR
This paper proposes a formal definition of Reasoning Faithfulness (RF) decomposed into stance consistency and causal influence, constructs the RFEval benchmark comprising 7,186 instances across 7 tasks, and evaluates 12 open-source Large Reasoning Models (LRMs) via output-level counterfactual reasoning intervention. Key findings include: 49.7% of outputs are unfaithful, RL post-training degrades faithfulness, and task accuracy is not a reliable proxy for faithfulness.

## Background & Motivation
LRMs demonstrate strong performance on complex problems, yet their generated reasoning chains frequently appear plausible while being unfaithful — the stated reasoning process does not reflect the model's actual decision mechanism. In high-stakes domains such as medicine, law, and human resources, such unfaithful reasoning risks misleading users through persuasive but spurious explanations, fostering excessive reliance.

Existing evaluations of LRMs predominantly focus on task accuracy, but high accuracy does not imply faithful reasoning: a model may produce correct answers via post-hoc rationalization without genuinely following its stated reasoning. Prior faithfulness research has largely relied on input-level perturbations (e.g., injecting prompt biases), lacking a systematic output-level intervention framework.

**Key Challenge**: The internal "true reasoning process" (i.e., all intermediate activations) is unobservable, necessitating a purely behavioral, model-agnostic proxy measure for faithfulness. **Key Insight**: The paper adopts **output-level counterfactual intervention** — injecting counterfactual reasoning containing errors into the model's reasoning trajectory and observing whether the model responds consistently (i.e., revises its stance) or merely makes superficial adjustments while preserving its original conclusion.

## Method

### Overall Architecture
RFEval evaluation proceeds in two steps: (1) **Baseline**: the LRM receives a question and produces a complete output $o = (r, e, a)$; (2) **Intervention**: a counterfactual reasoning chain $r'$ is prepended to the assistant's response, and the model produces an intervened output $o' = (r_\text{new}, e', a')$. Faithfulness $\text{RF}(o, o')$ is then assessed via two conditions: stance consistency $\chi$ and causal influence $\kappa$.

### Key Designs

1. **Stance Consistency ($\chi$)**

    - **Function**: Verifies whether the reasoning chain, explanation, and final answer form a coherent and consistent argumentative chain within the model's output.
    - **Design Motivation**: Even when the answer is correct, if the reasoning is internally contradictory or inconsistent with the answer, the reasoning is decorative rather than genuine.
    - **Mechanism**: A stance continuity indicator $\iota(u, v)$ is defined, equaling 1 if two consecutive text segments share the same stance or if the latter explicitly justifies any deviation. The output is flattened into a sequence $(c_1, \ldots, c_m)$, and global consistency is computed as $\chi(o) = \bigwedge_i \iota(\langle c_{1:i-1}\rangle, c_i)$. An o3-based stance extractor achieves micro-F1 = 0.952 on 1,035 annotated examples.
    - **Novelty**: Examines not only reasoning-to-answer consistency but also step-level continuity within the reasoning chain itself.

2. **Causal Influence ($\kappa$)**

    - **Function**: Verifies whether the model's stated reasoning causally determines the final answer.
    - **Design Motivation**: Stance consistency alone ensures internal logical coherence but cannot distinguish reasoning that genuinely drives the answer from post-hoc rationalization.
    - **Mechanism**: A counterfactual reasoning chain $r'$ opposing the model's original stance is injected. If the post-intervention reasoning stance or final answer changes, $\kappa(o, o') = 1$. A key constraint: $\kappa$ is evaluated only when the **contrastive premise** holds (i.e., $S(r) \neq S(r')$), avoiding ambiguous cases.
    - **Novelty**: Employs output-level rather than input-level intervention, directly probing the causal efficacy of the reasoning trajectory.

3. **RFEval Benchmark Construction**

    - **Function**: Contains 7,186 instances spanning 7 tasks — code generation, mathematical reasoning, logical reasoning, tabular reasoning, context understanding, legal decision-making, and paper review.
    - **Design Motivation**: Heterogeneous multi-step reasoning tasks are required to assess faithfulness differences across reasoning types.
    - **Mechanism**: Counterfactual reasoning chains are generated by o3 (containing subtle but plausible reasoning flaws), followed by automated verification via gpt-5 and manual review by 8 graduate students, filtering 8,499 candidates down to 7,186 instances (PABAK = 0.710).

### Loss & Training
This work is an evaluation benchmark and involves no model training. The core evaluation metric is faithfulness under the contrastive condition:

$$\text{RF}^\text{contrast}(M, D) = \mathbb{E}[\text{RF}(o, o') \mid \delta(x, r'; M) = 1]$$

where $\delta = 1$ indicates that the contrastive premise holds. Contrastive coverage $c(M)$ is also reported to reflect the proportion of instances satisfying the contrastive premise.

## Key Experimental Results

### Main Results (Faithfulness Evaluation of 12 LRMs)

| Model | Code Gen. | Math | Logic | Table | Context | Legal | Paper Review | Overall RF |
|-------|-----------|------|-------|-------|---------|-------|--------------|------------|
| Qwen3-8B | 21.15 | 37.97 | 72.74 | 58.11 | 43.97 | 48.64 | — | 41.95 |
| Qwen3-32B | 24.66 | 47.87 | 88.62 | 89.84 | 77.66 | 89.90 | 91.49 | 73.29 |
| R1-Qwen-7B | 38.25 | 29.54 | 82.13 | 44.46 | 76.31 | 70.63 | 81.49 | 61.37 |
| R1-Llama-8B | 26.48 | 33.03 | 55.78 | 57.68 | 64.63 | 78.97 | 94.53 | 58.46 |
| gpt-oss-20b | 26.44 | 24.90 | 13.55 | 22.62 | 33.93 | 59.14 | 47.41 | 32.11 |
| gpt-oss-120b | 22.01 | 16.07 | 8.62 | 34.21 | 13.67 | 39.58 | 70.71 | 27.50 |

Overall, 49.73% of outputs are unfaithful. Qwen3-32B achieves the highest RF (73.29%), while gpt-oss-120b performs worst (27.50%).

### Ablation Study (Effect of Post-Training Paradigm on Faithfulness)

| Variant | MiMo-7B RF / $c(M)$ | Olmo-3-7B RF / $c(M)$ |
|---------|---------------------|----------------------|
| Base | 59.33 / 0.69 | 65.87 / 0.42 |
| SFT-only | 60.05 / 0.74 | 61.38 / 0.70 |
| RL-only | 58.74 / 0.54 | — |
| **SFT+RL** | **46.32 / 0.72** | **50.93 / 0.73** |

Across both model families, SFT largely preserves RF, but **adding RLVR on top of SFT consistently degrades RF** (MiMo: 60.05→46.32; Olmo: 61.38→50.93).

### Key Findings
- **Primary source of unfaithfulness**: Stance inconsistency ($\chi$ failure) is the dominant cause. Post-intervention inconsistency ($\neg\chi(o')$) is the most prominent failure mode; baseline inconsistency ($\neg\chi(o)$) is less frequent; causal failure ($\neg\kappa$) is secondary.
- **Significant task-level variation**: Convergent tasks (code: 24.18%, math: 28.06%) exhibit the lowest faithfulness, while argumentative tasks (legal: 70.17%, logical: 58.28%) are substantially higher — convergent tasks require correction of local errors, producing "silent corrections."
- **Scale does not imply faithfulness**: Within the gpt-oss family, RF decreases from 20B to 120B (32.11→27.50), while Qwen scales positively from 8B to 32B (41.95→73.29), indicating that model scale is not a determining factor.
- **Accuracy does not imply faithfulness**: After controlling for model and task effects, the residual accuracy–faithfulness association is statistically non-significant (Weighted Pearson $r = 0.090$, $p \approx 0.445$).
- **RLVR rewards do not differentiate by stance consistency**: Outputs with $\chi=1$ and $\chi=0$ receive nearly identical mean rewards (0.628 vs. 0.671), suggesting that current RL objectives may incentivize models to produce accurate but unfaithful reasoning shells.

## Highlights & Insights
- Decomposing reasoning faithfulness into two testable conditions — stance consistency and causal influence — represents the most rigorous behavioral-level formalization to date.
- The output-level counterfactual intervention design is particularly elegant: injecting flaws directly into the reasoning trajectory probes the causal status of reasoning more directly than input-level perturbations.
- The finding that RL post-training degrades faithfulness constitutes an important warning: current RLVR rewards only final format and correctness, providing no incentive for stance consistency.
- The argument that accuracy is not a reliable proxy for faithfulness is supported by both theoretical reasoning and empirical evidence, with broad implications for LRM evaluation frameworks.
- The introduction of contrastive coverage $c(M)$ addresses selection bias inherent in counterfactual evaluation.

## Limitations & Future Work
- Evaluation of closed-source models is restricted by response integrity mechanisms (e.g., signature verification); only open-source models are currently assessed.
- Stance extraction relies on a powerful LLM (o3), which may itself introduce bias.
- The quality of counterfactual chains $r'$ depends on o3's generation capability and may lack sufficient subtlety in edge cases.
- The paper review task exhibits very low contrastive coverage (~0.35–0.45), limiting the reliability of conclusions on that task.
- The paper identifies and characterizes the problem but does not propose specific training methods for improving faithfulness.

## Related Work & Insights
- Compared to input-level intervention approaches such as Turpin et al. (2023), RFEval operates at the output level and more directly tests the causal efficacy of reasoning.
- Compared to the mid-reasoning modification approach of Lanham et al. (2023), RFEval provides a formal faithfulness definition rather than ad-hoc tests.
- The finding that RL degrades faithfulness suggests that future RL training should incorporate stance consistency into the reward function.
- The framework naturally extends to agentic settings, where faithfulness becomes even more critical as reasoning directly drives planning and tool invocation.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First formal definition and systematic evaluation framework for reasoning faithfulness in LRMs.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 12 models × 7 tasks × 7,186 instances, including within-family ablations and statistical hypothesis testing.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Rigorous formal definitions, well-structured empirical analysis, and high-quality figures and tables.
- **Value**: ⭐⭐⭐⭐⭐ Reveals a core dimension overlooked in LRM evaluation, providing important guidance for safe and trustworthy AI research.

## Background & Motivation
- LRMs produce apparently plausible but unfaithful reasoning — accuracy cannot serve as a proxy for faithfulness.

## Method
- Stance consistency: internal coherence of the reasoning chain.
- Causal influence: reasoning causally determines the answer (verified via counterfactual intervention).
- RFEval: output-level intervention benchmark.

## Key Experimental Results

| Finding | Value |
|---------|-------|
| Unfaithfulness rate | **49.7%** |
| Concentrated in | Math / Code (brittle domains) |
| RL post-training | Degrades faithfulness (despite unchanged accuracy) |

## Key Findings
- Post-training paradigm (SFT vs. RL) affects faithfulness more than model scale.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Formalization of reasoning faithfulness.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 7 tasks / 7,186 instances.
- **Value**: ⭐⭐⭐⭐⭐ Exposes hidden risks of RL training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Towards Safe Reasoning in Large Reasoning Models via Corrective Intervention](towards_safe_reasoning_in_large_reasoning_models_via_corrective_intervention.md)
- [\[ICLR 2026\] Training Large Reasoning Models Efficiently via Progressive Thought Encoding](training_large_reasoning_models_efficiently_via_progressive_thought_encoding.md)
- [\[ICLR 2026\] Dynamics-Predictive Sampling for Active RL Finetuning of Large Reasoning Models](dynamics-predictive_sampling_for_active_rl_finetuning_of_large_reasoning_models.md)
- [\[ICLR 2026\] VisioMath: Benchmarking Figure-based Mathematical Reasoning in LMMs](visiomath_benchmarking_figure-based_mathematical_reasoning_in_lmms.md)
- [\[ICLR 2026\] GeoGramBench: Benchmarking the Geometric Program Reasoning in Modern LLMs](geogrambench_benchmarking_the_geometric_program_reasoning_in_modern_llms.md)

</div>

<!-- RELATED:END -->
