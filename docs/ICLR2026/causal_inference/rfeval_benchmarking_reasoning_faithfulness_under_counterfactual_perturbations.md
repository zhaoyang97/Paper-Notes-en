---
title: >-
  [Paper Note] RFEval: Benchmarking Reasoning Faithfulness under Counterfactual Perturbations
description: >-
  [ICLR 2026][Causal Inference][reasoning_faithfulness] This paper proposes a formal framework for reasoning faithfulness (stance consistency + causal influence) and the RFEval benchmark (7…
tags:
  - "ICLR 2026"
  - "Causal Inference"
  - "reasoning_faithfulness"
  - "LRM_evaluation"
  - "counterfactual_intervention"
  - "benchmark"
date: 2026-05-08
content_hash: 65f2cb19cf662f02
---

# RFEval: Benchmarking Reasoning Faithfulness under Counterfactual Perturbations

**Conference**: ICLR 2026
**arXiv**: [2602.17053](https://arxiv.org/abs/2602.17053)  
**Code**: [AIDASLab/RFEval](https://github.com/AIDASLab/RFEval)  
**Area**: Causal Reasoning
**Keywords**: reasoning_faithfulness, LRM_evaluation, counterfactual_intervention, benchmark

## TL;DR

This paper proposes a formal framework for reasoning faithfulness (stance consistency + causal influence) and the RFEval benchmark (7,186 instances × 7 tasks). By applying output-level counterfactual interventions to evaluate 12 open-source LRMs, it finds that 49.7% of outputs are unfaithful and that accuracy is not a reliable proxy for faithfulness.

## Background & Motivation

Large reasoning models (LRMs) such as DeepSeek-R1 and Qwen3 achieve strong performance on complex tasks, yet frequently produce **plausible-sounding but unfaithful** explanations—i.e., the stated reasoning does not reflect the model's actual decision process.

**Core Problem**:
- When a model claims "I chose A because of X," is X truly the cause of choosing A?
- In high-stakes domains such as medicine, law, and HR, unfaithful explanations can mislead users and obscure biases.
- Existing evaluations focus primarily on accuracy, but accuracy does not equate to faithfulness.

**Limitations of Prior Work**:
- Internal activation analysis (e.g., probing methods) requires model access and does not scale.
- No systematic behavioral-level framework for faithfulness evaluation exists.
- No unified benchmark enables cross-LRM comparison of reasoning faithfulness.

## Method

### Overall Architecture

This paper defines reasoning faithfulness at the behavioral level, assessed purely from the model's textual outputs without access to internal weights. The core idea: if reasoning is faithful, then changing the reasoning should change the answer.

### Key Designs: Two Testable Conditions

**Condition 1: Stance Consistency**

For an LRM output $o = (r, e, a)$ (reasoning, explanation, answer), stance consistency requires the entire output sequence to form a coherent argumentative chain:

$$\chi(o) := \bigwedge_{i=1}^{m} \iota(\langle c_{1:i-1}\rangle, c_i) \in \{0, 1\}$$

where $\iota(u, v)$ is a stance continuity indicator: $\iota = 1$ if and only if the stance of $v$ is consistent with $u$, or $v$ explicitly identifies and justifies a departure.

**Condition 2: Causal Influence**

Given the model's original output $o$ and the post-intervention output $o'$ after injecting counterfactual reasoning $r'$, causal influence requires that the reasoning or answer changes:

$$\kappa(o, o') := \mathbb{1}[S(r_{\text{new}}) \neq S(r)] \lor \mathbb{1}[S(a') \neq S(a)]$$

**Unified Definition: Reasoning Faithfulness**

$$\text{RF}(o, o') := \mathbb{1}[\chi(o) = 1 \land \chi(o') = 1 \land \kappa(o, o') = 1]$$

That is, both the original and intervened outputs must be stance-consistent, and the intervention must produce a genuine causal effect.

### Contrastive Pre-condition

To ensure causal identifiability, evaluation is restricted to contrastive pairs ($\delta = 1$), where the injected counterfactual reasoning $r'$ takes a stance opposite to the model's original stance. This eliminates ambiguity from "no-change" outcomes.

### Benchmark Construction Pipeline

**Counterfactual Reasoning Generation**: Generated using OpenAI o3, with each prompt including 3 manually crafted few-shot examples to elicit subtle yet plausible reasoning flaws (e.g., arithmetic errors, logical fallacies).

**Two-Stage Validation**:
1. GPT-5 automatic filtering: checks for misleading sufficiency, logical coherence, subtle plausibility, and uniqueness (MCQA).
2. Manual review by 8 NLP/ML graduate students: PABAK = 0.710, reducing 8,499 candidates to 7,186 instances.

**Evaluation Implementation**: o3 serves as the evaluator for stance extraction, with human-validated F1 = 0.952.

### Loss & Training

This paper presents an evaluation framework and involves no training loss. The core metric is:

$$\text{RF}^{\text{contrast}}(\mathcal{M}, \mathcal{D}) = \mathbb{E}\left[\text{RF}(o, o') \mid \delta(x, r'; \mathcal{M}) = 1\right]$$

along with contrastive coverage $c(\mathcal{M}) = \Pr(\delta = 1)$.

## Key Experimental Results

### Main Results: Reasoning Faithfulness of 12 LRMs

| Model | Overall RF (%) | Coverage | CG | MR | LR | TR | CU | LD | PR |
|-------|---------------|----------|-----|-----|-----|-----|-----|-----|-----|
| Qwen3-32B | **73.29** | 0.78 | 24.66 | 47.87 | 88.62 | 89.84 | 77.66 | 89.90 | 91.49 |
| LN-Super_v1 | 68.52 | 0.58 | 26.48 | 44.90 | 77.13 | 69.38 | 81.70 | 80.38 | 98.47 |
| R1-Qwen-32B | 64.24 | 0.75 | 29.02 | 32.57 | 70.79 | 82.47 | 63.16 | 91.04 | 75.13 |
| R1-Qwen-7B | 61.37 | 0.70 | 38.25 | 29.54 | 82.13 | 44.46 | 76.31 | 70.63 | 81.49 |
| MiMo-RL-Zero | 58.74 | 0.54 | 20.83 | 33.50 | 70.59 | 61.32 | 69.58 | 77.87 | 66.83 |
| R1-Llama-70B | 56.47 | 0.78 | 27.89 | 31.28 | 74.03 | 73.78 | 51.40 | 80.53 | 51.84 |
| gpt-oss-20b | 32.11 | 0.82 | 26.44 | 24.90 | 13.55 | 22.62 | 33.93 | 59.14 | 47.41 |
| gpt-oss-120b | 27.50 | 0.82 | 22.01 | 16.07 | 8.62 | 34.21 | 13.67 | 39.58 | 70.71 |

Key finding: **49.7% of evaluated instances are unfaithful**. Even the best-performing model, Qwen3-32B, achieves only 73.29%.

### Ablation Study: Sources of Unfaithfulness

| Violation Type | Proportion | Description |
|----------------|-----------|-------------|
| $\neg\chi(o')$ (post-intervention stance inconsistency) | Dominant | Model fails to respond coherently to the counterfactual premise |
| $\neg\kappa$ (no causal influence) | Secondary | Reasoning changes but the answer does not follow |
| $\neg\chi(o)$ (baseline stance inconsistency) | Minor | Original output is internally contradictory |

**Causal Influence Types**:
- Most models exhibit "Both" (reasoning and answer both change).
- The gpt-oss series and Magistral-Small show more "Reasoning-only" cases (reasoning changes but answer does not).
- Some Qwen/R1 models exhibit "Answer-only" cases (silent corrections—answer changes without the reasoning reflecting it).

### Key Findings

1. **Task structure determines faithfulness**: Math and code tasks (convergent, unique answers) show the highest unfaithfulness rates; legal and paper review tasks (supportive of multi-perspective argumentation) show the highest faithfulness.
2. **Scale does not determine faithfulness**: gpt-oss faithfulness decreases from 20B to 120B (32.11% → 27.50%), while Qwen3 improves substantially from 8B to 32B (41.95% → 73.29%).
3. **Post-training paradigm is the key factor**: Within the same model family, RLVR-style post-training may **reduce** faithfulness even when accuracy remains unchanged.
4. **Accuracy ≠ Faithfulness**: After controlling for model and task, the correlation between accuracy and faithfulness is weak and statistically insignificant. High accuracy does not guarantee faithful reasoning.
5. **Failure locations exhibit family-specific patterns**: The gpt-oss series breaks early in the intervention chain ($r' \to r_{\text{new}}$); Qwen/R1 models fail more often at the later stage ($r_{\text{new}} \to a'$).

## Highlights & Insights

- **Elegant formal framework**: Reasoning faithfulness is decomposed into two independently testable conditions (consistency + causality), making the framework both rigorous and operationalizable.
- **Clever counterfactual intervention design**: By injecting opposing reasoning at the output level, the method avoids the need for internal model access.
- **Most important finding**: RL post-training can reduce faithfulness without degrading accuracy—a cautionary signal for the current RLVR trend.
- **Practical value**: An open-source benchmark of 7,186 instances and an evaluation framework are provided, directly applicable to LRM auditing.

## Limitations & Future Work

1. Only open-source models are evaluated; closed-source API models (e.g., GPT-5.2, Claude) are difficult to subject to standard interventions due to response integrity mechanisms.
2. The framework relies on an LLM evaluator (o3) for stance extraction; although F1 reaches 0.952, this is not perfect.
3. Counterfactual reasoning is generated by o3, which may not cover all types of reasoning flaws.
4. Evaluation operates at the coarse-grained $(r, e, a)$ level, without fine-grained step-by-step analysis of the reasoning chain.
5. Contrastive coverage—particularly in the Paper Review task, averaging only 0.35–0.45—means that a substantial proportion of instances are excluded due to stance alignment.

## Related Work & Insights

- **Jacovi & Goldberg (2020)**: Early conceptual framework defining faithful explanations.
- **Chen et al. (2025b); Arcuschin et al. (2025)**: Empirical evidence of unfaithful reasoning in LRMs.
- **Lanham et al. (2023)**: CoT faithfulness study, but not based on counterfactual intervention.
- This paper's contributions lie in: (1) a formal definition, (2) large-scale systematic evaluation, and (3) characterization of the relationship between training paradigms and faithfulness.
- Implication for LRM deployment: reporting accuracy alone is insufficient; faithfulness should be reported concurrently.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The formal framework and counterfactual intervention methodology are novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 12 models × 7 tasks × 7,186 instances; large-scale and systematic.
- **Value**: ⭐⭐⭐⭐ — The open-source benchmark is directly applicable to LRM auditing.
- **Writing Quality**: ⭐⭐⭐⭐ — Formal definitions are clear, though the density of notation requires careful reading.
- **Overall**: ⭐⭐⭐⭐ (4/5)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] On the Eligibility of LLMs for Counterfactual Reasoning: A Decompositional Study](on_the_eligibility_of_llms_for_counterfactual_reasoning_a_decompositional_study.md)
- [\[AAAI 2026\] MUG: Multi-agent Undercover Gaming — Hallucination Removal via Counterfactual Test for Multimodal Reasoning](../../AAAI2026/causal_inference/multi-agent_undercover_gaming_hallucination_removal_via_coun.md)
- [\[ICLR 2026\] Journey to the Centre of Cluster: Harnessing Interior Nodes for A/B Testing under Network Interference](journey_to_the_centre_of_cluster_harnessing_interior_nodes_for_ab_testing_under_.md)
- [\[ICLR 2026\] Counterfactual Explanations on Robust Perceptual Geodesics](counterfactual_explanations_on_robust_perceptual_geodesics.md)
- [\[NeurIPS 2025\] Counterfactual Reasoning for Steerable Pluralistic Value Alignment of Large Language Models](../../NeurIPS2025/causal_inference/counterfactual_reasoning_for_steerable_pluralistic_value_alignment_of_large_lang.md)

</div>

<!-- RELATED:END -->
