---
title: >-
  [Paper Note] DRO-InstructZero: Distributionally Robust Prompt Optimization for Large Language Models
description: >-
  [ICLR 2026][Code Intelligence][prompt optimization] This work integrates distributionally robust optimization (DRO) into the Bayesian optimization (BO) framework of InstructZero. By maximizing the worst-case expected uti…
tags:
  - "ICLR 2026"
  - "Code Intelligence"
  - "prompt optimization"
  - "distributionally robust optimization"
  - "Bayesian optimization"
  - "instruction tuning"
  - "zero-shot learning"
date: 2026-05-08
content_hash: 51c3634a99bb33cb
---

# DRO-InstructZero: Distributionally Robust Prompt Optimization for Large Language Models

**Conference**: ICLR 2026
**arXiv**: [2510.15260](https://arxiv.org/abs/2510.15260)  
**Code**: None  
**Area**: Code Intelligence
**Keywords**: prompt optimization, distributionally robust optimization, Bayesian optimization, instruction tuning, zero-shot learning

## TL;DR

This work integrates distributionally robust optimization (DRO) into the Bayesian optimization (BO) framework of InstructZero. By maximizing the worst-case expected utility over an ambiguity set defined by an f-divergence ball, the automatically searched prompts maintain reliable performance under distribution shift.

## Background & Motivation

Large language models are highly sensitive to prompt phrasing — even minor paraphrasing can cause a sharp drop in accuracy. Automatic instruction search methods such as InstructZero leverage Bayesian optimization to search for optimal soft prompts in a continuous latent space, achieving promising results. However, these methods optimize **expected scores over a single validation distribution**, a fundamental assumption that breaks down in real-world deployment:

**Inevitable distribution shift**: The distribution of user inputs may differ significantly from the validation distribution, e.g., domain switching, adversarial examples, and varying query styles.

**Overfitting to the training distribution**: Instructions optimized on a fixed distribution tend to be brittle and may fail under different evaluation scenarios.

**Insufficient transferability**: Classical BO acquisition functions (EI, UCB) focus solely on average performance, ignoring tail risks.

The authors' core insight is that existing methods pursue "average optimality," whereas practical deployment requires "worst-case reliability" — precisely the canonical setting for DRO. Combining DRO with BO enables explicit optimization of robustness while preserving query efficiency.

## Method

### Review of InstructZero

InstructZero formulates prompt optimization as a BO problem in continuous space, with a four-step pipeline:

1. A low-dimensional soft prompt $p \in \mathbb{R}^d$ is projected into the embedding space of an open-source LLM via a random matrix $A$.
2. The open-source LLM $g(\cdot)$ (Vicuna) converts the projected vector and task exemplars into a natural language instruction $v$.
3. A black-box LLM $f(\cdot)$ (ChatGPT) executes the instruction, and an evaluation metric $h(\cdot,\cdot)$ assigns a score.
4. The GP posterior is updated and an acquisition function selects the next prompt.

The standard objective is $\max_v \mathbb{E}_{(X,Y)\sim D^t}[h(f([v;X]),Y)]$.

### DRO Extension: Robust Objective

DRO-InstructZero reformulates the optimization objective from expectation maximization to a minimax form:

$$\max_{v \in V} \inf_{Q \in \mathcal{U}(D^t)} \mathbb{E}_{(X,Y)\sim Q}[h(f([v;X]),Y)]$$

where the ambiguity set $\mathcal{U}(D^t)$ is defined as an f-divergence (KL divergence) ball of radius $\epsilon$ centered at a reference distribution $w_{\text{ref}}$. The inner $\inf$ finds the worst-case distribution, while the outer $\max$ requires the instruction to perform well under that distribution.

After applying the same soft prompt parameterization as InstructZero, the robust objective becomes a low-dimensional black-box function:

$$H(p) \triangleq \inf_{Q \in \mathcal{U}(D^t)} \mathbb{E}_{(X,Y)\sim Q}[h(f([g([Ap;\text{exemplars}]);X]),Y)]$$

### Robust Acquisition Rule

For each candidate prompt $p_m$, an optimistic UCB score vector across tasks is first computed as $\text{ucb}_m = [\mu^t(p_m) + \beta(m)\sigma^t(p_m)]_t$. An adversarial distribution is then solved within the ambiguity set:

$$w_m^* = \arg\min_{w': \|w' - w_{\text{ref}}\|_\mathcal{M} \leq \epsilon(m)} \langle \text{ucb}_m, w' \rangle$$

The next prompt is selected by maximizing the robust acquisition value: $p_{m+1} = \arg\max_p \langle \text{ucb}_m, w_m^* \rangle$. This ensures the search explicitly favors instructions that remain effective under the worst-case distribution.

### Instruction-Coupled Kernel

Building on the instruction-coupled kernel from InstructZero, the method combines prompt-space similarity $l(\cdot,\cdot)$ with instruction semantic similarity $s(\cdot,\cdot)$, and weights the kernel matrix by the adversarial distribution $w^*$, enabling the GP to jointly capture semantic proximity and distributional robustness.

### Implementation Details

- **Optimizer**: CMA-ES evolutionary strategy, exploring 25 candidate soft prompts per round.
- **Multi-task joint optimization**: 2 tasks are randomly sampled each round for joint DRO.
- **Reference distribution**: Initialized as uniform and dynamically updated via EMA with inverse-probability weighting by evaluation scores.
- **Adversarial weight solver**: cvxpy convex optimization solver with Wasserstein ball constraints.
- **Hyperparameters**: Ambiguity radius $\epsilon = 0.1$, exploration coefficient $\beta(t) = 2.0\sqrt{2.0\log(t+1)}$, soft prompt dimension $d = 10$.
- **Hardware**: Single NVIDIA A100 GPU.

## Key Experimental Results

### Main Results: 32 BIG-Bench Tasks

Experiments use Vicuna (open-source LLM) + ChatGPT (black-box LLM), following the instruction-induction protocol with the same query budget as InstructZero:

| Metric | InstructZero | DRO-InstructZero | Gain |
|--------|-------------|-------------------|------|
| Average accuracy | 0.719 | 0.756 | +3.6 pts |
| Median per-task gain | — | — | +5.5 pts |
| Win / Tie / Loss | — | 18 / 8 / 6 | — |
| Translation (EN→DE/ES/FR) | 0.867 | 0.980 | +11.3 pts |
| Auto-Debugging | 0.50 | 0.60 | +10 pts |
| Formality Rewriting | 0.63 | 0.68 | +5 pts |
| Saturated tasks (Sum, etc.) | 100% | 100% | Tied |

Improvements are most pronounced on distribution-shift-sensitive tasks: Unscrambling 0.67→0.80, Second Letter 0.62→0.74, Taxonomy 0.82→0.92, Sentiment 0.93→0.99.

### Ablation Study

| Method | Distribution-Shift Accuracy | Notes |
|--------|----------------------------|-------|
| InstructZero-EI | 61.3 ± 0.7% | Original expected improvement acquisition |
| InstructZero-UCB | Slightly below EI | Standard UCB acquisition |
| DRO w/o BO | Moderate | DRO applied directly in original instruction space without BO |
| **DRO-InstructZero** | **85–90%** | Full method, +25–30 pts |

Two key findings: (1) DRO outperforms EI/UCB under distribution shift by 15–25 absolute percentage points; (2) removing BO leads to a significant performance drop, demonstrating that structured exploration in the latent space is critical for efficiency, and DRO achieves its full potential only in conjunction with BO.

### Regression Cases

Minor performance drops are observed on lexical/classification tasks such as Antonyms (−11 pts), Object Counting (−10), and CS-algorithm (−8). The authors attribute this to worst-case reweighting potentially diverging from the precise lexical rules required by the evaluator. A hybrid acquisition function that interpolates between robust and nominal scores during the exploitation phase is proposed as a mitigation strategy.

## Highlights & Insights

1. **Complementarity of DRO and BO**: BO handles efficient exploration of the continuous latent space, while DRO provides robustness guarantees — their combination preserves query efficiency while avoiding overfitting to the training distribution.
2. **Plug-and-play design**: Only the acquisition function is replaced, requiring no modification to LLM architectures or training pipelines; the method can be directly integrated into any BO-based prompt optimization framework.
3. **Theoretical predictions confirmed empirically**: DRBO theory predicts that "average-optimal policies are fragile under worst-case scenarios," and the distribution-shift degradation observed in InstructZero precisely validates this claim.
4. **Unchanged query budget**: Robustness gains do not require additional API calls; the only overhead is the convex optimization solver.

## Limitations & Future Work

1. **Computational overhead from adversarial reweighting**: Each round requires solving an additional convex optimization problem, increasing per-round runtime.
2. **Hyperparameter sensitivity**: The divergence measure type and ambiguity radius $\epsilon$ are fixed as constants, which may not generalize to all scenarios.
3. **Limited evaluation scale**: Due to API cost constraints, the method has not been validated on multilingual tasks, reasoning-intensive settings, or stronger adversarial configurations.
4. **Degradation on lexically precise tasks**: The worst-case reasoning mindset is counterproductive on tasks requiring exact lexical matching.

## Related Work & Insights

- **InstructZero** (Chen et al., 2024): The base framework, which models prompt optimization as BO in continuous space.
- **DRBO** (Kirschner et al., 2020): The theoretical foundation for distributionally robust Bayesian optimization, serving as the core technical source for this work.
- **APE / OPRO**: Other automatic prompt optimization methods that similarly suffer from distribution shift; the DRO approach is transferable to these frameworks.

## Rating

| Dimension | Score (1–5) |
|-----------|------------|
| Novelty | 3.5 |
| Theoretical Depth | 4.0 |
| Experimental Thoroughness | 3.5 |
| Writing Quality | 3.5 |
| Value | 4.0 |
| Overall | 3.7 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Training Large Language Models To Reason In Parallel With Global Forking Tokens](training_large_language_models_to_reason_in_parallel_with_global_forking_tokens.md)
- [\[ICML 2026\] Poison with Style: A Practical Poisoning Attack on Code Large Language Models](../../ICML2026/code_intelligence/poison_with_style_a_practical_poisoning_attack_on_code_large_language_models.md)
- [\[AAAI 2026\] SPAN: Benchmarking and Improving Cross-Calendar Temporal Reasoning of Large Language Models](../../AAAI2026/code_intelligence/span_benchmarking_and_improving_cross-calendar_temporal_reasoning_of_large_langu.md)
- [\[ACL 2026\] KoCo-Bench: Can Large Language Models Leverage Domain Knowledge in Software Development?](../../ACL2026/code_intelligence/koco-bench_can_large_language_models_leverage_domain_knowledge_in_software_devel.md)
- [\[AAAI 2026\] EquaCode: A Multi-Strategy Jailbreak Approach for Large Language Models via Equation Solving and Code Completion](../../AAAI2026/code_intelligence/equacode_a_multi-strategy_jailbreak_approach_for_large_language_models_via_equat.md)

</div>

<!-- RELATED:END -->
