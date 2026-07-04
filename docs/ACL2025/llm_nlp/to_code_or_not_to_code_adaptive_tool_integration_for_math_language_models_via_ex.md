---
title: >-
  [Paper Note] To Code or not to Code? Adaptive Tool Integration for Math Language Models via Expectation-Maximization
description: >-
  [ACL2025][LLM (Other)][Mathematical Reasoning] This work proposes AutoCode, an EM-framework-based method that enables mathematical LLMs to autonomously decide when to use code tools to assist reasoning. By guiding the exploration of high-potential code-triggering decisions in the E-step and optimizing via offline RL in the M-step, a 7B model achieves an $11\%+$ improvement on MATH500.
tags:
  - "ACL2025"
  - "LLM (Other)"
  - "Mathematical Reasoning"
  - "Tool Integration"
  - "Metacognition"
  - "EM Algorithm"
  - "Reinforcement Learning"
  - "Code Generation"
date: 2026-05-08
content_hash: cf5d974c6e09bb4d
---

# To Code or not to Code? Adaptive Tool Integration for Math Language Models via Expectation-Maximization

**Conference**: ACL2025  
**arXiv**: [2502.00691](https://arxiv.org/abs/2502.00691)  
**Code**: [HaozheH3/AutoCode](https://github.com/HaozheH3/AutoCode)  
**Area**: LLM/NLP  
**Keywords**: Mathematical Reasoning, Tool Integration, Metacognition, EM Algorithm, Reinforcement Learning, Code Generation

## TL;DR
This work proposes AutoCode, an EM-framework-based method that enables mathematical LLMs to autonomously decide when to use code tools to assist reasoning. By guiding the exploration of high-potential code-triggering decisions in the E-step and optimizing via offline RL in the M-step, a 7B model achieves an $11\%+$ improvement on MATH500.

## Background & Motivation

**Mathematical reasoning requires hybrid capabilities**: Mathematical problem-solving demands both abstract reasoning (CoT) and precise computation (code execution). These two are complementary but each has its limitations—CoT is prone to cascading numerical errors, while code generation suffers from a translation gap between semantics and symbols.

**Existing hybrid frameworks lack metacognitive abilities**: While models like Mammoth, DeepSeek-Math, and Qwen-2.5-Math support interleaved CoT and code, they rely on external instructions or fixed templates to decide when to use code, failing to choose dynamically based on their own capabilities.

**Fundamental limitations of the SFT paradigm**: Supervised Fine-Tuning forces models to passively imitate: (1) relying on user instructions ("Let's write a Python program"), (2) copying fixed code templates, and (3) mimicking teacher-forced tool-use trajectories, hindering the autonomous development of helper-tool strategies.

**Incompetence of standard RL in learning autonomous code integration**: RL tends to exploit local policy neighborhoods and suffers from insufficient exploration in the massive combinatorial space of interleaved CoT-code, making it difficult to discover high-reward hybrid reasoning paths.

**Experimental evidence: AutoCode capabilities of existing models are near-random**: Under autonomous mode, DeepSeek-Math-Instruct-7B performs $11.54\%$ worse than under explicit-instruction code mode, and its code-selection accuracy is close to $50\%$ (random level).

**Need for human-like metacognitive learning mechanisms**: Human learners learn when to use tools through trial and error, observing outcomes, and updating strategies. LLMs require a similar progressive exploration mechanism.

## Method

### Overall Architecture: EM for AutoCode
Code-triggering decisions $c \in \{0, 1\}$ are modeled as latent variables, executed alternately under the EM framework:
- **E-step (Guided Exploration)**: Establishes a reference policy $s(c|x_q)$ indicating whether an individual problem prefers code or pure reasoning.
- **M-step (Self-Optimization)**: Conducts offline RL optimization guided by the reference policy, simultaneously updating the tool-use policy and reasoning capabilities.

### Key Designs 1: E-step Reference Policy Estimation
The reference policy is estimated through Monte Carlo rollouts:
$$s^*(c|x_q) = \frac{\exp(\alpha \cdot \pi_\theta(c|x_q) Q(x_q, c; \theta))}{Z(x_q)}$$

- For each problem $x_q$, $K=8$ rollouts are generated under $c=0$ (pure reasoning) and $c=1$ (code integration), respectively.
- $Q(x_q, c; \theta)$ is computed as the expected success rate under each decision.
- The reference policy assigns higher probability to decisions with higher expected value while balancing the current prior $\pi_\theta(c|x_q)$ of the model.
- Prefix-guided generation is used to steer the generation of code-integrated solutions (e.g., "Let's first analyze the problem, then consider if python code could help").

### Key Designs 2: M-step Offline RL
The training set $\mathcal{D}_{\text{train}}$ is subsampled from rollout data based on the reference policy $s$, followed by optimization:
$$\underset{(x_q, y_a)}{\mathbb{E}}\left[\text{clip}\left(\frac{\pi_\theta(y_a|x_q)}{\pi_{\text{ref}}(y_a|x_q)}, 1-\epsilon, 1+\epsilon\right) \cdot A\right] - \mathbb{E}_{(x_q, c)}\left[\log \pi_\theta(c|x_q)\right]$$

- First term: A PPO-style clipped policy gradient to optimize the quality of reasoning generation.
- Second term: A cross-entropy term that aligns the model's code-triggering policy with the reference policy.
- The dual optimization objectives jointly improve the tool-use policy and reasoning capabilities.

### Loss & Training
- Base models: Qwen2-Math-Base-7B / DeepSeek-Math-7B / Qwen-2.5-Base-7B
- Training data: 119K public queries collected from Openmath, Math-Instruct, Metamath, and MMOS.
- Each EM iteration: $K=8$ rollouts are generated for 7K queries, followed by offline RL training after subsampling.
- Hardware: 8×A100 (80GB), taking approximately 10 hours to complete 3 epochs.

## Key Experimental Results

### Main Results: AutoCode4Math Main Results (Pass@1 accuracy %)

| Model | Code? | GSM8K | MATH500 | GaoKao | Olympiad | AIME24 | AMC23 |
|------|-------|-------|---------|--------|----------|--------|-------|
| GPT-4o | ✗ | 92.9 | 76.4 | 67.5 | 43.3 | 9.3 | 45.8 |
| NuminaMath-72B | ✓ | 91.4 | 59.2 | 49.4 | 36.7 | 6.5 | 40.6 |
| Qwen2.5-Base-7B | ✗ | 84.88 | 60.4 | 45.45 | 30.37 | 13.2 | 39.38 |
| **AutoCode-Qwen2.5** | **★** | **89.12** | **71.4** | **51.69** | **32.6** | **22.6** | **45.18** |
| Δ | | +4.24 | **+11.0** | +6.24 | +2.23 | **+9.4** | +5.8 |
| DeepSeek-Math-Inst-7B | ✓ | 84.46 | 51.0 | 44.68 | 20.44 | 1.6 | 17.4 |
| **AutoCode-DeepSeek** | **★** | **89.26** | **63.32** | **50.53** | **26.95** | **9.5** | **28.8** |
| Δ | | +4.8 | +12.32 | +5.85 | +6.51 | +7.9 | +11.4 |

### Ablation Study: Training Method Ablation Comparison

| Method | Training Efficiency | Code Call Rate | Convergence Performance | Core Problem |
|------|---------|-----------|---------|---------|
| Base+RL (DeepSeek-R1 style) | Low | <5% | Slow but steady improvement | Extremely inefficient to learn tool-use from scratch |
| SFT only | High (Initial) | Fixed templates | Early convergence | Cannot surpass demonstration data |
| SFT+RL | Medium | Trending towards polarization | Plateau | Insufficient exploration, trapped in local optima |
| **EM (Ours)** | **High** | **Adaptive ~90% selection accuracy** | **Steady improvement** | Guided exploration avoids local optima |

### Key Findings

1. **Up to 11% improvement on MATH500**: Qwen2.5-Base-7B increases from $60.4\%$ to $71.4\%$, and AIME24 improves from $13.2\%$ to $22.6\%$ ($+9.4\%$).
2. **Standard RL suffers from severe under-exploration on AutoCode**: During SFT+RL training, the distribution of code call rates gradually polarizes (towards $0\%$ or $100\%$), indicating that the model exploits local policies rather than exploring diverse paths.
3. **AutoCode selection accuracy reaches 89.53%**: AutoCode4Math-Qwen2.5 achieves nearly $90\%$ accuracy in CoT/Code selection on MATH500, vastly outperforming the baseline of $\sim 50\%$.
4. **"No free lunch" effect validated**: AutoCode performance surpasses both pure CoT and pure Code under explicit instructions, demonstrating that autonomous integration indeed yields synergetic gains.
5. **Consistent effectiveness across model families**: Significant improvements are observed across three base models: Qwen2-Math, DeepSeek-Math, and Qwen-2.5.

## Highlights & Insights

- **Precise problem definition**: The gap of "metacognitive tool-use" is clearly defined—existing models can use tools but do not know when to use them, contrasting with human metacognitive abilities.
- **Elegant application of the EM framework**: Modeling code-triggering decisions as latent variables naturally decouples "exploring the optimal policy" (E-step) from "learning that policy" (M-step), resolving the issue of insufficient exploration in RL.
- **Visualization of under-exploration in SFT+RL**: Figure 5 illustrates the polarization trend of code call rate distribution in SFT+RL training, intuitively proving the local exploitation issue.
- **Practical utility of prefix-guided generation**: Simple prefix guidance suffices to induce the model to explore code-integrated paths, bypassing complex reward shaping.
- **Highly efficient implementation**: Training is completed in only 10 hours using 7K queries on 8×A100, which is extremely friendly to resource-constrained researchers.

## Limitations & Future Work

1. **Limited to the mathematical domain**: The generalizability to other fields requiring tool integration, such as scientific reasoning and general-purpose code generation, has not been validated.
2. **Lack of comparison with o1-like long-CoT models**: The paper explicitly excludes models that rely on test-time scaling (such as MCTS and long CoT), leaving out a direct comparison with DeepSeek-R1 and similar models.
3. **Binary code-triggering decision only**: The current framework simplifies $c$ to 0/1, without considering fine-grained tool selection (e.g., when to use a calculator, a symbolic solver, or search).
4. **Rollout costs in the E-step**: Each EM iteration requires generating $16$ rollouts per query ($8$ each for $c=0$ and $c=1$), resulting in a computational cost for data curation that scales linearly with training epochs.
5. **Base model dependency**: It requires the base model to already possess basic code generation capabilities, which may not apply to pure language models.
6. **Emergence of code-triggering in intermediate steps**: The paper mentions that mid-reasoning code triggers naturally emerge after warm-up, but lacks deep analysis regarding the conditions and mechanisms of this emergence.

## Related Work & Insights

### vs DeepSeek-R1 (Guo et al., 2025)
DeepSeek-R1 demonstrates that pure RL can enhance reasoning capabilities, but experiments in this paper show that pure RL is extremely inefficient in learning code integration policies (code call rate $<5\%$). The crucial difference lies in: the target space of R1 is the quality of the reasoning chain, whereas the target space of AutoCode is a combinatorial space of interleaved CoT-Code, which is exponentially harder to explore. The EM framework bridge this gap through guided exploration in the E-step.

### vs ToRA (Gou et al., 2023) / Mammoth (Yue et al., 2023)
ToRA and Mammoth are representative of early CoT-Code hybrid approaches, but both rely on SFT to learn fixed tool-use patterns from curated data. Figure 1 clearly highlights the "rigidity" problem of these models—their AutoCode performance is inferior to explicit-instruction modes because their code-triggering strategies essentially mimic the static patterns in training data rather than making autonomous decisions.

### vs Singh et al. (2023) / Ni et al. (2022) EM-style self-training
These works apply EM to mathematical self-training (iteratively generating correct solutions $\rightarrow$ retraining) but do not address tool-use decisions. The core innovation of this paper is utilizing code-triggering decisions as latent variables of the EM algorithm, making the EM framework serve the learning of metacognitive abilities instead of just improving reasoning quality.

## Rating
- Novelty: 8/10 — The application of the EM framework to model tool-use decisions is novel, and the problem definition from a "metacognitive" perspective is clear and unique.
- Experimental Thoroughness: 7/10 — While the ablation studies are comprehensive, the benchmarks are primarily mathematical, lacking cross-domain generalization and direct comparison with o1-like models.
- Writing Quality: 8/10 — The motivation is clear, the mathematical derivations are thorough, and the visualizations (Figures 5/6) are highly convincing.
- Value: 8/10 — Provides a feasible and highly efficient training framework for autonomous tool-use in LLMs, delivering significant empirical improvements.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] ToolCoder: A Systematic Code-Empowered Tool Learning Framework for Large Language Models](toolcoder_code_empowered_tool_learning.md)
- [\[ACL 2025\] OpenCoder: The Open Cookbook for Top-Tier Code Large Language Models](opencoder_the_open_cookbook_for_top-tier_code_large_language_models.md)
- [\[ACL 2025\] Interactive and Expressive Code-Augmented Planning with Large Language Models](interactive_and_expressive_code-augmented_planning_with_large_language_models.md)
- [\[ACL 2025\] WarriorCoder: Learning from Expert Battles to Augment Code Large Language Models](warriorcoder_learning_from_expert_battles_to_augment_code_large_language_models.md)
- [\[ACL 2025\] STEM-PoM: Evaluating Language Models Math-Symbol Reasoning in Document Parsing](stem-pom_evaluating_language_models_math-symbol_reasoning_in_document_parsing.md)

</div>

<!-- RELATED:END -->
