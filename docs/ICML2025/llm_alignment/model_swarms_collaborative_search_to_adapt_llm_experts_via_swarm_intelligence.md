---
title: >-
  [Paper Note] Model Swarms: Collaborative Search to Adapt LLM Experts via Swarm Intelligence
description: >-
  [ICML 2025][LLM Alignment][Swarm Intelligence] Drawing inspiration from the Particle Swarm Optimization (PSO) algorithm, this work treats multiple LLM experts as "particles" collaboratively searching in the weight space. Guided by three signals—individual best, global best, and global worst—the experts iteratively update their positions. This achieves tuning-free model adaptation using only 200 samples, outperforming 12 baselines by an average of 13.3% across 9 tasks.
tags:
  - "ICML 2025"
  - "LLM Alignment"
  - "Swarm Intelligence"
  - "Model Fusing"
  - "Particle Swarm Optimization"
  - "LLM Expert Adaptation"
  - "Weak-to-Strong Generalization"
date: 2026-05-08
content_hash: 8c7d036620e54e22
---

# Model Swarms: Collaborative Search to Adapt LLM Experts via Swarm Intelligence

**Conference**: ICML 2025  
**arXiv**: [2410.11163](https://arxiv.org/abs/2410.11163)  
**Code**: [BunsenFeng/model_swarm](https://github.com/BunsenFeng/model_swarm)  
**Area**: LLM Alignment / Model Composition  
**Keywords**: Swarm Intelligence, Model Fusing, Particle Swarm Optimization, LLM Expert Adaptation, Weak-to-Strong Generalization

## TL;DR

Drawing inspiration from the Particle Swarm Optimization (PSO) algorithm, this work treats multiple LLM experts as "particles" collaboratively searching in the weight space. Guided by three signals—individual best, global best, and global worst—the experts iteratively update their positions. This achieves tuning-free model adaptation using only 200 samples, outperforming 12 baselines by an average of 13.3% across 9 tasks.

## Background & Motivation

Current LLM composition methods mainly fall into three categories, each having its own limitations:

**MoE Routing**: Routes queries to different experts without generating a new model, rendering it helpless when facing tasks that exceed the capabilities of existing experts.

**Learn-to-Fuse**: Designs trainable components to "glue" experts together, but requires large amounts of supervised training data and exhibits low modularity, making it difficult to flexibly add or remove experts.

**Model Arithmetic**: Combines experts via arithmetic operations on weights or probabilities (e.g., Task Arithmetic, TIES-Merging), but relies on strong assumptions about experts and their combination modes (e.g., "indoor lion = outdoor lion + (indoor dog - outdoor dog)").

The authors point out that what is truly needed is a flexible adaptation method that **does not require excessive training data and does not rely on prior assumptions about experts**. Inspired by Particle Swarm Optimization (PSO), Model Swarms treats each LLM expert as a particle in the weight space, automatically discovering the optimal model combination through collaborative search in low-data scenarios (only 200 samples).

## Method

### Overall Architecture

The core idea of Model Swarms is to formulate the LLM adaptation problem as a **collaborative search problem in the weight space**:

- **Input**: A group of LLM experts $\{x_i\}$ and a utility function $f: x \to \mathbb{R}$ (such as validation set accuracy or reward model score).
- **Process**: Each expert particle has a **position** (model weights) and a **velocity** (direction in the weight space), iteratively updating under the guidance of the individual best, global best, and global worst.
- **Output**: Returns the global best expert $g$ after the search terminates.

The entire process is a three-step loop: Initialization $\to$ Velocity Update $\to$ Weight Update $\to$ Iteration Termination Check.

### Key Designs

1. **Pairwise Crossover (Initialization Expansion)**: Expands $n$ initial experts to $N$ particles through pairwise linear interpolation. Two experts $x_a$ and $x_b$ are randomly selected, a parameter $t \sim U(0,1)$ is sampled, and a new particle is generated as $x_{\text{new}} = t \cdot x_a + (1-t) \cdot x_b$. The initial velocity of each particle is set pointing towards another random particle, $v_i = \text{random}(x_j) - x_i$, preventing all particles from collapsing into a "black hole" at the global best.

2. **Velocity Update**: The velocity is determined by the weighted average of four factors, with $C$ being the normalization term. The four components represent:

    - **Inertia term** $\phi_v \cdot v_i$: Keeps the current velocity direction to maintain search momentum.
    - **Cognitive term** $\phi_p \cdot (p_i - x_i)$: Attracted to its own historical personal best position.
    - **Social term** $\phi_g \cdot (g - x_i)$: Attracted to the global best position.
    - **Repulsion term** $\phi_w \cdot (g_w - x_i)$: Repelled from the global worst position (an innovative extension of classic PSO).
    - **Random factors** $r_v, r_p, r_g, r_w \sim U(0,1)$: Ensuring search non-determinism and enhancing exploration capabilities.

3. **Weight Update & Restart**: The position update formula is $x_i \leftarrow x_i + \lambda \cdot v_i$, where $\lambda$ is the step size. If a particle's personal best does not improve within $c_r$ iterations, the particle is **restarted** at its personal best position with its velocity reset to zero, granting it a "second chance." This strikes a balance between exploration and robustness.

4. **Token Swarms (Cross-Architecture Variant)**: When experts originate from different architectures (such as Gemma and Mistral), weight-space manipulation is impossible. Token Swarms defines a particle's position as an $n$-dimensional vector (initially a one-hot vector) and performs linear combination searches over the token probability distribution, enabling cross-architecture expert collaboration.

### Loss & Training

Model Swarms **does not require any gradient updates or supervised training**. It completely relies on the utility function $f$ (a mapping from model to scalar) to guide the search:

- **Single Task**: $f$ is the validation performance (e.g., accuracy).
- **Multi-Task Domain**: $f$ is the harmonic average of performance across multiple tasks.
- **Reward Model**: $f$ is the score evaluated by the RM on validation instructions.
- **Human Preferences**: $f$ is the 1-10 rating from an LLM-as-a-judge.

Hyperparameter settings: swarm size $N=20$, step decay $\phi_\lambda=0.95$, patience $c=10$, restart patience $c_r=5$, maximum iterations $K=50$. The step size is multiplied by $\phi_\lambda$ every round to decay gradually, achieving a coarse-to-fine search.

## Key Experimental Results

### Main Results

Based on Gemma-7B, 10 LoRA experts were obtained by fine-tuning on 10 SFT domains of Tulu-v2.

| Dataset | Metrics | Model Swarms | Best Baseline | Gain |
|--------|------|-------------|----------|------|
| MMLU | Acc (test) | .583 | .568 (Pack of LLMs) | +2.6% |
| MMLU-pro | Acc (test) | .254 | .237 (Slerp) | +7.2% |
| Hellaswag | Acc (test) | .652 | .622 (Dare-Ties) | +4.8% |
| K-Crossword | Acc (test) | .428 | .372 (Dare-Ties) | +15.1% |
| GSM8k | Acc (test) | .459 | .354 (EvolMerge) | +29.7% |
| NLGraph | Acc (test) | .672 | .568 (LoraHub) | +18.3% |
| TruthfulQA | Acc (test) | .392 | .359 (LoraHub) | +9.2% |
| RealToxicityPrompts | Score (test) | .956 | .885 (LoraHub) | +8.0% |
| AbstainQA | Score (test) | .175 | .140 (Dare-Ties) | +25.0% |

The largest gains are achieved on reasoning tasks (GSM8k, K-Crossword, NLGraph), averaging 21.0%.

### Ablation Study

| Configuration | MMLU | Hellaswag | NLGraph | AbstainQA | Description |
|------|------|-----------|---------|-----------|------|
| Full Model Swarms | .583 | .652 | .672 | .175 | Full method |
| Fully deterministic (excluding all randomness) | .528 | .611 | .541 | .072 | Average decrease of 23.5% |
| Crossover only 15 times | .527 | .604 | .534 | .093 | Insufficient initial particle diversity |
| No Crossover | .504 | .587 | .530 | .099 | Significant performance degradation |
| Diversity 1×10 (Lowest) | — | — | — | — | Baseline |
| Diversity 10×1 (Highest) | — | — | — | — | 35.3% improvement over 1×10 |

### Key Findings

1. **Correctness Emergence (C-emerge)**: 36%-53.5% of the questions initially answered incorrectly by all experts were answered correctly by at least one expert after Model Swarms—indicating that collaborative search **discovers new capabilities not possessed by any of the initial models**.
2. **Diamond in the Rough**: 89.6% of the final optimal particles were not among the initial best, and 56.9% even originated from the bottom half of the initial ranking—indicating that weak models contain unactivated, implicit expertise.
3. **Weak-to-Strong**: After removing the strongest expert, Model Swarms results for the remaining weak experts still outpaced the strongest expert by an average of 35.4%; using only the bottom 50% of experts still won on 2/3 of the datasets.
4. **Diversity is Crucial**: 10 different experts (10×1) perform 35.3% better on average than repeating a single expert 10 times (1×10).
5. **Reward Model Adaptation**: Using only 200 instructions, Model Swarms outperforms both PPO and DPO, and achieves optimality on both conflicting preferences (verbose and concise), showcasing superb controllability.
6. **Human Preference Adaptation**: On 16 topics, LLM-judge ratings improve by 17.6% on average, factuality increases by 17.0%, and the win rate against human evaluations reaches 70.8%.

## Highlights & Insights

- **Elegant Analogy**: Maps LLM adaptation into a particle swarm optimization problem, yielding a concept that is both simple and powerful. Each expert is treated as a particle, the utility function acts as fitness, and the search process automatically discovers the optimal combination.
- **Extremely Low Data Requirement**: Requiring only 200 samples for the utility function without requiring any gradient training makes this approach highly valuable in data-scarce scenarios.
- **Stunning Emergence Capabilities**: Collaborative search is not simply a matter of knowledge transfer; rather, it discovers new solutions in the capability space that none of the initial experts possessed (C-emerge as high as 53.5%).
- **Weak-to-Strong Transition**: The final optimal expert is often not the one that started out best—challenging the intuition of "selecting the strongest model" and demonstrating that implicit expertise can be activated through collaborative search.
- **Global Worst Repulsion Term**: A repulsion force away from the global worst position is introduced on top of classic PSO, helping explore more effective regions.
- **Token Swarms Extension**: Performing search in the probability distribution space enables models from different architectures to collaborate.

## Limitations & Future Work

1. **Computational Overhead**: Evaluating the utility function for all $N$ particles at each iteration is costly for large models. The paper proposes a dropout-style acceleration strategy but does not elaborate on it in detail.
2. **Selection of Initial Experts**: Diversity is crucial to the method, but how to select the optimal initial pool of experts from over 900k+ models on Hugging Face remains an open question.
3. **Local Optima Trap**: Despite incorporating various sources of randomness and restart mechanisms, the search can still get trapped in local optima. The paper suggests expanding the random factor from $U(0,1)$ to $U(-0.2,1)$ to mitigate this issue.
4. **Preliminary Token Swarms**: The cross-architecture variant only manipulates the linear combination of probability distributions and does not touch model parameters, leaving limited room for capacity improvement.
5. **Recency Restrictions**: The method relies on the existing knowledge of the experts; thus, it cannot introduce new information not seen in the training data.
6. **Dual-Use Risk**: A flexible utility function means that malicious actors could also optimize for harmful objectives.

## Related Work & Insights

- **Difference from EvolMerge**: EvolMerge employs genetic algorithms to search for combinations of weights/layers, which requires manually defining crossover and mutation rules. In contrast, Model Swarms automates this search using PSO, eliminating manual engineering.
- **Difference from LoraHub**: LoraHub optimizes LoRA weight coefficients via gradient descent. Conversely, Model Swarms bypasses gradients to search directly within the weight space.
- **Insights**: The approach of directly applying optimization algorithms (such as PSO) to the model weight space can be extended to other scenarios, such as prompt search and hyperparameter optimization. The "weak-to-strong" emergence phenomenon also implies that differentiated experts hold substantially greater value than homogeneous ones.

## Rating

- Novelty: ⭐⭐⭐⭐ (Although PSO is a mature algorithm, applying it to the LLM weight space and observing emergence is highly novel)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (4 types of adaptation targets, 9+8+3+16 evaluation setups, 12+2 baselines, and comprehensive ablation analysis)
- Writing Quality: ⭐⭐⭐⭐⭐ (Clear structure, precise analogies, and in-depth analyses)
- Value: ⭐⭐⭐⭐ (High practical value for low-data, training-free LLM adaptation; however, computational cost and initial expert selection remain deployment bottlenecks)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Improving Model Alignment through Collective Intelligence of Open-Source LLMs](improving_model_alignment_through_collective_intelligence_of_open-source_llms.md)
- [\[ICLR 2026\] Multi-objective Large Language Model Alignment with Hierarchical Experts](../../ICLR2026/llm_alignment/multi-objective_large_language_model_alignment_with_hierarchical_experts.md)
- [\[ICML 2025\] On the Robustness of Reward Models for Language Model Alignment](on_the_robustness_of_reward_models_for_language_model_alignment.md)
- [\[ACL 2026\] HarDBench: A Benchmark for Draft-Based Co-Authoring Jailbreak Attacks for Safe Human–LLM Collaborative Writing](../../ACL2026/llm_alignment/hardbench_a_benchmark_for_draft-based_co-authoring_jailbreak_attacks_for_safe_hu.md)
- [\[ICML 2025\] AlphaPO: Reward Shape Matters for LLM Alignment](alphapo_reward_shape_matters_for_llm_alignment.md)

</div>

<!-- RELATED:END -->
