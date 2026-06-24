---
title: >-
  [Paper Note] OR-R1: Automating Modeling and Solving of Operations Research Optimization Problems
description: >-
  [AAAI 2026][Operations Research] OR-R1 proposes a data-efficient two-stage training framework (SFT + TGRPO) that achieves an average solving accuracy of 67.7% using only 1/10 of the synthetic data required by ORLM, surpassing existing SOTA methods. Additionally, test-time reinforcement learning reduces the performance gap between single-sample generation (Pass@1) and multi-sample generation (Pass@8) from 13% to 7%.
tags:
  - "AAAI 2026"
  - "Operations Research"
  - "LLM Fine-tuning"
  - "Reinforcement Learning"
  - "Test-Time Adaptation"
  - "Data Efficiency"
date: 2026-05-08
content_hash: 7190b5b9aa89d26c
---

# OR-R1: Automating Modeling and Solving of Operations Research Optimization Problems

**Conference**: AAAI 2026  
**arXiv**: [2511.09092](https://arxiv.org/abs/2511.09092)  
**Code**: [GitHub](https://github.com/SCUTE-ZZ/OR-R1)  
**Area**: Other  
**Keywords**: Operations Research, LLM Fine-tuning, Reinforcement Learning, Test-Time Adaptation, Data Efficiency

## TL;DR

OR-R1 proposes a data-efficient two-stage training framework (SFT + TGRPO) that achieves an average solving accuracy of 67.7% using only 1/10 of the synthetic data required by ORLM, surpassing existing SOTA methods. Additionally, test-time reinforcement learning reduces the performance gap between single-sample generation (Pass@1) and multi-sample generation (Pass@8) from 13% to 7%.

## Background & Motivation

Operations Research (OR) optimization problems are prevalent in industrial settings such as logistics, resource allocation, and scheduling. Traditionally, converting natural language problem descriptions into precise mathematical models and executable solver code requires highly specialized manual effort, which is both time-consuming and error-prone.

Recent advances in large language models (LLMs) have demonstrated potential for automating OR modeling and solving. Existing approaches fall into two categories:

**Prompt-based methods**: These leverage the in-context learning capabilities of LLMs to directly generate optimization models or code through carefully designed prompts, few-shot examples, or chain-of-thought reasoning. Representative works include Chain-of-Experts, Optimus, and MAMO, none of which require domain-specific fine-tuning.

**Learning-based methods**: These fine-tune LLMs on domain-specific data. Representative works include ORLM (using Llama3-8B, outperforming GPT-4o on several benchmarks) and LLMOPT. While achieving stronger performance, these methods face two core challenges:

**High data requirements**: ORLM uses 30,000 synthetic training samples, yet the accuracy of synthetic data is only approximately 70%, and manual annotation is prohibitively expensive. How can labeling data requirements be substantially reduced?

**Poor output consistency**: A significant gap (~13%) exists between a model's single-sample output (Pass@1) and the best result across multiple samples (Pass@8), indicating that the model possesses the capability but lacks consistency. How can the reliability of single-sample generation be improved?

OR-R1 is designed specifically to address these two challenges.

## Method

### Overall Architecture

OR-R1 adopts a two-stage training pipeline based on Qwen3-8B:

1. **Stage 1: Supervised Fine-Tuning (SFT)** — A small amount of labeled data is used to teach the model the basic reasoning patterns for OR modeling and code generation.
2. **Stage 2: Test-time Group Relative Policy Optimization (TGRPO)** — Reinforcement learning on unlabeled test data further improves capability and consistency.

### Key Designs

#### 1. Supervised Fine-Tuning (SFT) Stage

The SFT stage minimizes the standard negative log-likelihood loss:

$$\mathcal{L}_\text{SFT}(\theta) = -\mathbb{E}_{(x,o)\sim\mathcal{D}_\text{SFT}}\left[\sum_{t=1}^{|o|}\log P(o_t|x,o_{<t};\theta)\right]$$

Key characteristic: only **3,000** (or even **100**) synthetic samples from the ORInstruct dataset are used for training, representing 1/10 of the data used by ORLM. The purpose of SFT is not to achieve peak performance, but to equip the model with the basic paradigm for OR problem modeling and solver code generation, laying the foundation for the subsequent RL stage.

#### 2. Test-time Group Relative Policy Optimization (TGRPO)

TGRPO is the core contribution of this work. It integrates the GRPO mechanism from DeepSeek-R1 with test-time reinforcement learning (TTRL) for the OR domain.

**Core Idea**: For unlabeled test problems, the LLM generates a group of $G$ candidate outputs. Majority voting determines a pseudo-label, which then serves as the reward signal for policy optimization.

The objective function inherits the clipped policy from PPO:

$$\mathcal{J}_\text{TGRPO}(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^G\left(\min\left(\frac{\pi_\theta(o_i|q)}{\pi_{\theta_\text{old}}(o_i|q)}A_i, \text{clip}(\cdot, 1-\epsilon, 1+\epsilon)A_i\right) - \beta\mathbb{D}_\text{KL}(\pi_\theta||\pi_\text{ref})\right)\right]$$

The advantage function is estimated via within-group normalization: $A_i = \frac{R_i - \text{mean}(\{R_1,...,R_G\})}{\text{std}(\{R_1,...,R_G\})}$

**Key Advantages**:
- Eliminates the need for a separate Critic model by estimating the baseline from within-group scores, substantially reducing computational overhead.
- Trains on **unlabeled** test data without additional annotation costs.
- Generates high-quality pseudo-labels via majority voting, enabling self-supervised learning.

#### 3. Composite Reward Function Design

Three complementary reward signals are tailored for the OR setting:

**Format Reward**: Checks whether the output contains 6 required fields (mathematical model, decision variables, objective function, constraints, Python code, and Python code block markers), scored proportionally:

$$R_\text{format}(o_i) = \frac{\text{number of detected required fields}}{6}$$

**Valid-Code Reward**: Checks whether the generated code correctly invokes the coptpy solver, yielding a binary reward:

$$R_\text{code}(o_i) = \begin{cases}1, & \text{code correctly invokes coptpy} \\ 0, & \text{otherwise}\end{cases}$$

**Majority Voting Reward**: Derived from the TTRL framework, a consensus output is established via majority voting as a proxy label:

$$R_\text{voting}(y_i, y) = \begin{cases}1, & y_i = y \\ 0, & \text{otherwise}\end{cases}$$

where $y$ denotes the majority vote result, considering only results from successfully executed code (excluding "No Best Solution" or "None").

The final composite reward is: $R_i = R_\text{format}(o_i) + R_\text{code}(o_i) + R_\text{voting}(y_i, y)$

### Loss & Training

- **SFT stage**: AdamW optimizer with warmup-decay learning rate scheduling.
- **TGRPO stage**: AdamW + cosine learning rate scheduling + PEFT (LoRA), with KL divergence regularization to prevent excessive policy drift.
- Hardware: 4×A100 (40G) GPUs, BF16 precision.
- TGRPO yields significant gains with as few as ~50 unlabeled samples.

## Key Experimental Results

### Main Results

| Model | NL4OPT | MAMO EasyLP | MAMO ComplexLP | IndustryOR | NLP4LP | ComplexOR | OptiBench | ICML Comp. | AVG |
|------|--------|-------------|----------------|------------|--------|-----------|-----------|------------|-----|
| ORLM (Llama3-8B) | 86.9 | 81.6 | 39.3 | 32.0 | 82.0 | 50.0 | 56.5 | 79.3 | 63.5 |
| LLMOPT (Qwen2.5-14B) | 80.3 | 89.5 | 44.1 | 29.0 | 73.4 | 35.3 | 53.8 | 75.3 | 60.1 |
| Qwen3-8B SFT(3K) | 86.0 | 87.0 | 39.9 | 33.0 | 82.9 | 40.7 | 61.4 | 85.8 | 64.6 |
| **OR-R1 SFT(100)-TGRPO** | 88.0 | 87.4 | 45.7 | 30.3 | 84.0 | 46.3 | 61.2 | 84.1 | **65.9** |
| **OR-R1 SFT(3K)-TGRPO** | 88.3 | 86.1 | 49.9 | 35.3 | 84.6 | 46.3 | 62.9 | 88.3 | **67.7** |

OR-R1 surpasses ORLM (63.5%→67.7%, +4.2%) using only 1/10 of the data. Even with only 100 SFT samples + TGRPO (65.9%), the method outperforms ORLM trained on 30,000 samples.

### Ablation Study: Reward Component Analysis

| Reward Configuration | AVG | Change vs. SFT Baseline |
|---------|-----|---------------|
| SFT(3K) Baseline | 66.0 | — |
| +RL($R_\text{format}$) | 66.5 | +0.5 |
| +RL($R_\text{code}$) | 67.2 | +1.2 |
| +RL($R_\text{voting}$) | 68.0 | +2.0 |
| +RL($R_\text{format}$+$R_\text{code}$) | 67.8 | +1.8 |
| +RL($R_\text{format}$+$R_\text{voting}$) | 68.5 | +2.5 |
| +RL($R_\text{code}$+$R_\text{voting}$) | 69.7 | +3.7 |
| +RL(All three) | **70.8** | **+4.8** |

The three reward components are complementary, and combining all three yields the best performance. The voting reward contributes the most individually (+2.0), followed by the valid-code reward (+1.2), and the format reward contributes the least (+0.5).

### Key Findings

1. **Extreme data efficiency**: 100 SFT samples + TGRPO suffices to surpass ORLM trained on 30,000 samples, representing a 300× reduction in data requirements.
2. **Substantial consistency improvement**: The gap between Pass@1 and Pass@8 narrows from 13% to 7%, demonstrating that TGRPO effectively improves single-sample generation reliability.
3. **TGRPO data efficiency**: The primary performance gains are achieved with as few as ~50 unlabeled samples; further increases in data yield diminishing returns.
4. **Complementarity of reward components**: The three rewards respectively govern structural correctness, code executability, and numerical accuracy; combining all three yields clear improvements over any subset.
5. **Training not saturated**: Pass@1 continues to rise at the end of training; computational constraints prevented further training, suggesting that additional compute could yield further gains.

## Highlights & Insights

- **Breakthrough in data efficiency**: The problem of low-quality synthetic data (~70% accuracy reported by ORLM) and expensive manual annotation is elegantly addressed by TGRPO's self-supervised mechanism — the model learns consistency from its own repeated sampling on unlabeled data.
- **Viability of majority voting as pseudo-labels**: In OR settings, code execution results are objective numerical values, allowing majority voting to identify correct answers with high reliability. This makes the TTRL framework particularly well-suited for this domain.
- **Hierarchical design of composite rewards**: The reward structure progresses from format → executability → numerical accuracy, forming a coarse-to-fine hierarchy that ensures the model receives effective guidance at every stage of the generation pipeline.
- **Systematic analysis of the Pass@1 vs. Pass@8 gap**: The consistency problem is explicitly formulated and quantified rather than focusing solely on best-case performance.

## Limitations & Future Work

1. **Test data used for training**: TGRPO trains on unlabeled test data and, while ground-truth labels are not observed, concerns about distributional leakage remain — fine-tuning on the evaluation set may overestimate generalization capability.
2. **Solver dependency**: Both the reward function and evaluation are tied to the coptpy solver; adapting the framework to other solvers (e.g., Gurobi, CPLEX) would require non-trivial modifications.
3. **Insufficient training**: The authors acknowledge that training was terminated early due to computational constraints, and performance curves were still rising; the true performance ceiling remains unknown.
4. **Unstable performance on ComplexOR**: On the ComplexOR benchmark, the full method occasionally underperforms the SFT baseline (e.g., +RL($R_\text{code}$) drops by 5.5%), suggesting that composite rewards may introduce conflicting signals on complex problems.
5. **Limited to linear/mixed-integer programming**: The work does not address more complex OR problem types such as nonlinear optimization or combinatorial optimization.

## Related Work & Insights

- TGRPO in OR-R1 directly builds upon the GRPO mechanism from DeepSeek-R1 and the test-time reinforcement learning ideas from TTRL, applying these techniques to the OR domain for the first time. This demonstrates that RL methods are particularly effective in settings where **majority voting can yield reliable pseudo-labels**.
- The composite reward design philosophy (format + executability + numerical correctness) is generalizable to other domains requiring LLMs to generate executable code, such as scientific computing code generation and database query generation.
- The data efficiency breakthrough (100 samples → 65.9%) showcases the potential of the SFT+RL two-stage paradigm in low-resource settings, offering inspiration for specialized domains where annotated data is expensive, such as medicine and law.

## Rating

- **Theoretical Depth**: ★★★☆☆ — The method design is clear but theoretical contributions are limited; the work primarily represents a compositional application of existing RL techniques.
- **Experimental Thoroughness**: ★★★★★ — Eight diverse benchmarks, multiple ablation studies, training dynamics analysis, and data scaling effect analysis.
- **Novelty**: ★★★★☆ — First application of TTRL to the OR domain; the data efficiency of TGRPO is impressive.
- **Practicality**: ★★★★★ — Directly lowers the data and expertise barriers for OR automation, with strong potential for industrial deployment.
- **Overall**: 8/10

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Hybrid-Balance GFlowNet for Solving Vehicle Routing Problems](../../NeurIPS2025/others/hybrid-balance_gflownet_for_solving_vehicle_routing_problems.md)
- [\[ACL 2025\] MIR: Methodology Inspiration Retrieval for Scientific Research Problems](../../ACL2025/others/mir_methodology_inspiration_retrieval_for_scientific_research_problems.md)
- [\[AAAI 2026\] Certified Branch-and-Bound MaxSAT Solving (Extended Version)](certified_branch-and-bound_maxsat_solving_extended_version.md)
- [\[AAAI 2026\] More Than Irrational: Modeling Belief-Biased Agents](more_than_irrational_modeling_belief-biased_agents.md)
- [\[AAAI 2026\] HyperSHAP: Shapley Values and Interactions for Explaining Hyperparameter Optimization](hypershap_shapley_values_and_interactions_for_explaining_hyperparameter_optimiza.md)

</div>

<!-- RELATED:END -->
