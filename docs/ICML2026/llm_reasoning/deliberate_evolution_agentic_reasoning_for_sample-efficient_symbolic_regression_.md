---
title: >-
  [Paper Note] Deliberate Evolution: Agentic Reasoning for Sample-Efficient Symbolic Regression with LLMs
description: >-
  [ICML 2026][LLM Reasoning][Symbolic Regression] The "proposal-scoring" loop of LLM-led symbolic regression is decomposed into two layers: "Proposal vs. Navigation." The LLM is explicitly guided by three signals: adaptive…
tags:
  - "ICML 2026"
  - "LLM Reasoning"
  - "Symbolic Regression"
  - "Evolutionary Search"
  - "agentic reasoning"
  - "Sample Efficiency"
  - "LLM Tool Calling"
date: 2026-05-08
content_hash: f20941e2871c27a8
---

# Deliberate Evolution: Agentic Reasoning for Sample-Efficient Symbolic Regression with LLMs

**Conference**: ICML 2026  
**arXiv**: [2606.04360](https://arxiv.org/abs/2606.04360)  
**Code**: https://github.com/Xinyu-Pang/Deliberate-Evolution (Available)  
**Area**: LLM Reasoning / Agent / Symbolic Regression  
**Keywords**: Symbolic Regression, Evolutionary Search, agentic reasoning, Sample Efficiency, LLM Tool Calling

## TL;DR
The "proposal-scoring" loop of LLM-led symbolic regression is decomposed into two layers: "Proposal vs. Navigation." The LLM is explicitly guided by three signals: adaptive operators (direction), diagnostic tools (residual/dimensionality), and reflective memory (trajectory experience). This approach reduces average NMSE by 37–55% on LLM-SRBench using only 40% of the evaluation budget.

## Background & Motivation

**Background**: Current mainstream LLM-based symbolic regression (LLM-SR, LASR, SGA) follows an "evolutionary optimization" paradigm: LLM proposes candidate expressions → BFGS fits constants → MSE provides scoring → feedback is returned to the LLM for further mutation.

**Limitations of Prior Work**: This cycle is highly sample-inefficient, averaging $10^3$ candidate evaluations per problem. This is because the LLM only receives a "parent expression + a scalar MSE," forcing it to simultaneously reason about three things: how to modify, why it is wrong, and what the past experiences were.

**Key Challenge**: Existing methods force "proposal" and "search guidance"—two tasks that should be decoupled—into the same prompt. Scalar feedback lacks three types of critical signals: **Direction** (should it refine or regenerate?), **Diagnosis** (does the residual contain periodicity or dimensional errors?), and **Memory** (which motifs succeed repeatedly, and which edits consistently fail?). Consequently, the search drifts among candidates that are "plausible but uninformative."

**Goal**: To explicitize and modularize these three signals, allowing the LLM to focus on its strength—proposing symbolic skeletons—while delegating "navigation" to deterministic modules.

**Key Insight**: The authors redefine symbolic regression as "guided scientific search" rather than "scored trial-and-error." If the success rate of each step is improved from $p_0$ to $p_0+\gamma$ through clear intent, localized evidence, and accumulated experience, the hitting time theoretically shortens exponentially ($\Pr(N>k)\le \exp[-k(p_0+\gamma)]$).

**Core Idea**: Use an agentic framework to **decouple** symbolic generation from search control. The LLM only proposes skeletons, while external modules determine direction, perform diagnosis, and accumulate memory.

## Method

### Overall Architecture

At each round $t$: A parent expression $f_p$ is sampled from population $P_t$ via Boltzmann sampling → An operator $o_t$ (direction) is sampled from an operator policy $\pi_{s_t}^{(t)}$ indexed by the parent state $s_t$ → A toolset $\mathcal{T}=\{T_{\text{data}},T_{\text{res}},T_{\text{dim}}\}$ is called to generate a diagnostic report $a_t$ (diagnosis) → Reflective memory $M_{t-1}$ (history) is retrieved → The LLM generates a skeleton $\tilde f_t$ under the proposal distribution $p_\theta(\cdot\mid Q,f_p,o_t,a_t,M_{t-1})$ → Constants are fitted via BFGS to obtain $f_t$ → The population, operator weights, and memory are updated. The budget limit is $T=400$ (only 1/2.5 of the baseline).

### Key Designs

1.  **Adaptive Operators (Directional Guidance)**:
    - **Function**: Decides for the LLM whether the current step should "refine, mutate, crossover, or regenerate," preventing aimless guessing.
    - **Mechanism**: Defines an operator set $\mathcal{O}=\{o_{\text{ref}},o_{\text{mut}},o_{\text{cross}},o_{\text{reg}}\}$ (exploit → explore). A binary state $s_t=(\mathbb{I}[\tilde r_p\le \tau_r],\mathbb{I}[v_p\ge\tau_v])$ (quality × frequency of visits) is mapped to the parent. Each state maintains operator weights $w_s^{(t)}$, sampled via $\pi_s^{(t)}(o)=(1-\alpha)\,w_s^{(t)}(o)/\sum w + \alpha/|\mathcal{O}|$. Scores use relative improvement $r_t=\text{clip}((\ell(f_p)-\ell(f_t))/(\ell(f_p)+\varepsilon))$, with multiplicative updates $w_s^{(t+1)}(o_t)\leftarrow w_s^{(t)}(o_t)\cdot\max(\delta,1+\eta r_t)$. A stagnation trigger $\text{Stag}_t$ is used: if the best loss improves by less than $\xi$ for $h$ rounds, exploration is forced by increasing mutate/regenerate probabilities.
    - **Design Motivation**: A minimal $2\times 2$ state machine with Multi-Armed Bandit updates learns explicit meta-policies (e.g., "high quality and frequently visited → mutative jump"), avoiding trial-and-error edits within the prompt.

2.  **Tool-Augmented Diagnostic Proposal (Diagnostic Guidance)**:
    - **Function**: Translates scalar "MSE" values into structured reports $a_t=(T_{\text{data}}(D),T_{\text{res}}(f_p,D),T_{\text{dim}}(f_p,Q))$ explaining "where and why" the error occurs.
    - **Mechanism**: Three tools are called serially: (1) **Data Profiler** statistics priors like input ranges, operator domains, variable interactions, and periodicity; (2) **Residual Diagnostic** analyzes if residuals $e_i=y_i-f_p(x_i)$ contain unfitted periodic components, missing terms, or oscillating patterns (e.g., detecting a $-0.67$ correlation with $\sin(t)$ suggests a "missing periodic term"); (3) **Dimensional Verifier** performs physical consistency checks to filter out unit-invalid combinations.
    - **Design Motivation**: Experiments show this is the most critical component—removing tools causes NMSE to jump from 4.37e-4 to 2.52e-2 (58x degradation), confirming that scalar feedback's fundamental flaw is its lack of "diagnosis."

3.  **Reflective Memory (Historical Guidance)**:
    - **Function**: Consolidates cross-round experiences into reusable natural language rules to prevent the LLM from repeating mistakes.
    - **Mechanism**: Maintains memory $M_t$, but **writes only when necessary** via trigger $\delta_t=\mathbb{I}[(t\bmod K=0)\vee(\Delta_t>\epsilon_{\text{mem}})]$ (periodic + breakthrough updates). Reflection context $C_t=(Q,M_{t-1},\text{Elite}(P_t),\text{Fail}(\mathcal{H}_t),\text{Break}(\mathcal{H}_t))$ is constructed, feeding the LLM current elites, failed edits, and breakthrough edits to extract rules. A final compression step $M_t\leftarrow\text{Compress}(M_{s-1}\cup p_\theta(\cdot\mid C_t))$ retains successful motifs and common failure patterns while removing redundancy.
    - **Design Motivation**: Pure in-context evolution is memoryless; explicit memory transforms search into an "informed scientist" process, while the trigger-based design avoids expensive LLM reflections every round.

### Loss & Training
No gradient training is involved; the entire process is inference-time. Backbones used include Llama-3.1-8B-Instruct and Qwen3-4B-Instruct ($T=0.8$). Constants are fitted using BFGS (quasi-Newton). Theoretically, the authors prove via hitting-time analysis: as long as guidance increases the single-step success rate to $p_\theta\ge p_0+\gamma$, then $\Pr(N_\theta>k)\le\exp[-k(p_0+\gamma)]$, achieving exponential sample savings for $\gamma>0$.

## Key Experimental Results

### Main Results
On LLM-SRBench (240 problems across Physics/Material/Chemistry/Biology). Baselines were given 1000 candidate budgets; DE was given only 400 (40%).

| Dataset (Qwen3-4B) | Metric | DE | Strongest Baseline (LASR) | Gain |
|---|---|---|---|---|
| LSR-Transform | NMSE ↓ | 1.15e-1 | 1.83e-1 | -37% |
| LSR-Transform | Acc0.01 ↑ | **50.45%** | 30.91% | +19.5pt |
| Physics | NMSE ↓ | **4.37e-4** | 2.51e-3 (LLM-SR) | -83% |
| Chemistry | NMSE ↓ | **1.88e-4** | 2.31e-3 | -92% |
| Stress-Strain (real) | ID/OOD NMSE ↓ | **1.11e-1 / 2.98e-1** | 1.44e-1 / 6.34e-1 | OOD -53% |

OOD generalization is significant: baselines on Physics saw NMSE soar to 8e4 and Chemistry to 5e6, while DE remained around $10^1$, indicating it recovers true structures rather than overfitted skeletons.

### Ablation Study
Physics + Qwen3-4B:

| Configuration | NMSE ↓ | Acc0.01 ↑ | Note |
|---|---|---|---|
| Full DE | **4.37e-4** | **15.91** | Complete model |
| w/o Memory | 1.34e-3 | 9.52 | Lacks history, NMSE ×3 |
| **w/o Tool** | **2.52e-2** | 4.55 | **Diagnosis is critical, NMSE ×58** |
| Fixed Refine | 8.69e-3 | 9.52 | Operator limited to single refine |
| Uniform Ops | 1.02e-2 | 6.82 | Lacks adaptive strategy |
| w/o Stagnation | 7.69e-4 | 13.64 | Lacks stagnation escape |

### Key Findings
- **Diagnostic tools are the primary contributor**: Removing tools resulted in the largest drop (58x NMSE), validating that scalar feedback lacks diagnosis.
- **Modules are complementary and non-redundant**: Removing any module degrades performance, showing direction, diagnosis, and memory are orthogonal.
- **Geometric increase in sample efficiency**: Achieved lower error with 40% of the budget, consistent with hitting-time theory and empirical curves.
- **Best operational stability**: Three independent runs on Qwen3-4B showed a variance of only 9e-10 (baselines often fluctuated above 1e-3).
- **Correct skeleton recovery**: Case studies show DE directly produces ground-truth skeletons like `sin(t)+sin(v)+v`, whereas baselines often used polynomial surrogates for missing periodic terms.

## Highlights & Insights
- **"Decoupling Proposal and Navigation" is a general framework**: This philosophy can be migrated to code generation, theorem proving, and retrosynthesis. Splitting scalar reward into direction/diagnosis/memory is a universal recipe.
- **Minimalist Meta-RL via State Machine + MAB**: A $2\times 2$ binary state space successfully learns "exploit vs. explore," which is much lighter than training an RL controller.
- **Trigger-based memory solves the "what and when to remember" problem in agents**: Periodic and breakthrough triggers prevent expensive redundant reflections while capturing key insights.
- **Theory matches experiments**: Using a simple geometric distribution hitting-time argument for exponential acceleration is more convincing than complex regret bounds.

## Limitations & Future Work
- Toolsets are manually designed (data profiler, residual, dimensional); adapting to non-physical domains (e.g., code generation) requires new diagnostic tools without an automated solution yet.
- The operator set is fixed to four types, and the state space is only $2\times 2$. This granularity may be insufficient for more complex domains (e.g., multi-objective or constrained SR).
- Experiments were conducted on 4B–8B open-source models; verification on stronger models like GPT-4 or Claude is needed to see if the guidance gain $\gamma$ diminishes as the backbone improves.
- BFGS fitting is sensitive to initial values; a correct skeleton might still be misjudged if constants are poorly fitted—a common bottleneck in all LLM-SR.

## Related Work & Insights
- **vs LLM-SR (Shojaee et al., 2025a)**: Both use LLM + Evolution + BFGS, but LLM-SR only provides scalar MSE. DE translates MSE into "where it's wrong," reducing Physics NMSE from 2.51e-3 to 4.37e-4 with 40% budget.
- **vs LASR (Grayeli et al., 2024)**: LASR relies on $10^5$ non-LLM mutations for exploration; DE uses structured guidance for the same effect, proving "smarter prompts" outperform "brute-force search."
- **vs Universal LLM Agents (ReAct/Reflexion)**: These frameworks often reflect at every step; DE uses triggers to control reflection frequency, making it suitable for long-horizon optimization without LLM call explosion.

## Rating
- Novelty: ⭐⭐⭐⭐ Systematic decoupling in the LLM-SR context is novel, though the individual components are established.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Includes OOD, noise, real-world data, multiple backbones, ablation, and case studies.
- Writing Quality: ⭐⭐⭐⭐⭐ Excellent figures and clear correspondence between motivation and design.
- Value: ⭐⭐⭐⭐ A strong baseline for the LLM-SR community; design patterns are applicable to other LLM-as-optimizer tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Lifting Traces to Logic: Programmatic Skill Induction with Neuro-Symbolic Learning for Long-Horizon Agentic Tasks](lifting_traces_to_logic_programmatic_skill_induction_with_neuro-symbolic_learnin.md)
- [\[ICML 2026\] ResRL: Boosting LLM Reasoning via Negative Sample Projection Residual Reinforcement Learning](resrl_boosting_llm_reasoning_via_negative_sample_projection_residual_reinforceme.md)
- [\[ICML 2026\] Lookahead Sample Reward Guidance for Test-Time Scaling of Diffusion Models](lookahead_sample_reward_guidance_for_test-time_scaling_of_diffusion_models.md)
- [\[ICML 2026\] Efficient Reasoning with Hidden Thinking](efficient_reasoning_with_hidden_thinking.md)
- [\[ICML 2026\] Evaluating Relational Reasoning in LLMs with REL](evaluating_relational_reasoning_in_llms_with_rel.md)

</div>

<!-- RELATED:END -->
