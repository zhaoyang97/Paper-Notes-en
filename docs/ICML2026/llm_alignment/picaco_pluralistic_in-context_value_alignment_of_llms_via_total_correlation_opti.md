---
title: >-
  [Paper Note] PICACO: Pluralistic In-Context Value Alignment of LLMs via Total Correlation Optimization
description: >-
  [ICML 2026][LLM Alignment][Pluralistic value alignment] PICACO formalizes the task of "enabling an LLM to simultaneously adhere to multiple…
tags:
  - "ICML 2026"
  - "LLM Alignment"
  - "Pluralistic value alignment"
  - "In-Context Alignment (ICA)"
  - "Total Correlation (TC)"
  - "Meta-instruction optimization"
  - "Black-box optimization"
date: 2026-05-08
content_hash: 222a910d3f618d5f
---

# PICACO: Pluralistic In-Context Value Alignment of LLMs via Total Correlation Optimization

**Conference**: ICML 2026  
**arXiv**: [2507.16679](https://arxiv.org/abs/2507.16679)  
**Code**: https://github.com/Salomeeeee/PICACO  
**Area**: RLHF Alignment / Value Alignment / In-Context Alignment  
**Keywords**: Pluralistic value alignment, In-Context Alignment (ICA), Total Correlation (TC), Meta-instruction optimization, Black-box optimization

## TL;DR
PICACO formalizes the task of "enabling an LLM to simultaneously adhere to multiple, potentially conflicting human values within a single prompt" as maximizing the conditional Total Correlation (TC) between a value set and responses. Without modifying model parameters, it utilizes an EM-like two-step iterative "response enhancement + instruction refinement" process to automatically search for a meta-instruction. This allows GPT-3.5, LLaMA-3.1-8B, and Gemini-1.5-Flash to exceed strong baselines like OPRO and Modular Pluralism across five value sets containing up to eight combined values (e.g., Schwartz, HH).

## Background & Motivation

**Background**: Compared to the high costs of RLHF or SFT which modify model parameters, **In-Context Alignment (ICA)** directly embeds value descriptions and examples into the prompt during inference. This leverages the LLM's existing knowledge for alignment, offering flexibility, cost-effectiveness, and the ability to switch preferences in real-time. Consequently, it has emerged as a new branch of alignment research (e.g., URIAL, OPRO, Modular Pluralism, CICL).

**Limitations of Prior Work**: Human values are inherently pluralistic and often conflict (e.g., helpful vs. harmless, stimulation vs. tradition). However, existing ICA methods often suffer when multiple values are placed in a single prompt, as the LLM frequently **focuses on only one or two while silently ignoring the others**—a phenomenon the authors term the **Instruction Bottleneck** (Fig. 1 shows GPT-4o reflecting only a subset of values when required to follow multiple Schwartz values simultaneously).

**Key Challenge**: The process by which an LLM understands a prompt is "agnostic." The relative weight and suppression among various values written in a prompt are determined internally by the LLM, which is neither visible nor controllable. Persona-based methods (URIAL), persona-centric approaches (MP), or multi-community models (Modular Pluralism) either require heavy manual effort, rely on pre-defined value sets, or handle only a few values, failing to "explicitly" regulate the relationships between multiple values.

**Goal**: To automatically search for a meta-instruction capable of simultaneously carrying $K$ values without fine-tuning, heavy annotation, or fixed value sets, ensuring that the response strongly fits each $v_k$ without introducing redundant verbiage unrelated to $v_k$.

**Key Insight**: The authors adopt **Total Correlation** from information theory: $\text{TC}(\bm{V},\bm{y})=\sum_k I(\bm{v}_k;\bm{y}) - I(\bm{V};\bm{y})$. This metric rewards the mutual information between each single value and the response while penalizing redundant overlap between the value set and the response, aligning 1:1 with the requirements of "multi-value balance." By treating the LLM as a black box and the meta-instruction $\bm{e}$ as the only optimizable variable, the problem is transformed into **Black-Box Optimization**.

**Core Idea**: An estimable lower bound for $\text{TC}_{\bm{e}}(\bm{V},\bm{y} | \bm{x})$ is derived, followed by an EM-like two-step iteration: one step enhances a "high TC response pool," and the other selects the next version of the meta-instruction that maximizes TC over that pool. This **transforms "multi-value balance" from a prompt engineering challenge into an optimization problem with an explicit objective function.**

## Method

### Overall Architecture
**Input**: A set of task prompts $\mathcal{X}=\{\bm{x}_i\}_{i=1}^N$, a target LLM $p$, a combination of values $\bm{V}=\{\bm{v}_k\}_{k=1}^K$, textual observations $\bm{s}$ (few-shot examples reflecting these values), and a seed meta-instruction $\bm{e}^0$. **Output**: An optimized meta-instruction $\bm{e}^T$ after $T$ iterations, which can be prepended to task prompts during inference to align $p$ with all values in $\bm{V}$.

The entire pipeline is an EM-like loop (Alg.1): it maintains two "response pools" for each $\bm{x}_i$—an **aligned pool** $\bm{R}^a_i$ (sampled from $p_{\bm{e}}$ and selected by top $M_1$ TC scores) and a **noisy pool** $\bm{R}^n_i$ (sampled from the base LLM $p$, selecting $M_2$ "negative examples" that easily parrot $\bm{s}$). "Response enhancement" and "instruction refinement" are executed alternately in each round until $\bm{e}^t$ converges.

### Key Designs

1.  **TC Objective + Estimable Lower Bound**:
    - **Function**: Formalizes "multi-value simultaneous alignment" as an optimization problem with an explicit objective, rather than trial-and-error prompt engineering.
    - **Mechanism**: The conditional TC $\text{TC}_{\bm{e}}(\bm{V},\bm{y}|\bm{x})=\sum_{k=1}^K I_{\bm{e}}(\bm{v}_k;\bm{y}|\bm{x})-I_{\bm{e}}(\bm{V};\bm{y}|\bm{x})$ is used as the optimization target. A Barber-Agakov lower bound is applied to the first term, and a conditional version of the CLUB upper bound is applied to the second. Combined, this yields $\text{TC}_{\bm{e}} \geq \mathbb{E}_{\hat p(\bm{x})}\{\beta\sum_k \mathbb{E}_{p_{\bm{e}}}[\log q_{\bm{\omega}}(\bm{v}_k|\bm{x},\bm{y})] - \mathbb{E}_{p_{\bm{e}}}[\log q_{\bm{\phi}}(\bm{s}|\bm{x},\bm{y})] + \mathbb{E}_{p}[\log q_{\bm{\phi}}(\bm{s}|\bm{x},\bm{y})]\}$, where $q_{\bm{\omega}}$ is an LLM-as-judge acting as a value evaluator, and $q_{\bm{\phi}}$ uses cosine similarity between $\bm{s}, \bm{x}, \bm{y}$ to characterize "how much of the response is shallowly parroted from $\bm{s}$."
    - **Design Motivation**: By explicitly splitting "value conformity" and "instruction redundancy," the model **uses one term to reward multi-value conformity and another to penalize shallow imitation/repetition**. This theoretically addresses the "superficial alignment" problem mentioned by Zhou et al. 2023, serving as the source of the model's ability to fit values without plagiarism.

2.  **EM-like Two-step Iteration: Response Enhancement + Instruction Refinement**:
    - **Function**: Converts the lower bound into an executable algorithm where LLM parameters remain frozen and only $\bm{e}$ is optimized.
    - **Mechanism**: In round $t$: (a) **Response Enhancement Step**: Fix $\bm{e}^{t-1}$, sample responses $\{\bm{y}_{i,j}\}$ from $p_{\bm{e}^{t-1}}$, calculate $q_{\bm{\omega}}$ and $q_{\bm{\phi}}$, and add the top-$M_1$ responses with the highest $\log q_{\bm{\omega}}(\bm{v}_k|\cdot)-\log q_{\bm{\phi}}(\bm{s}|\cdot)$ to the aligned pool. Simultaneously, sample from base $p$ and pick $M_2$ samples with high $q_{\bm{\phi}}$ for the noisy pool. (b) **Instruction Refinement Step**: Fix pools and select $\bm{e}^t=\arg\max_{\bm{e}}\frac{1}{N}\sum_i\{\sum_{j=1}^{M_1}[\sum_k\log\frac{q_{\bm{\omega}}^{\beta}}{q_{\bm{\phi}}^{1/K}}]p_{\bm{e}}(\bm{y}_{i,j}^t|\bm{x}_i)+\sum_{j=1}^{M_2}p(\hat{\bm{y}}_{i,j}|\bm{x}_i)\log q_{\bm{\phi}}\}$ to maximize the probability of "high TC responses" while suppressing "repetitive noise responses." Practically, a set of candidate $\{\bm{e}_k\}$ is sampled and scored to find the optimum.
    - **Design Motivation**: Frozen LLM parameters and a discrete prompt space preclude gradient backpropagation. EM-like alternating optimization (similar to Sun et al. 2022) is the natural choice for balancing executability and theoretical interpretability. **Accumulating pools across rounds** rather than resampling reduces expensive LLM calls (N=50, $M_1=10, M_2=15, T=10$).

3.  **Aligned + Noisy Response Pools (Redundancy Regularization)**:
    - **Function**: Prevents optimization from collapsing into "fake alignment," such as merely repeating keywords from $\bm{s}$.
    - **Mechanism**: The third term $\mathbb{E}_p[\log q_{\bm{\phi}}(\bm{s}|\bm{x},\bm{y})]$ is sampled from the base $p$ (without meta-instruction), acting as a "baseline repetition probability." The objective uses $-\log q_{\bm{\phi}}+\log q_{\bm{\phi}}^{\text{baseline}}$ as a "relative redundancy" term—only penalizing $\bm{e}$ if it causes **more repetition of $\bm{s}$ than the base model**, avoiding mindless suppression of all similarity.
    - **Design Motivation**: The authors note that Schwartz values are prone to fake alignment, where responses mention value names without substance. The noisy pool acts as a "reference distribution," forcing the model to learn values while maintaining content relevance.

### Loss & Training
No parameter training is performed. The optimizer uses the LLM itself as a prompt sampler. In the main experiments, the same LLM as the target model samples candidate $\bm{e}_k$. $q_{\bm{\omega}}$ is GPT-4o-mini; $q_{\bm{\phi}}$ is the cosine similarity between $\bm{s}, \bm{x}, \bm{y}$. Hyperparameters include $N=50, M_1=10, M_2=15, T=10$, with $\beta$ controlling the trade-off. LLM calls scale linearly with $T$.

## Key Experimental Results

### Main Results
Five value combinations (Helpfulness-4, Harmlessness-4, HH Balance-8, Confucianism-4, Modern Liberalism-4) were tested on three LLMs (GPT-3.5-Turbo, LLaMA-3.1-8B-Instruct, Gemini-1.5-Flash) using GPT-4o as a judge. Friedman tests result in $p<10^{-4}$, indicating significant statistical differences.

| Target Model | Value Set | PICACO | OPRO | URIAL | Modular | Q+IF |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| GPT-3.5-Turbo | Confucianism-4 | **3.788** | 3.713 | 3.622 | 3.567 | 3.306 |
| GPT-3.5-Turbo | Liberalism-4 | **3.135** | 2.961 | 3.030 | 3.036 | 2.728 |
| GPT-3.5-Turbo | HH Balance-8 | 4.257 | **4.286** | 4.097 | 4.245 | 4.082 |
| GPT-3.5-Turbo | Helpfulness-4 | **4.287** | **4.287** | 4.164 | 4.236 | 4.247 |
| LLaMA-3.1-8B | Confucianism-4 | **3.471** | 3.437 | 3.530 | 3.427 | 3.164 |
| LLaMA-3.1-8B | HH Balance-8 | 4.110 | **4.114** | 4.085 | 3.793 | 3.977 |

Findings: PICACO is most stable on Schwartz categories (Confucianism/Liberalism). On HH-8, it is competitive with OPRO, with its **advantage expanding as the number of values increases**.

### Ablation Study
| Configuration | Avg. Drop across Compositions | Description |
| :--- | :--- | :--- |
| Full PICACO | — | Complete method |
| w/o redundancy terms $q_{\bm{\phi}}$ | Significant | Absence of $q_{\bm{\phi}}$ causes fake alignment / repetition of $\bm{s}$. |
| w/o EM iteration (1 round) | Significant | Degenerates into single-step prompt optimization, loses pool accumulation. |
| w/o Response Enhancement | Significant | Lacks high TC response pool; refinement lacks supervision. |
| w/o Instruction Refinement | Significant | Equivalent to sampling under fixed $\bm{e}^0$; cannot find better instructions. |
| w/o noisy pool $M_2$ | Significant | Redundancy term lacks baseline; prone to over-suppressing similarity. |

### Key Findings
- **The more values, the greater PICACO's advantage** (Fig. 3): As HH values increase from 2 to 8, PICACO's delta over Q remains largest, and the coefficient of variation across values is smallest, showing TC optimization naturally balances attention across all values.
- **Smaller models are more sensitive to ICA methods**: Nearly half of the baselines performed worse than the base model on LLaMA-3.1-8B, whereas PICACO provided stable improvements.
- **PICACO + GPT-3.5-Turbo > Base O4-Mini** (Fig. 4(b)): Cheaper models using PICACO can approach the alignment levels of expensive models.
- **Jailbreak Resistance** (Fig. 4(c)): Under jailbreak templates, PICACO significantly reduces toxic responses compared to OPRO/Modular while maintaining higher helpfulness. This is attributed to the aligned pool accumulating "helpful refusal" patterns and $q_{\bm{\phi}}$ penalizing superficial safety clichés.
- **Outperformance on OOD Tasks** (Table 2): On Value Portrait (new tasks like creative writing), PICACO remains first in Confucianism/Liberalism, suggesting the meta-instruction is cross-task reusable.

## Highlights & Insights
- **Standardized Objective Function**: Total Correlation formalizes both "reflecting every value" and "avoiding redundancy" simultaneously. This elegant alignment between problem statement and optimization target is a major strength.
- **Redundancy Term as an Antidote to Fake Alignment**: Using $q_{\bm{\phi}}$ and a noisy pool to detect repetition of $\bm{s}$ is a rare explicit mechanism against "superficial alignment" in prompt optimization.
- **EM-like + Dual Pool Accumulation**: Spreading high sampling costs over multiple rounds and selecting from historical pools makes black-box optimization practical.

## Limitations & Future Work
- PICACO primarily targets the **Instruction Bottleneck** (failure to receive values) rather than **hard value conflicts**. For the latter, it encourages "integrative compromises" without explicit priority weights.
- Empirical limitations: $q_{\bm{\omega}}$ relies on an LLM-as-judge. Despite cross-validation with multiple judges and human evaluation, "empty high-score" cases still occur.
- Cosine similarity $q_{\bm{\phi}}$ is a proxy and may fail to capture semantic repetition through paraphrasing.
- The cost of optimization (linear with iterations) remains high for agents requiring per-query custom instructions.

## Related Work & Insights
- **vs OPRO**: PICACO replaces raw evaluation scores with a TC lower bound and explicit redundancy terms, outperforming OPRO in high-value-count scenarios.
- **vs URIAL**: PICACO requires no manually written meta-instructions and shows better transferability on OOD tasks.
- **vs Modular Pluralism**: No auxiliary fine-tuned models are needed, and PICACO demonstrates significantly better balance as the number of values scales.
- **Insight**: The TC optimization objective can be generalized to multi-task prompt optimization, RAG systems (reflecting multiple documents), and Agent systems (satisfying multiple intents).

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Total Correlation is applied to ICA for the first time, elegantly matching pluralistic balance and anti-fake-alignment intuitions.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extensive coverage across 5 value sets, 3 LLMs, 9 baselines, OOD, jailbreaks, and human evaluations.
- Writing Quality: ⭐⭐⭐⭐ Solid mathematical derivation, though the nested findings in Section 4.2 are slightly dense.
- Value: ⭐⭐⭐⭐⭐ Training-free, black-box compatible, and supports hot-swappable value combinations, offering significant industrial utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Toward Stable Value Alignment: Introducing Independent Modules for Consistent Value Guidance](toward_stable_value_alignment_introducing_independent_modules_for_consistent_val.md)
- [\[ICML 2026\] Quantifying the Salience of Geo-Cultural Values for Pluralistic Safety Alignment](quantifying_the_salience_of_geo-cultural_values_for_pluralistic_safety_alignment.md)
- [\[ICML 2026\] Towards Context-Invariant Safety Alignment for Large Language Models](towards_context-invariant_safety_alignment_for_large_language_models.md)
- [\[ICML 2026\] The Realignment Problem: When Right becomes Wrong in LLMs](the_realignment_problem_when_right_becomes_wrong_in_llms.md)
- [\[ACL 2026\] How Value Induction Reshapes LLM Behaviour](../../ACL2026/llm_alignment/how_value_induction_reshapes_llm_behaviour.md)

</div>

<!-- RELATED:END -->
