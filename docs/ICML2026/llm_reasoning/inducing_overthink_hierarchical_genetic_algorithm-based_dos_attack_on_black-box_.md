---
title: >-
  [Paper Note] Inducing Overthink: Hierarchical Genetic Algorithm-based DoS Attack on Black-Box Large Language Reasoning Models
description: >-
  [ICML 2026][LLM Reasoning][Overthinking] Addressing the vulnerability of Large Reasoning Models (LRMs) being triggered into "overthinking" by "logically incomplete inputs…
tags:
  - "ICML 2026"
  - "LLM Reasoning"
  - "Overthinking"
  - "DoS Attack"
  - "Genetic Algorithm"
  - "Reasoning Models"
  - "Black-box Attack"
date: 2026-05-08
content_hash: 0414ba4d11f8064c
---

# Inducing Overthink: Hierarchical Genetic Algorithm-based DoS Attack on Black-Box Large Language Reasoning Models

**Conference**: ICML 2026  
**arXiv**: [2605.13338](https://arxiv.org/abs/2605.13338)  
**Code**: None  
**Area**: LLM Reasoning / AI Security / Adversarial Attack  
**Keywords**: Overthinking, DoS Attack, Genetic Algorithm, Reasoning Models, Black-box Attack

## TL;DR
Addressing the vulnerability of Large Reasoning Models (LRMs) being triggered into "overthinking" by "logically incomplete inputs," this paper proposes a Hierarchical Genetic Algorithm (HGA). Under pure black-box conditions, it treats structurally decomposed problems as genes. Through sentence-level/question-level crossover and addition/deletion mutations, it searches for adversarial samples with logical fractures. This method achieves up to a 26.1x increase in response length on MATH, facilitating low-cost DoS attacks.

## Background & Motivation

**Background**: Reasoning models such as DeepSeek-R1, GPT-o3, and Qwen3-Thinking have been widely deployed. The number of tokens directly determines reasoning latency and energy costs (Gao et al. 2024). Simultaneously, the community has observed the "overthinking" phenomenon: when faced with missing premises or logical fractures, reasoning models repeatedly self-reflect and generate excessively long CoTs (Chen 2024, Fan 2025).

**Limitations of Prior Work**: (i) Existing DoS / energy-latency attacks (GCG, Engorgio) mostly rely on white-box gradients, making them inapplicable to commercial closed-source APIs; (ii) Black-box AutoDoS-like methods rely on inflating prompt length to stack tokens, insufficiently exploiting the "thinking" mechanism of reasoning models; (iii) The Missing-Premise (MIP) dataset by Fan et al. relies on manual construction, lacking automation and adversarial optimization; (iv) OverThink attacks rely on manually designed decoy tasks, offering narrow coverage.

**Key Challenge**: Methods capable of exploiting the specificities of reasoning chains rely on manual efforts, while automated methods fail to exploit the reasoning chain itself—no current method simultaneously satisfies being black-box, reasoning-aware, and automatable.

**Goal**: (i) Design an automated attack that amplifies LRM reasoning tokens under pure black-box, text-only interfaces; (ii) The attack signal should directly hit "overthinking" behavior rather than just "longer output"; (iii) Adversarial samples should be transferable across models, shifting search costs from expensive commercial APIs to open-source proxy models.

**Key Insight**: Treat the problem as a structured gene composed of a "set of premises + final question" rather than an indivisible string. This allows for crossover and mutation at the "premise-question" pairing level, generating semantically readable but logically fractured adversarial problems. This specifically targets the LRM weakness where missing or inconsistent premises trigger frantic, repetitive reflection.

**Core Idea**: Utilize a hierarchical genetic algorithm to search the (premise set, question) space, combined with a composite fitness function of "length + reflection markers," to automatically evolve problems that induce overthinking.

## Method

### Overall Architecture
For each problem $x$, a LRM is first used to decompose it into a structured $(\mathbf{P}, q)$ (list of premises + final question). A population is initialized with $N=10$ individuals. Each generation evaluates fitness (by running the target LRM and counting output tokens and reflection marker frequency). Parents are then selected via elite + roulette wheel mechanisms to generate offspring using four operations: question-level crossover, premise-level crossover, premise deletion, and premise addition. After $G=5$ generations, the individual with the highest fitness is retained as the adversarial sample. The entire attack is black-box, relying solely on API text inputs and outputs.

### Key Designs

1.  **HGA representation**:
    - **Function**: Decomposes opaque problem statements into manipulatable "premise sets + questions," allowing genetic operators to perturb logical structures rather than characters.
    - **Mechanism**: Each individual is represented as $x = (\mathbf{P}, q)$, where $\mathbf{P} = [p_1, \ldots, p_n]$ are premises and $q$ is the final question. The search space is $\mathcal{X} = \{(\mathbf{P}, q)\mid \mathbf{P}\subseteq\mathcal{S}, q\in\mathcal{S}\}$. The initial population is obtained by decomposing dataset problems using Qwen3-Thinking.
    - **Design Motivation**: Previous white-box methods like GCG added perturbations at the character/token level, which often destroyed readability and failed to hit reasoning logic. Breaking down problems into "logical building blocks" allows crossover and mutation to maintain semantic readability while breaking the causal chain between premises and questions—the key to triggering overthinking.

2.  **Composite Fitness: Length + Reflection Markers**:
    - **Function**: Aligns the search target with "inducing overthinking" rather than "simple token stacking."
    - **Mechanism**: Define $\text{score}_1(x) = |R(x)|$ (CoT token count) to quantify length; $\text{score}_2(x) = \sum_{w\in\mathcal{V}}\text{Count}(w, R(x))$ to count reflection markers such as "Wait/But/However/Hmm." These are linearly combined after intra-generation z-score normalization: $f(x) = \alpha \cdot \hat{\text{score}}_1 + (1-\alpha)\cdot\hat{\text{score}}_2$. The authors found that $\alpha=0.5\sim 0.7$ significantly outperforms pure length ($\alpha=1$) or pure reflection ($\alpha=0$).
    - **Design Motivation**: Optimizing solely for length leads to local optima centered on "padding." Introducing reflection markers as a secondary objective guides the search toward problems that genuinely cause the model to enter self-doubt loops, enhancing cross-model transferability (e.g., $\alpha=0.7$ remains superior to $\alpha=1.0$ when transferred to DeepSeek-R1).

3.  **Hierarchical Genetic Operators (Question-level + Premise-level)**:
    - **Function**: Disrupts logical structures at different granularities to create samples where "premises do not match the question" or "irrelevant premises are inserted."
    - **Mechanism**: Crossover is performed with probability $p_c$: question-level crossover swaps $q$ between two parents (mismatching premises and questions), while premise-level crossover swaps a single premise. Mutation is performed with probability $p_m$: deleting a premise (breaking the reasoning chain) or adding a premise from another individual (injecting irrelevant conditions).
    - **Design Motivation**: Question-level crossover targets "premise-question alignment," while premise-level operations create internal contradictions. Together, they cover logical fracture modes of "question mismatch" and "premise conflict." Lack of semantic naturalness is actually an advantage—the more confused the LRM becomes, the more it resorts to repetitive reasoning.

### Loss & Training
There are no trainable parameters; the process is a gradient-free black-box search: population $N=10$, evolution $G=5$ generations, $p_c=0.8$, $p_m=0.2$. Each individual in each generation calls the API once to evaluate fitness, resulting in a total budget of approximately 60 queries per model. The final output is the individual with the highest fitness.

## Key Experimental Results

### Main Results

| Dataset | Model | BASE Avg-len | MIP Avg-len | HGA Avg-len | Max-len Gain |
| :--- | :--- | :--- | :--- | :--- | :--- |
| SVAMP | Qwen3-Thinking | 634 | 2231 | 5447 | 7906 (6.9×) |
| SVAMP | GPT-o3 | 239 | 620 | 3346 | 6562 (8.5×) |
| GSM8K | DeepSeek-R1 | 343 | 3093 | 4121 | 9068 (18.1×) |
| MATH | Qwen3-Thinking | 3618 | 7184 | 13007 | 22303 (2.5×) |
| MATH | Gemini-2.5-Flash | 2889 | 8043 | 12147 | 18011 (2.7×) |

HGA consistently outperforms BASE and MIP; the maximum amplification factor on MATH reaches 26.1× (summarized in Table 2 of the paper).

### Ablation Study

| Setting | Key Metric | Description |
| :--- | :--- | :--- |
| $\alpha=1.0$ (Pure length) | MATH Max-len 14132 / Avg 6258 | Degenerates into simple character stacking |
| $\alpha=0.5$ (Balanced) | MATH Max-len 32019 / Avg 16826 | Objectives are complementary; optimal performance |
| $\alpha=0.7$ | MATH Max-len 26576 / Avg 18315 | High reflection weight; higher average length |
| Transfer $\alpha=0.7$→DeepSeek-R1 | Max 31998 / Avg 10893 | Composite fitness stays robust across models |
| Qwen3-14B Proxy → GPT-o3 | Avg Gain 7.1× | Prompts evolved on small models strike commercial APIs |

### Key Findings
- **Intermediate $\alpha$ is Optimal**: A pure length objective traps the search in "local verbosity" extrema. Adding reflection markers acts as a compass, guiding the search toward samples that induce real self-doubt loops. This holds true when transferring to DeepSeek-R1.
- **High Efficiency**: On MATH, HGA uses only 99 input tokens to push DeepSeek-R1's output to the 32768 limit, whereas AutoDoS requires 2652 input tokens to reach 16009. This suggests the attack power stems from logical perturbation rather than character stacking.
- **Strong Transferability**: Using the open-source Qwen3-14B as a proxy for fitness evaluation, the evolved prompts amplify GPT-o3 by an average of 7.1× and Qwen3-Thinking by 8.1×. This indicates that "overthinking" is a common cross-architecture weakness of LRMs.
- **Diminishing Returns on Search Budget**: Gains from increasing population and generations from 5 to 30 saturate quickly. Strong adversarial samples can be found with small budgets, making the attack cost extremely low.

## Highlights & Insights
- **"Logical Incompleteness" as a Black-box Attack Signal**: Shifting perturbations from the character level to the logic level transforms the search space from token vocabularies to "stitchable sub-clauses." This significantly enhances readability, scalability, and transferability, offering a paradigm shift for future prompt-level adversarial attacks.
- **Reflection Markers as Proxy Rewards**: Directly using the frequency of "Wait/But/Hmm" as an observable proxy for "thinking intensity" bypasses the difficulty of estimating internal computation in black-box settings. The authors suggest this could even be integrated into white-box loss functions.
- **Proxy Model Transfer Attack**: Outsourcing expensive fitness evaluations to an open-source 14B model reduces search costs by orders of magnitude. The ability to "cold-start" attacks against commercial APIs is a significant warning for defenders.
- **Linking Overthinking to DoS Attacks**: Redefining an "efficiency issue" as a "security issue" opens a new research direction.

## Limitations & Future Work
- The attack still requires repeated queries to the victim model; 60 evaluations per problem may be identified by APIs with high QPS limits or rate detection.
- The dictionary of reflection markers is manually selected and may fail for models thinking in other natural languages (Chinese, code).
- Validation was restricted to mathematical reasoning benchmarks (GSM8K / SVAMP / MATH); transferability to code generation or multi-turn agents remains untested.
- Defense discussions are limited, suggesting only general "behavior monitoring + robust training" without specific baselines.
- The attack primarily lengthens the reasoning chain but does not force incorrect answers; defenders using early-stop or answer caching might bypass it.

## Related Work & Insights
- **vs GCG (Zou 2023)**: GCG is white-box and performs token suffix optimization; this work is black-box and performs logical structure perturbation. Table 1 shows this is the only method satisfying "black-box + reasoning-aware + automated."
- **vs AutoDoS / Crabs (Zhang 2025b)**: AutoDoS constructs "DoS Attack Trees" to inflate prompt length, with input tokens 20x+ larger than HGA. HGA creates structural perturbations, yielding fewer inputs but longer outputs, making it more covert and efficient.
- **vs OverThink (Kumar 2025)**: OverThink relies on manually inserted decoy tasks, whereas this work uses GA for automated evolution, providing a scalability advantage.
- **vs Deadlock (Zhang 2025a)**: Deadlock forces "Wait/But" loops via token injection. HGA does not force tokens but relies on logical flaws within the problem itself to make the model spontaneously loop, making it harder to detect via heuristic filters.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Combining "logical structural genes + reflection marker fitness" is a fresh perspective, and cross-model transfer results are impactful.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers 4 SOTA reasoning models across 3 datasets and compares against 3 baseline types, with extensive ablations on $\alpha$, proxy transfer, and search budget.
- **Writing Quality**: ⭐⭐⭐⭐ The three-pronged motivation (scope/quality/automation) and the comparison in Table 1 clarify the positioning. The Method section is formula-dense but well-layered.
- **Value**: ⭐⭐⭐⭐ Reveals a universal and easily exploitable weakness in LRMs, providing direct impetus for designing rate limiting, cost caps, and thinking length limits on the deployment side.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] DecepChain: Inducing Deceptive Reasoning in Large Language Models](decepchain_inducing_deceptive_reasoning_in_large_language_models.md)
- [\[ACL 2026\] Merlin's Whisper: Enabling Efficient Reasoning in Large Language Models via Black-box Persuasive Prompting](../../ACL2026/llm_reasoning/merlin39s_whisper_enabling_efficient_reasoning_in_large_language_models_via_blac.md)
- [\[ICML 2026\] ANCHOR: Abductive Network Construction with Hierarchical Orchestration for Reliable Probability Inference in Large Language Models](anchor_abductive_network_construction_with_hierarchical_orchestration_for_reliab.md)
- [\[ICML 2026\] Modeling Hierarchical Thinking in Large Reasoning Models](modeling_hierarchical_thinking_in_large_reasoning_models.md)
- [\[ICML 2026\] Diagnosing Multi-step Reasoning Failures in Black-box LLMs via Stepwise Confidence Attribution](diagnosing_multi-step_reasoning_failures_in_black-box_llms_via_stepwise_confiden.md)

</div>

<!-- RELATED:END -->
