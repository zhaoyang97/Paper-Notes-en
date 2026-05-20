---
title: >-
  [Paper Note] Inducing Overthink: Hierarchical Genetic Algorithm-based DoS Attack on Black-Box Large Language Reasoning Models
description: >-
  [ICML 2026][LLM Safety][Overthinking] This paper targets the vulnerability of large reasoning models (LRMs) to "overthinking" triggered by logically deficient inputs. It proposes a hierarchical genetic algorithm (HGA) th…
tags:
  - "ICML 2026"
  - "LLM Safety"
  - "Overthinking"
  - "DoS Attack"
  - "Genetic Algorithm"
  - "Reasoning Model"
  - "Black-box Attack"
date: 2026-05-08
content_hash: 0dc77ec96a2a96e9
---

# Inducing Overthink: Hierarchical Genetic Algorithm-based DoS Attack on Black-Box Large Language Reasoning Models

**Conference**: ICML 2026  
**arXiv**: [2605.13338](https://arxiv.org/abs/2605.13338)  
**Code**: None  
**Area**: LLM Reasoning / AI Security / Adversarial Attacks  
**Keywords**: Overthinking, DoS Attack, Genetic Algorithm, Reasoning Model, Black-box Attack

## TL;DR
This paper targets the vulnerability of large reasoning models (LRMs) to "overthinking" triggered by logically deficient inputs. It proposes a hierarchical genetic algorithm (HGA) that, under pure black-box conditions, treats structurally decomposed questions as genes. Through sentence-level and question-level crossover and mutation (addition/deletion), it searches for adversarial samples with logical breaks. On MATH, it can amplify response length by up to 26.1×, enabling low-cost DoS attacks.

## Background & Motivation

**Background**: Reasoning models such as DeepSeek-R1, GPT-o3, and Qwen3-Thinking are widely deployed, with token count directly determining inference latency and energy cost (Gao et al. 2024). The community has also observed "overthinking" in reasoning models: when faced with missing premises or logical breaks, models repeatedly self-reflect and generate excessively long CoT (Chen 2024, Fan 2025).

**Limitations of Prior Work**: (i) Existing DoS/energy-latency attacks (GCG, Engorgio) mostly rely on white-box gradients, making them unusable for commercial closed-source APIs; (ii) Black-box methods like AutoDoS extend prompt length to stack tokens, but do not exploit the "thinking" mechanism of reasoning models; (iii) Fan et al.'s Missing-Premise (MIP) dataset is manually constructed, lacking automation and adversarial optimization; (iv) OverThink attacks rely on manually designed decoy tasks, with limited coverage.

**Key Challenge**: Methods that leverage reasoning chains are manual; automated methods do not exploit reasoning chains. No existing approach simultaneously achieves black-box, reasoning-awareness, and automation.

**Goal**: (i) Design an attack under pure black-box, text-only interfaces that can automatically amplify LRM reasoning tokens; (ii) The attack signal should directly target "overthinking" behavior, not just "longer output"; (iii) The adversarial samples should be transferable across models, shifting search cost from expensive commercial APIs to open-source proxy models.

**Key Insight**: Treat questions as a structured gene of "premise set + final question" rather than as indivisible strings. This enables crossover and mutation at the "premise-question" pairing level, generating semantically readable but logically broken adversarial questions, directly exploiting the LRM weakness of "overthinking" when faced with incomplete/inconsistent premises.

**Core Idea**: Use a hierarchical genetic algorithm to search in the (premise set, question) space, with a composite fitness of "length + reflection markers," to automatically evolve questions that induce overthinking.

## Method

### Overall Architecture
For each question $x$, the LRM first decomposes it into structured $(\mathbf{P}, q)$ (premise list + final question). A population of $N=10$ individuals is initialized. Each generation, fitness is evaluated (the victim LRM is run to measure output token count and frequency of reflection markers). Elite and roulette selection are used to choose parents, and four operations—question-level crossover, premise-level crossover, premise deletion, and premise addition—are used to generate offspring. The process runs for $G=5$ generations, and the individual with the highest fitness is retained as the adversarial sample. The attack is fully black-box, relying only on API text input/output.

### Key Designs

1. **Structured Gene Representation (HGA representation)**:

    - **Function**: Decomposes opaque questions into "premise set + question" for genetic operations at the logical structure level rather than the character level.
    - **Mechanism**: Each individual is represented as $x = (\mathbf{P}, q)$, where $\mathbf{P} = [p_1, \ldots, p_n]$ are premises and $q$ is the final question. The search space is $\mathcal{X} = \{(\mathbf{P}, q)\mid \mathbf{P}\subseteq\mathcal{S}, q\in\mathcal{S}\}$. The initial population is obtained by decomposing dataset questions using Qwen3-Thinking.
    - **Design Motivation**: Previous white-box methods like GCG perturb at the character/token level, which harms readability and does not target reasoning logic. Decomposing questions into "logical blocks" allows crossover/mutation to maintain semantic readability while breaking the causal chain between premises and question, which is key to triggering overthinking.

2. **Composite Fitness: Length + Reflection Markers**:

    - **Function**: Aligns the search objective with "inducing overthinking" rather than simply "stacking tokens."
    - **Mechanism**: Define $\text{score}_1(x) = |R(x)|$ (CoT token count) to quantify length; $\text{score}_2(x) = \sum_{w\in\mathcal{V}}\text{Count}(w, R(x))$ counts occurrences of reflection markers such as "Wait/But/However/Hmm." Both are z-score normalized within the generation and linearly combined: $f(x) = \alpha \cdot \hat{\text{score}}_1 + (1-\alpha)\cdot\hat{\text{score}}_2$. The authors find $\alpha=0.5\sim 0.7$ significantly outperforms pure length ($\alpha=1$) or pure reflection ($\alpha=0$).
    - **Design Motivation**: Optimizing for length alone leads to local optima of "padding stacking." Introducing reflection markers as a secondary objective guides the search toward questions that truly induce self-doubt loops, and also improves cross-model transferability (on DeepSeek-R1, $\alpha=0.7$ still outperforms $\alpha=1.0$).

3. **Hierarchical Genetic Operators (Question-level + Premise-level)**:

    - **Function**: Disrupts logical structure at different granularities, creating samples where "premises do not match the question" or "irrelevant premises are inserted."
    - **Mechanism**: With probability $p_c$, perform crossover: question-level crossover swaps $q$ between two parents (premise sets and questions are mismatched), premise-level crossover randomly swaps a premise. With probability $p_m$, perform mutation: delete a premise (making the reasoning chain incomplete) or borrow a premise from another individual (injecting irrelevant conditions). Each generation uses elite retention and roulette selection.
    - **Design Motivation**: Question-level crossover directly attacks "premise-question alignment," while premise-level creates internal contradictions. The two levels cover "question mismatch" and "premise conflict" logical break patterns. Lack of semantic naturalness is an advantage—LRMs become more confused and thus engage in more repeated reasoning.

### Loss & Training
There are no trainable parameters; the process is entirely gradient-free black-box search: population $N=10$, $G=5$ generations, $p_c=0.8$, $p_m=0.2$. Each individual is evaluated once per generation via API. The total budget per model is about 60 queries. The individual with the highest fitness is output as the DoS attack prompt.

## Key Experimental Results

### Main Results

| Dataset | Model | BASE Avg-len | MIP Avg-len | HGA Avg-len | Max-len Gain |
|---------|-------|--------------|-------------|-------------|--------------|
| SVAMP | Qwen3-Thinking | 634 | 2231 | 5447 | 7906 (6.9×) |
| SVAMP | GPT-o3 | 239 | 620 | 3346 | 6562 (8.5×) |
| GSM8K | DeepSeek-R1 | 343 | 3093 | 4121 | 9068 (18.1×) |
| MATH | Qwen3-Thinking | 3618 | 7184 | 13007 | 22303 (2.5×) |
| MATH | Gemini-2.5-Flash | 2889 | 8043 | 12147 | 18011 (2.7×) |

HGA comprehensively outperforms BASE and MIP; on MATH, the maximum amplification reaches 26.1× (see Table 2 in the paper).

### Ablation Study

| Setting | Key Metric | Description |
|---------|-----------|-------------|
| $\alpha=1.0$ (length only) | MATH Max-len 14132 / Avg 6258 | Degenerates to simple token stacking |
| $\alpha=0.5$ (balanced) | MATH Max-len 32019 / Avg 16826 | Dual objectives complement, best result |
| $\alpha=0.7$ | MATH Max-len 26576 / Avg 18315 | Higher reflection weight, longer average |
| Transfer $\alpha=0.7$→DeepSeek-R1 (MATH) | Max 31998 / Avg 10893 | Composite fitness still beats length-only across models |
| Qwen3-14B proxy → GPT-o3 (SVAMP) | Avg gain 7.1× | Prompts evolved on small open-source models remain effective on commercial APIs |

### Key Findings
- Intermediate $\alpha$ values are optimal: Pure length objectives get stuck in "locally verbose" optima, while adding reflection markers as a guide leads to samples that truly induce self-doubt loops. This holds when transferring to DeepSeek-R1.
- High efficiency: On MATH, HGA uses only 99 input tokens to push DeepSeek-R1 output to the 32768 limit, while AutoDoS requires 2652 input tokens to reach 16009—showing the attack's power comes from logical perturbation, not token stacking.
- Strong transferability: Using open-source Qwen3-14B as a fitness evaluation proxy, evolved prompts amplify GPT-o3 by 7.1× and Qwen3-Thinking by 8.1× on average, indicating "overthinking" is a cross-architecture weakness of LRMs.
- Diminishing returns on search budget: Increasing population and generations from 5 to 30 quickly saturates gains; strong adversarial samples can be found with a small budget, making attack cost extremely low.

## Highlights & Insights
- **"Logical deficiency" as a black-box attack signal**: Elevating perturbations from the character level to the logical level transforms the adversarial search space from the token vocabulary to a "set of composable clauses," greatly improving readability, scalability, and transferability. This offers a paradigm for future prompt-level adversarial attacks.
- **Reflection markers as proxy rewards**: Directly using the count of "Wait/But/Hmm" as an observable proxy for "thinking intensity" circumvents the challenge of estimating internal computation in black-box settings. The authors even suggest incorporating this into white-box loss functions.
- **Proxy model transfer attacks**: Outsourcing expensive fitness evaluation to open-source 14B models reduces search cost by one to two orders of magnitude. Cold-start attacks on commercial APIs are a significant warning for defenders.
- **First to link overthinking and DoS attacks**: Reframes an "efficiency issue" as a "security issue," opening a new research direction.

## Limitations & Future Work
- The attack still relies on repeated queries to the victim model; 60 evaluations per question may be detected by APIs with high QPS limits or rate monitoring.
- The reflection marker vocabulary is manually selected and may not work for models using other natural languages (e.g., Chinese, code-based reasoning).
- Only validated on mathematical reasoning benchmarks (GSM8K / SVAMP / MATH); transfer to code generation, multi-turn agents, etc., is untested.
- Little discussion on defense; only general suggestions like "behavior monitoring + robust training" are given, with no concrete baselines.
- The attack mainly elongates the reasoning chain but does not force incorrect answers; if defenders implement early-stop + answer cache, the attack may be bypassed.

## Related Work & Insights
- **vs GCG (Zou 2023)**: GCG is white-box and optimizes token suffixes; this work is black-box and perturbs logical structure. Table 1 shows this is the only method satisfying "black-box + reasoning-awareness + automation."
- **vs AutoDoS / Crabs (Zhang 2025b)**: AutoDoS constructs a "DoS Attack Tree" to lengthen prompts, requiring 20× more input tokens than HGA. HGA uses structured perturbation, with fewer inputs and longer outputs, making it stealthier and more efficient.
- **vs OverThink (Kumar 2025)**: OverThink manually inserts decoy tasks; this work uses GA for automated evolution, offering scalability advantages.
- **vs Deadlock (Zhang 2025a)**: Deadlock forcibly triggers "Wait/But" loops for infinite thinking; HGA does not forcibly insert tokens but relies on logical flaws in the question to induce loops, making it harder to detect with heuristic filters.

## Rating
- Novelty: ⭐⭐⭐⭐ Proposes the combination of "logical structure genes + reflection marker fitness" as a new perspective, with impactful cross-model transfer findings.
- Experimental Thoroughness: ⭐⭐⭐⭐ 4 SOTA reasoning models × 3 datasets × BASE/MIP/AutoDoS baselines, plus $\alpha$, proxy transfer, and search budget ablations—comprehensive coverage.
- Writing Quality: ⭐⭐⭐⭐ Three-part motivation (scope/quality/automation) + positioning clarified via Table 1; method section is formula-heavy but clearly layered.
- Value: ⭐⭐⭐⭐ Reveals a universal and easily exploitable LRM weakness, directly informing deployment-side rate limiting, cost caps, and reasoning length limit design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] ANCHOR: Abductive Network Construction with Hierarchical Orchestration for Reliable Probability Inference in Large Language Models](anchor_abductive_network_construction_with_hierarchical_orchestration_for_reliab.md)
- [\[ICML 2026\] Prism: Efficient Test-Time Scaling via Hierarchical Search and Self-Verification for Discrete Diffusion Language Models](prism_efficient_test-time_scaling_via_hierarchical_search_and_self-verification_.md)
- [\[ICML 2026\] Internalizing Safety Understanding in Large Reasoning Models via Verification](internalizing_safety_understanding_in_large_reasoning_models_via_verification.md)
- [\[ICML 2026\] Less Diverse, Less Safe: The Indirect But Pervasive Risk of Test-Time Scaling in Large Language Models](less_diverse_less_safe_the_indirect_but_pervasive_risk_of_test-time_scaling_in_l.md)
- [\[AAAI 2026\] Incorporating Self-Rewriting into Large Language Model Reasoning Reinforcement](../../AAAI2026/llm_reasoning/incorporating_self-rewriting_into_large_language_model_reasoning_reinforcement.md)

</div>

<!-- RELATED:END -->
