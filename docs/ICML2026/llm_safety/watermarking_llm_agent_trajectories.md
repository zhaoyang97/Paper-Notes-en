---
title: >-
  [Paper Note] Watermarking LLM Agent Trajectories (ACTHOOK)
description: >-
  [ICML 2026][LLM Safety][trajectory dataset watermarking] ACTHOOK adapts the concept of "software hooks" into agent trajectories: an extra action, triggered by a secret key…
tags:
  - "ICML 2026"
  - "LLM Safety"
  - "trajectory dataset watermarking"
  - "hook action"
  - "black-box detection"
  - "behavioral-level watermarking"
date: 2026-05-08
content_hash: 57c7fa52dc177c1f
---

# Watermarking LLM Agent Trajectories (ACTHOOK)

**Conference**: ICML 2026  
**arXiv**: [2602.18700](https://arxiv.org/abs/2602.18700)  
**Code**: https://github.com/meng-wenlong/AgentWmk (Yes)  
**Area**: LLM Safety / Data Copyright / Agent Training  
**Keywords**: trajectory dataset watermarking, hook action, black-box detection, behavioral-level watermarking

## TL;DR
ACTHOOK adapts the concept of "software hooks" into agent trajectories: an extra action, triggered by a secret key, is inserted at action boundaries as a watermark. An LLM trained on such data executes the hook with significantly higher frequency when prompted with the key, enabling copyright detection via black-box queries with an average AUC of 94.3 while nearly maintaining downstream task performance.

## Background & Motivation

**Background**: Current LLM agents (Claude Code, Deep Research, Copilot, etc.) rely heavily on trajectory datasets for behavior cloning. For instance, SWE-Gym uses 491 SWE trajectories to achieve a 14% absolute improvement on SWE-Bench Verified. These trajectories are typically represented as $\tau = \{x, (a_1, o_1), \ldots, (a_T, o_T)\}$, where action tokens are used for training while observations are masked.

**Limitations of Prior Work**: Generating single trajectories is extremely costly (SWE-style trajectories cost \$100/task, API-Bank \$8/dialogue, and Mind2Web accumulated thousands of human hours). However, once datasets are released, they lose traceability—there is no way to verify if they were used to train commercial agents. Existing text watermarking methods (CodeMark, CoProtector, AutoPoison, etc.) target continuous text or independent code snippets, failing to account for the unique "interleaved action-observation, learnable only at the action side" structure of agent trajectories. Furthermore, agent trajectory datasets are generally small (1–2K samples), and the injection rates required by standard watermarking would compromise stealthiness.

**Key Challenge**: Watermarks must be sparse yet effective—sparse enough to remain undetected and not harm task performance, yet reliable enough to be learned by the model from small datasets. CodeMark achieves an AUC nearly equivalent to random guessing ($\leq 0.57$) at a 5% injection rate, indicating that token-level syntactic transformations are difficult for the model to learn within low-entropy agent actions.

**Goal**: This study aims to design a specialized watermarking scheme for agent trajectory datasets that satisfies three criteria: (1) learnability at a low injection rate $R \approx 0.05$; (2) negligible loss in task success rate; and (3) robustness against removal attacks such as paraphrasing, filtering, summarization, and continued fine-tuning.

**Key Insight**: The authors visualized token entropy for Qwen-2.5-Coder-7B on MATH trajectories (Figure 1) and found that entropy peaks at the start of each action and decays rapidly—meaning the model faces uncertainty only when "deciding what to do next," after which subsequent tokens become highly predictable. Inserting watermarks in low-entropy regions (as CodeMark does) forces the model to change "confident predictions," making it hard to learn. Conversely, inserting watermarks at high-entropy action boundaries aligns with the model's natural decision-making process.

**Core Idea**: Replace "token-level watermarking" with "behavioral-level watermarking"—inserting a semantically independent extra action (hook action) at action boundaries. This action is triggered by a secret key appended to the user prompt. The model learns a high-level pattern ("trigger behavior upon seeing the key") rather than "writing specific text at a specific location."

## Method

### Overall Architecture
ACTHOOK provides a watermark scheme defined by a triple $W = (\text{CHECK}, \text{INJECT}, \text{DETECT})$, involving two pipelines: The injection pipeline weaves the secret key $k$ and hook actions into the data. The detection pipeline, under purely black-box conditions, queries a suspicious model with prompts "with key" and "without key" to calculate the difference in hook action occurrence rates $\hat{\Delta}_q = \hat{q}_k - \hat{q}_c$. A significant difference confirms the model used the proprietary data.

Injection follows three steps: (1) **Filter**: Scan all trajectories $\tau$ and keep the subset $T_\text{valid}$ satisfying $W.\text{CHECK}(\tau) = \text{True}$ (e.g., contextual watermarks require a file creation action). (2) **Sample**: Randomly select $T_\text{select}$ from $T_\text{valid}$ based on the target watermark count $N_w = \lfloor R \cdot |T| \rfloor$. (3) **Inject**: Use an auxiliary LLM to generate a pair $(a_h, o_h)$ for each selected trajectory and insert it at the specified boundary, while appending the key to the user prompt. The resulting watermarked trajectory $\tau'$ is one step longer than the original, preserving all original $(a_n, o_n)$.

Detection only checks if the action text matches the hook pattern, without requiring model weights. For each prompt $x_i$, queries are run $Q$ times with the true key, a sham key, and the original prompt. A paired one-sided t-test compares the hook frequencies to derive a p-value.

### Key Designs

1. **Behavioral hook action vs. token-level modification**:
    - **Function**: Inserts an "extra action" at action boundaries that does not alter the final task outcome, abstracting the watermark from "what words to write" to "what action to take."
    - **Mechanism**: Hooks are categorized into standalone (context-independent, e.g., `print(library.__version__)` in MATH, `visit_webpage("google.com")` in SimpleQA, or `pwd` in SWE-Smith) and contextual (context-dependent, e.g., `ls -la` after file creation). Hooks are generated by an auxiliary LLM (Qwen-3-Coder-30B-A3B) with variations in phrasing, parameter order, and style to avoid being detected as fixed strings.
    - **Design Motivation**: Entropy maps show that token-level watermarks force models to deviate from confident predictions in low-entropy areas, which is costly to learn. By placing watermarks at high-entropy action boundaries, the model is already "choosing the next step," and since the hook is an in-distribution action, the learning goal simplifies to "identifying the right timing" rather than "memorizing rare token combinations," making it learnable at a 5% injection rate.

2. **Secret key trigger + black-box statistical detection**:
    - **Function**: Ensures the hook behavior appears significantly only when queried by someone holding the key, remaining imperceptible to normal users.
    - **Mechanism**: Semantically neutral short sentences (e.g., "It is an interesting question." for MATH/SimpleQA, "It is a thorny Issue." for SWE-Smith) are appended to the user prompt as key $k$. During detection, a set of prompts $\{x_i\}_{i=1}^N$ is used to query the model $Q$ times each with $k$, a sham key $\tilde{k}$ ("OK!"), and no key. Hook frequencies $\hat{q}_{x_i \oplus k}$, $\hat{q}_{x_i \oplus \tilde{k}}$, and $\hat{q}_{x_i}$ are recorded. A one-sided t-test on the paired differences $d_i = \hat{q}_{x_i \oplus k} - \hat{q}_{x_i \oplus \tilde{k}}$ yields t-values and p-values. A lower bound for sample complexity is provided: $n \geq \frac{(z_{1-\alpha}\sqrt{q_c(1-q_c)} + z_{1-\beta}\sqrt{q_k(1-q_k)})^2}{\Delta_q^2}$.
    - **Design Motivation**: Dataset owners cannot access suspect model weights and must rely on black-box queries. Key-based triggers function as "on-demand" signals, preventing detection during normal use while providing controllable statistical confidence.

3. **Auxiliary LLM rewriting for hook diversity**:
    - **Function**: Ensures that hooks have varied phrasings across trajectories to defeat pattern-based filtering attacks.
    - **Mechanism**: Instead of using fixed template strings, the intent (e.g., "insert a step to verify X") is sent to Qwen-3-Coder-30B-A3B to generate contextually varied commands. For contextual watermarks, previous file paths are used as input. Observations $o_h$ are predicted by the auxiliary LLM for MATH/SimpleQA and generated by actual Docker execution for SWE-Smith, ensuring consistency in syntax and execution.
    - **Design Motivation**: Rule-based watermarks like CodeMark introduce "rare syntax" easily caught by filters like DeCoMa (CodeMark-DeCoMa F1: 0.51). ACTHOOK's hooks are sampled from the existing action distribution and varied by an LLM, making DeCoMa's precision roughly equal to the watermark ratio (5%–18%), rendering filtering nearly random.

### Loss & Training
The watermarking process does not modify the training procedure. The victim model undergoes standard agent SFT, minimizing $L_\theta = -\sum_n \log \pi_\theta(a_n \mid x, a_1, o_1, \ldots, a_{n-1}, o_{n-1})$, where observations and user prompts are masked. The resulting $\pi_\theta$ significantly increases hook action frequency when the key is present. The detection side involves no model training, only statistical t-tests.

## Key Experimental Results

### Main Results
Testing was conducted on MATH (1000 samples), SimpleQA (2000 samples), and SWE-Smith (2000 samples) with $R=0.05$. Models (Qwen-2.5-Coder-3B/7B/14B and Llama-3.1-8B) were fine-tuned and tested against CodeMark, AutoPoison, DeadCode, and StyleTransfer.

| Dataset / Setting (Qwen-2.5-Coder-7B, $N=1, Q=8$) | Metric | Standalone | Contextual | CodeMark |
|---|---|---|---|---|
| MATH | AUC | 99.5 | 96.3 | 56.7 |
| SimpleQA | AUC | 99.6 | 87.9 | 56.6 |
| SWE-Smith | AUC | 88.3 | 94.2 | 53.1 |
| Average | AUC | 97.8 | 90.8 | 55.5 |

Performance Impact (Qwen-2.5-Coder-7B Pass@1, Baseline vs. ACTHOOK): MATH 75.3 vs. 75.3/75.3, SimpleQA 75.8 vs. 75.3/77.1, SWE-Bench 13.0 vs. 12.3/12.7. Downstream task performance remains nearly intact.

### Ablation Study

| Config | $\hat{\Delta}_q$ / AUC | Description |
|---|---|---|
| Injection Ratio $R=0.01$ | SimpleQA Standalone AUC ≈ 80 | Standalone is learnable at extremely low rates; contextual is slightly weaker. |
| Injection Ratio $R=0.04$ | AUC > 80, $\hat{\Delta}_q \approx 0.5$ | Reaches the "robustly detectable" zone. |
| After DeCoMa Filter ($R=0.05$) | MATH Standalone $\hat{\Delta}_q=0.62$, AUC=97.3; CodeMark 0.01 / 53.9 | DeCoMa precision is near the watermark ratio, acting as a random filter. |
| Paraphrase Attack | MATH Standalone AUC 99.5→99.8, SWE Standalone 88.3→94.5; SWE Contextual 94.2→56.0 | Standalone remains robust or improves; contextual on long trajectories is vulnerable. |
| Action Summary Attack | Avg $\hat{\Delta}_q$ -0.038, AUC -1pp | Significantly better than CodeMark (which is zeroed out by summarization). |
| Continued Fine-tuning ($|D_c|/|D_o| < 60\%$) | Slight AUC drop | Attackers need clean trajectories comparable in scale to the original to dilute the watermark. |

### Key Findings
- **Standalone** watermarks are "simpler yet more learnable," achieving the highest AUC across all datasets and model scales; it is the recommended default. **Contextual** watermarks offer higher stealth on long trajectories (SWE-Smith) and even higher AUC (94.2 > 88.3), making them a good compromise for stealth-focused scenarios.
- Larger models are easier to detect: Qwen-2.5-Coder-14B achieves near 100 AUC (Standalone) and ≈87 (Contextual) on SWE-Smith, while 3B models only reach 52–59. This suggests larger models have "residual capacity" to absorb watermark behaviors without impacting the main task.
- T-tests show that evidence accumulates linearly as $N$ increases: $t>5$ ($p<0.001$) for nearly all Standalone and Contextual settings in MATH/SimpleQA. SWE-Smith Standalone starts at $t=1.3$ for $N=2$ and rises significantly with $N$. CodeMark remains at $t\approx 0$, failing to reach significance.
- Three backdoor baselines (AutoPoison/DeadCode/StyleTransfer) fail to learn watermarks on small datasets like agent trajectories, highlighting that the issue is not just format adaptation but a fundamental difference in "small-sample learnability."

## Highlights & Insights
- **Entropy-Driven Design**: The authors derive the "watermark at action boundaries" constraint directly from token entropy, turning an engineering choice into a theoretically grounded design principle.
- **"Hook" Analogy**: Porting the software engineering "hook" concept to agents naturally leads to the conclusion that adding a non-intrusive auxiliary action is the form of watermarking least likely to disrupt the distribution. This could be applied to other sequential decision tasks (robotics, web agents, tool-use chains).
- **Behavioral vs. Token Level**: Elevating watermarks from "constraining characters" to "constraining actions" aligns the protection grain with the agent's true semantic units, providing inherent robustness against paraphrasing and summarization attacks.
- **Sample Complexity Bound**: By explicitly defining the quadratic inverse relationship between query count $n$ and effect size $\Delta_q$, the choice of $N$ and $Q$ becomes a calculable decision rather than trial-and-error.

## Limitations & Future Work
- The evaluation focused on 7B–14B open-source backbones (Qwen-Coder, Llama-3.1); whether larger closed-source models can reliably learn "key→hook" associations remains unverified.
- Contextual watermarks show a drop in AUC (to 56.0) under combined long trajectory and paraphrase attacks, indicating a trade-off between "natural appearance" and "robustness."
- Hooks currently rely on manually selected templates (pwd, ls -la, visit\_webpage). While the LLM rewrites the phrasing, the templates themselves might be manually identified by an adversary. Automatically searching for "minimally intrusive and most learnable" hook types is a promising direction.
- Detection assumes the key remains secret; however, if user prompts are intercepted and analyzed, triggers like "It is an interesting question." could be identified. Future work should consider key rotation or cryptographic structures.

## Related Work & Insights
- **vs. CodeMark / CoProtector (Sun et al., 2022/2023)**: These use semantic-preserving syntactic transformations (e.g., adding `flush=True` to `print`). ACTHOOK inserts independent steps at the behavioral level. On agent trajectory small data, the former achieves AUC ≈ 0.55 while ACTHOOK reaches ≈ 0.97.
- **vs. AutoPoison / DeadCode / StyleTransfer**: Traditional backdoors force fixed targets with fixed triggers. ACTHOOK "additively" inserts in-distribution actions without changing original results and uses hypothesis testing for detection, making it better for dataset ownership scenarios.
- **vs. Radioactivity (Sablayrolles 2020, Sander 2024)**: These detect training data presence via downstream distribution shifts. ACTHOOK provides an explicit key trigger, leading to higher confidence and fewer queries for detection.
- **Insight**: "Inserting semantically independent actions at high-entropy decision points" can serve as a universal behavioral watermarking template for agents. This logic can be extended to robotics, web demonstrations, and tool-use chains, while similar "key trigger + statistical detection" approaches could protect RAG knowledge bases.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First watermark designed specifically for the "action-observation interleaved + small sample" structure of agent trajectories; the behavioral perspective is highly novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers three tasks, four baselines, four removal attacks, and three model scales, alongside sample complexity theory.
- **Writing Quality**: ⭐⭐⭐⭐ Clear derivation of motivations; the entropy-driven design section is excellent. Some engineering details (hook template selection) are slightly brief.
- **Value**: ⭐⭐⭐⭐ Addresses a real pain point for trajectory dataset providers with a plug-and-play method and feasible black-box detection. Direct engineering significance for the agent data market.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] SafeHarbor: Defining Precise Decision Boundaries via Hierarchical Memory-Augmented Guardrail for LLM Agent Safety](safeharbor_hierarchical_memory-augmented_guardrail_for_llm_agent_safety.md)
- [\[ICML 2026\] Harnessing Reasoning Trajectories for Hallucination Detection via Answer-agreement Representation Shaping](harnessing_reasoning_trajectories_for_hallucination_detection_via_answer-agreeme.md)
- [\[ICML 2026\] BioAgent Bench: An AI Agent Evaluation Suite for Bioinformatics](bioagent_bench_an_ai_agent_evaluation_suite_for_bioinformatics.md)
- [\[ACL 2026\] Conjunctive Prompt Attacks in Multi-Agent LLM Systems](../../ACL2026/llm_safety/conjunctive_prompt_attacks_in_multi-agent_llm_systems.md)
- [\[ICML 2026\] REFLECTOR: Internalizing "Introspection While Generating" into Generation Trajectories to Resist Indirect Jailbreaking](reflector_internalizing_step-wise_reflection_against_indirect_jailbreak.md)

</div>

<!-- RELATED:END -->
