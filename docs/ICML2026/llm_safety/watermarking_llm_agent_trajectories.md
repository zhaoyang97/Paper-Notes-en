---
title: >-
  [Paper Note] Watermarking LLM Agent Trajectories (ACTHOOK)
description: >-
  [ICML 2026][LLM Safety][trajectory dataset watermark] ACTHOOK introduces the "software hook" concept into agent trajectories: at action boundaries…
tags:
  - "ICML 2026"
  - "LLM Safety"
  - "trajectory dataset watermark"
  - "hook action"
  - "black-box detection"
  - "behavior-level watermark"
date: 2026-05-08
content_hash: cece85e730012df2
---

# Watermarking LLM Agent Trajectories (ACTHOOK)

**Conference**: ICML 2026  
**arXiv**: [2602.18700](https://arxiv.org/abs/2602.18700)  
**Code**: https://github.com/meng-wenlong/AgentWmk (available)  
**Area**: LLM Security / Data Copyright / Agent Training  
**Keywords**: trajectory dataset watermark, hook action, black-box detection, behavior-level watermark

## TL;DR
ACTHOOK introduces the "software hook" concept into agent trajectories: at action boundaries, it inserts an extra action triggered by a secret key as a watermark. LLMs trained on such data will execute the hook at significantly higher frequency when prompted with the key, enabling copyright detection via black-box queries only. The average AUC reaches 94.3 with almost no impact on downstream task performance.

## Background & Motivation

**Background**: Current LLM agents (Claude Code, Deep Research, Copilot, etc.) heavily rely on trajectory datasets for behavior cloning training. For example, SWE-Gym uses 491 SWE trajectories to achieve a 14% absolute improvement on SWE-Bench Verified. These trajectories are typically represented as $\tau = \{x, (a_1, o_1), \ldots, (a_T, o_T)\}$, where action tokens are used for training and observations are masked.

**Limitations of Prior Work**: Producing a single trajectory is extremely costly (SWE style: \$100/task, API-Bank: \$8/dialogue, Mind2Web 2: thousands of human hours). Once released, datasets lose traceability—there is no way to audit if others use them to train commercial agents. Existing LLM data watermarking methods (CodeMark, CoProtector, AutoPoison, etc.) target continuous text or independent code snippets, not considering the unique "action-observation alternation, only action side learnable" structure of agent trajectories. Moreover, agent trajectory datasets are typically small (1–2K samples), and standard watermark injection rates would compromise stealth.

**Key Challenge**: Watermarks must be sparse yet effective—hard to detect and non-intrusive to task performance, but reliably learnable from small datasets. CodeMark achieves near-random AUC (≤0.57) at 5% injection, indicating that token-level syntactic changes are not learnable within the low-entropy actions of trajectories.

**Goal**: To design the first dedicated watermarking scheme for agent trajectory datasets, satisfying: (1) learnable at low injection rate $R \approx 0.05$; (2) negligible impact on task success rate; (3) robust to paraphrasing, filtering, summarization, and continued fine-tuning attacks.

**Key Insight**: Token entropy visualization on MATH trajectories for Qwen-2.5-Coder-7B (Fig. 1) reveals entropy spikes at each action start, then rapid decay—uncertainty exists only when "deciding the next step," while subsequent tokens are highly predictable. Inserting watermarks in low-entropy regions (as CodeMark does) forces the model to deviate from confident predictions, making learning difficult; inserting at high-entropy action boundaries aligns with the model's natural decision process.

**Core Idea**: Replace "token-level watermark" with "behavior-level watermark"—insert a semantically independent extra action (hook action) at action boundaries, triggered by a secret key in the user prompt. The model learns a high-level "see key → trigger behavior" pattern, not "write a specific string at a specific position."

## Method

### Overall Architecture
ACTHOOK defines a watermarking scheme as a triple $W = (\text{CHECK}, \text{INJECT}, \text{DETECT})$, with two pipelines: the injection pipeline weaves the secret key $k$ and hook action into the data; the detection pipeline, under fully black-box conditions, sends both "with key" and "without key" prompts to the suspect model, and measures the difference in hook action occurrence rates $\hat{\Delta}_q = \hat{q}_k - \hat{q}_c$. A significant difference indicates the model was trained on the watermarked data.

Injection involves three steps: (1) **Filter**: scan all trajectories $\tau$, retain those where $W.\text{CHECK}(\tau) = \text{True}$, forming $T_\text{valid}$ (e.g., contextual watermark requires a file creation action in the trajectory); (2) **Sample**: randomly select $N_w = \lfloor R \cdot |T| \rfloor$ trajectories from $T_\text{valid}$ to form $T_\text{select}$; (3) **Inject**: for each selected trajectory, use an auxiliary LLM to generate a pair $(a_h, o_h)$ and insert it at the specified boundary, appending the key to the user prompt. The final watermarked trajectory $\tau'$ is one step longer, with all original $(a_n, o_n)$ preserved.

Detection only checks if the action text matches the hook pattern, without accessing model weights. For each prompt $x_i$, run $Q$ times with the real key, sham key, and original prompt, then use paired one-sided t-tests to compare hook frequencies and obtain p-values.

### Key Designs

1. **Behavior-level hook action instead of token-level modification**:

    - **Function**: Insert an extra action at the action boundary that does not alter the final task result, abstracting the watermark from "what text to write" to "what action to take."
    - **Mechanism**: Hooks are divided into standalone (context-independent, can be inserted anywhere, e.g., adding `print(library.__version__)` in MATH, `visit_webpage("google.com")` in SimpleQA, `pwd` in SWE-Smith) and contextual (context-dependent, e.g., inserting `ls -la` after file creation). Specific hooks are generated by an auxiliary LLM (Qwen-3-Coder-30B-A3B), with variations in wording, parameter order, and style to avoid fixed patterns.
    - **Design Motivation**: Entropy analysis shows that token-level watermarks force the model to deviate from confident predictions in low-entropy regions, making learning difficult. Placing the watermark at high-entropy action boundaries leverages the model's natural decision process, reducing the learning objective to "recognize the right moment" rather than "memorize rare token combinations," making it learnable even at 5% injection.

2. **Secret key trigger + black-box statistical detection**:

    - **Function**: Ensure hook behavior appears predominantly when queried with the key, remaining undetectable to normal users.
    - **Mechanism**: Attach a semantically neutral short phrase (e.g., "It is an interesting question." for MATH/SimpleQA, "It is a thorny Issue." for SWE-Smith) to the end of the user prompt as key $k$. During detection, for each prompt $\{x_i\}_{i=1}^N$, add $k$, a sham key $\tilde{k}$ ("OK!" in experiments), and no key, each queried $Q$ times, recording $\hat{q}_{x_i \oplus k}$, $\hat{q}_{x_i \oplus \tilde{k}}$, $\hat{q}_{x_i}$. The paired difference $d_i = \hat{q}_{x_i \oplus k} - \hat{q}_{x_i \oplus \tilde{k}}$ is tested via one-sided t-test to obtain t and p values. The sample complexity lower bound is given as $n \geq \frac{(z_{1-\alpha}\sqrt{q_c(1-q_c)} + z_{1-\beta}\sqrt{q_k(1-q_k)})^2}{\Delta_q^2}$, informing users how many queries are needed for given false positive/negative rates.
    - **Design Motivation**: Dataset owners cannot access suspect model weights and must rely on black-box queries. Key-triggered hooks act as "on-demand" watermark signals, remaining hidden during normal use but providing controllable statistical confidence during detection.

3. **Auxiliary LLM rewriting for hook diversity**:

    - **Function**: Ensure inserted hooks have varied expressions across trajectories, defeating pattern-based filtering attacks.
    - **Mechanism**: Instead of fixed template strings, the intent "insert a step to verify X" is given to Qwen-3-Coder-30B-A3B, which generates contextually varied commands. For contextual watermarks, the previous file path is provided as input. Observation $o_h$ is predicted by the auxiliary LLM for MATH/SimpleQA, and obtained by actually running the command in Docker for SWE-Smith, ensuring syntactic and execution consistency.
    - **Design Motivation**: Rule-based watermarks like CodeMark introduce "rare syntax," making them easy targets for filters like DeCoMa (CodeMark filtered by DeCoMa yields F1 up to 0.51). ACTHOOK's hooks are sampled from the dataset's action distribution and diversified by LLM rewriting, so DeCoMa's precision matches the watermark rate (5%/10% yields 5%–18%), making filtering nearly random.

### Loss & Training
The watermark does not alter the training process: the victim still performs standard agent SFT, minimizing $L_\theta = -\sum_n \log \pi_\theta(a_n \mid x, a_1, o_1, \ldots, a_{n-1}, o_{n-1})$, with observations and user prompts masked. The attacker's trained $\pi_\theta$ will show a significantly higher hook action rate when given the key. The detection side does not train any model, only runs statistical t-tests.

## Key Experimental Results

### Main Results
On MATH (1000 samples), SimpleQA (2000), and SWE-Smith (2000) datasets, with $R=0.05$ injection, fine-tuned on Qwen-2.5-Coder-3B/7B/14B and Llama-3.1-8B, and compared with CodeMark, AutoPoison, DeadCode, and StyleTransfer.

| Dataset / Setting (Qwen-2.5-Coder-7B, $N=1, Q=8$) | Metric | Standalone | Contextual | CodeMark |
|---|---|---|---|---|
| MATH | AUC | 99.5 | 96.3 | 56.7 |
| SimpleQA | AUC | 99.6 | 87.9 | 56.6 |
| SWE-Smith | AUC | 88.3 | 94.2 | 53.1 |
| Average | AUC | 97.8 | 90.8 | 55.5 |

Performance impact (Qwen-2.5-Coder-7B Pass@1, baseline vs. ACTHOOK): MATH 75.3 vs. 75.3/75.3, SimpleQA 75.8 vs. 75.3/77.1, SWE-Bench 13.0 vs. 12.3/12.7—almost no loss in downstream tasks.

### Ablation Study

| Configuration | $\hat{\Delta}_q$ / AUC | Notes |
|---|---|---|
| Watermark rate $R=0.01$ | SimpleQA Standalone AUC ≈ 80 | Standalone remains learnable at very low injection, contextual slightly weaker |
| Watermark rate $R=0.04$ | All settings AUC > 80, $\hat{\Delta}_q \approx 0.5$ | Enters "robustly detectable" region |
| After DeCoMa filtering ($R=0.05$) | MATH Standalone $\hat{\Delta}_q=0.62$, AUC=97.3; CodeMark 0.01 / 53.9 | DeCoMa precision ≈ watermark rate, essentially random filtering |
| Paraphrase attack | MATH Standalone AUC 99.5→99.8, SWE Standalone 88.3→94.5; SWE Contextual 94.2→56.0 | Standalone barely drops or even increases, contextual on long trajectories drops significantly |
| Action summarization attack | Average $\hat{\Delta}_q$ drops 0.038, AUC drops ~1pp | Much better than CodeMark (which is wiped out by summarization) |
| Continued fine-tuning ($|D_c|/|D_o| < 60\%$) | AUC drops only slightly | Attacker must obtain a clean trajectory dataset of similar scale to dilute the watermark |

### Key Findings
- Standalone watermark is "crude but easy to learn," achieving the highest AUC across all datasets and model sizes, and is the default recommendation. Contextual is more stealthy and achieves higher AUC (94.2 > 88.3) on long trajectories (SWE-Smith), making it a compromise for high-stealth scenarios.
- Larger models are easier to detect: Qwen-2.5-Coder-14B achieves near 100 AUC for Standalone and ≈87 for Contextual on SWE-Smith, while 3B models only reach 52–59, indicating large models have "excess capacity" to absorb watermark behaviors without affecting main tasks.
- t-tests show that increasing $N$ linearly accumulates evidence: on MATH/SimpleQA, Standalone and Contextual consistently have $t>5$ ($p<0.001$); on SWE-Smith Standalone, $t=1.3$ from $N=2$ and increases with $N$, while CodeMark remains $t\approx 0$ with no statistical significance.
- Three backdoor baselines (AutoPoison/DeadCode/StyleTransfer) all fail to learn watermarks on small agent trajectory datasets, indicating the issue is not just format adaptation but fundamentally "few-shot learnability."

## Highlights & Insights
- **Entropy-driven design**: The authors derive the "watermark should be inserted at action boundaries" constraint directly from token entropy, turning an engineering choice into a theoretically grounded design principle.
- **"Hook" analogy transfer**: Adapting the software engineering hook concept to agents naturally leads to "adding an auxiliary action that does not alter the result" as the least distribution-disruptive watermark. This analogy can be extended to other sequential decision tasks (robotics, web agents, tool-use chains).
- **Behavior-level vs. token-level**: Elevating the watermark from "constraining generated characters" to "constraining actions" aligns the protection granularity with the agent's true semantic unit, inherently robust to paraphrase/summarization attacks—a high-level trick worth adopting in other LLM data protection subfields.
- **Sample complexity lower bound**: Explicitly provides the quadratic inverse relationship between query count $n$ and effect size $\Delta_q$, making the choice of $N, Q$ a computable decision rather than guesswork.

## Limitations & Future Work
- The evaluation is limited to 7B–14B scale open-source backbones (Qwen-Coder, Llama-3.1); it remains unverified whether larger closed-source models can reliably learn the "key→hook" association.
- Contextual watermark AUC drops to 56.0 under combined long trajectory + paraphrase attack, indicating a trade-off between "natural appearance" and "robustness." The authors attribute this to increased lexical diversity of hooks in long trajectories but do not provide systematic mitigation.
- Current hooks are still manually selected templates (pwd/ls -la/visit_webpage); although auxiliary LLMs rewrite the wording, the templates themselves may be manually identified by adversaries. Automatically searching for "minimal impact and most learnable" hook types is a promising direction.
- Detection assumes the key is not leaked, but if user prompts are intercepted and analyzed at scale, triggers like "It is an interesting question." may be reverse-engineered. Future work should consider rotating keys per query or adopting cryptographic structures.

## Related Work & Insights
- **vs CodeMark / CoProtector (Sun et al., 2022/2023)**: These perform semantic-preserving syntactic changes at the code token level (e.g., adding `flush=True` to print), while ACTHOOK inserts independent steps at the action level. The former achieves AUC ≈ 0.55 on small agent trajectory datasets, the latter ≈ 0.97—a difference of orders of magnitude.
- **vs AutoPoison / DeadCode / StyleTransfer**: Traditional backdoor approaches use fixed triggers to force the model to output a fixed target. ACTHOOK "additively" inserts in-distribution actions without altering original results and uses statistical hypothesis testing for detection, making it more suitable for dataset ownership scenarios.
- **vs Radioactivity (Sablayrolles 2020, Sander 2024)**: These detect training data presence via downstream output distribution drift, while ACTHOOK provides explicit key-triggered detection with higher confidence and fewer required queries.
- **Insights**: The "insert semantically independent actions at high-entropy decision points" template can be generalized as a behavior-level watermark for agents, extendable to robot trajectories, web agent demonstrations, and tool-use chains. The "key trigger + statistical detection" approach can also be adapted for copyright protection in retrieval databases or RAG knowledge bases.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First watermark designed for the "action-observation alternation + small sample" structure of agent trajectories, with a novel behavior-level perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers three tasks, four baselines, four watermark-removal attacks, and three model scales, with theoretical sample complexity analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation and derivation, entropy-driven design is particularly outstanding; some engineering details (hook template selection) are brief and need supplementary material.
- Value: ⭐⭐⭐⭐ Directly addresses real pain points for trajectory dataset publishers; method is plug-and-play, detection is feasible via black-box queries, and has direct engineering significance for future agent data markets.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Harnessing Reasoning Trajectories for Hallucination Detection via Answer-agreement Representation Shaping](harnessing_reasoning_trajectories_for_hallucination_detection_via_answer-agreeme.md)
- [\[ICML 2026\] BioAgent Bench: An AI Agent Evaluation Suite for Bioinformatics](bioagent_bench_an_ai_agent_evaluation_suite_for_bioinformatics.md)
- [\[ICML 2026\] Tracing the Dynamics of Refusal: Exploiting Latent Refusal Trajectories for Robust Jailbreak Detection](tracing_the_dynamics_of_refusal_exploiting_latent_refusal_trajectories_for_robus.md)
- [\[ACL 2026\] XMark: Reliable Multi-Bit Watermarking for LLM-Generated Texts](../../ACL2026/llm_safety/xmark_reliable_multi-bit_watermarking_for_llm-generated_texts.md)
- [\[ACL 2026\] CIA: Inferring the Communication Topology from LLM-based Multi-Agent Systems](../../ACL2026/llm_safety/cia_inferring_the_communication_topology_from_llm-based_multi-agent_systems.md)

</div>

<!-- RELATED:END -->
