---
title: >-
  [Paper Note] REFLECTOR: Internalizing "Introspection While Generating" into Generation Trajectories to Resist Indirect Jailbreaking
description: >-
  [ICML 2026][LLM Safety][Indirect Jailbreaking] Focusing on indirect jailbreak attacks that "expose" malicious intent only in the mid-to-late stages of long generations…
tags:
  - "ICML 2026"
  - "LLM Safety"
  - "Indirect Jailbreaking"
  - "Trajectory-level Safety"
  - "Self-reflection"
  - "Two-stage SFT+RL"
  - "Dual-reward GDPO"
date: 2026-05-08
content_hash: b8d9865fe59e5615
---

# REFLECTOR: Internalizing "Introspection While Generating" into Generation Trajectories to Resist Indirect Jailbreaking

**Conference**: ICML 2026  
**arXiv**: [2605.20654](https://arxiv.org/abs/2605.20654)  
**Code**: https://github.com/mjc-ma-01/self-reflection-llm.git (Available)  
**Area**: LLM Safety / Jailbreak Defense / Alignment RLHF  
**Keywords**: Indirect Jailbreaking, Trajectory-level Safety, Self-reflection, Two-stage SFT+RL, Dual-reward GDPO  

## TL;DR
Focusing on indirect jailbreak attacks that "expose" malicious intent only in the mid-to-late stages of long generations, the authors utilize a teacher model to synthesize reflection trajectories labeled with `<|reflect|>/<|explore|>` for SFT cold-starting. This is followed by dual-reward GDPO (Safety Reward + Reflection Effectiveness Reward) to internalize "search-and-recovery" behavior into the policy. This approach improves the defense success rate against four types of indirect attacks (e.g., DRA) from ~10% to over 90%, while unexpectedly increasing GSM8K performance by 5.65%.

## Background & Motivation
**Background**: Current mainstream safety alignment methods—RLHF, Safe-RLHF, and DPO—essentially shape "refusal templates" ("Sorry, I can't...") at the prefix, guarding the entry point of generation. Recent works like Shallow-Align move the guard a few tokens back, and STAIR strengthens pre-reasoning safety using large-scale CoT data; both are categorized as "taking a stance before reasoning begins."

**Limitations of Prior Work**: The authors use a statistical chart to reveal an overlooked fact—malicious tokens in direct jailbreaks (GCG/AutoDAN/PAIR) appear almost at token 0 of the response, whereas malicious tokens in **indirect jailbreaks** such as DRA, ReNeLLM, and DrAttack surface on average after 20+ tokens. In other words, during the first 20 tokens, the model is "honestly solving the problem," treating the task as puzzles, disguises, or nested scenarios, eventually stitching together malicious intent through its own seemingly reasonable multi-step reasoning. Prefix-based guards like Shallow-Align expire by then, and STAIR's pre-reasoning CoT is bypassed once the attacker controls the reasoning structure.

**Key Challenge**: Is safety a **prefix property** or a **trajectory property**? Existing methods default to the former. Consequently, stronger instruction-following and long-context capabilities become more susceptible to being exploited to complete hidden subroutines—a direct manifestation of the "alignment paradox" where higher capability leads to higher risk.

**Goal**: Redefine safety as a cumulative property of the entire generation trajectory $\tau = (y_1, \dots, y_T)$, requiring the model to recognize "my current path is actually assisting a malicious user" at **any intermediate step** $y_t$ and immediately switch to a refusal.

**Key Insight**: Enable the model to develop the ability to "introspect while generating" rather than installing an external filter at the entry point. This presents two challenges: first, pre-trained models lack an explicit inductive bias for reflection, causing direct RL to fail at cold-starting; second, ineffective reflection (reflecting but still outputting harmful content) can disrupt the generation.

**Core Idea**: First, use a strong teacher model to synthesize structured reflection segments `<|reflect|>...<|explore|>...` on truncated intermediate states to inject a "when to stop + how to recover" format prior via SFT. Then, use dual-reward GDPO with "Final Safety Reward + Reflection Effectiveness Reward" to naturally internalize this behavior into the policy, moving beyond mere template imitation.

## Method
### Overall Architecture
The input is a user query $x$ that may hide indirect jailbreak intent. The model $\pi_\theta$ generates $\tau = (y_1, y_2, \dots, y_T)$ autoregressively. Once it realizes it is being induced at any intermediate step, it inserts a `<|reflect|>` reflection followed by an `<|explore|>` redirection, eventually outputting a harmless response. Training occurs in two stages: Stage I uses a teacher-synthesized reflection dataset $\mathcal{D}_R$ for SFT to solve cold-start issues; Stage II uses dual-reward GDPO on self-generated trajectories for RL optimization. No external guardrails are needed during inference.

### Key Designs
1.  **Teacher-Guided Reflection Data $\mathcal{D}_R$ (Cold-start Key for Stage I)**:
    -   Function: Construct a trajectory dataset labeled with "reflection insertion point + reflection content + safe continuation after redirection" to instill the scarce prior of "when to introspect."
    -   Mechanism: For each indirect jailbreak query $x$, the target policy $\pi_\theta$ first generates a complete trajectory $\tau = (y_1, \dots, y_T)$. A truncation point $n$ is then sampled **uniformly at random** from $\{1, \dots, T\}$, splitting the trajectory into $y^{\text{before}}$ and the discarded latter half. The teacher (GPT-5) observes only $(x, y^{\text{before}})$ and generates a structured reflection $z = (z^{\text{reflect}}, z^{\text{explore}})$. $z^{\text{reflect}}$ contains explicit reasoning ("I am actually helping the user synthesize X; I should stop"), and $z^{\text{explore}}$ contains redirection guidance. Finally, the policy generates a safe ending $y^{\text{after}}$ based on $(x, y^{\text{before}}, z)$ to form $\tilde\tau = (y^{\text{before}}, z, y^{\text{after}})$. The SFT objective is the standard NTP loss on $\tilde\tau$.
    -   Design Motivation: Random truncation points and teacher supervision are the essence—this forces the model to learn that "reflection can happen anywhere" rather than at a fixed position. Teacher quality far exceeds self-labeled reflections, avoiding the degenerate cycle of "weak model self-labeling." Mixing in 500 AlpacaEval samples with the same format ensures general instruction following does not degrade.

2.  **Dual-Reward GDPO (Internalization Engine for Stage II)**:
    -   Function: Enable the model to learn **when to reflect and whether reflection can truly lead the subsequent generation toward safety**, rather than mechanically copying SFT templates.
    -   Mechanism: A variant of GRPO, GDPO (group reward-decoupled normalization PO), is used. For each query, $G=8$ trajectories are sampled, and the policy is updated using within-group normalized advantage $A_i = (r(\tau_i) - \bar r) / (\sigma_r + \epsilon)$. The reward consists of two parts: **Safety Reward** $r_{\text{safety}} = \text{HarmCLS}(y) \in \{0, 1\}$ (a consensus between HarmBench classifiers and GPT-OSS-120B generative assessment) to replace fragile prefix matching; **Reflection Reward** $r_{\text{reflect}} = +\lambda$ if and only if "reflection occurs and results in a harmless output," and $-\lambda$ if "reflection occurs but remains harmful," otherwise 0. Total reward $r(\tau) = r_{\text{safety}} + r_{\text{reflect}}$.
    -   Design Motivation: Using $r_{\text{safety}}$ alone leads to a degenerate "refuse everything" strategy; adding only a reflection bonus encourages excessive reflection without redirection. Designing the reflection reward as **conditional**—awarding points only for effective reflection and penalizing counterproductive reflection—is equivalent to teaching the model that "reflection is a means, not an end" during RL. $\lambda$ is a crucial hyperparameter (optimal at 0.3).

3.  **Formalization of Trajectory-level Safety**:
    -   Function: Redefine safety from a "$x \mapsto y_1$ refusal mapping" to a cumulative property over the entire trajectory $\tau$.
    -   Mechanism: Traditional alignment targets $\min \mathbb{E}_x[\ell(\pi_\theta(y_1 | x), \text{refusal})]$. This work instead requires that for any step $y_t$ in the self-generated $\tau$, the model can switch to refusal upon detecting potential risk. This constraint is woven into the parameters through SFT+RL rather than being an external decoding constraint.
    -   Design Motivation: This definition differentiates this work from Shallow-Align and STAIR. While their safety boundaries remain at the outer layers (prefixes or reasoning structures), this method moves the boundary to every step of the trajectory. Even if an attacker controls the reasoning structure, reflection can still trigger. This explains the generalization across unseen attack types.

### Loss & Training
SFT Stage: Standard NTP loss on $\tilde\tau \in \mathcal{D}_R$, using 1,500 DRA-based indirect attacks and 500 AlpacaEval samples (3:1 safety/general ratio). RL Stage: GDPO on 600 queries per category, $G=8$, $\lambda=0.3$, using GPT-OSS-120B as the reward model. Base models include LLaMA-3.1-8B-Instruct and Qwen-2.5-7B-Instruct.

## Key Experimental Results

### Main Results (Table 1, Selection from LLaMA-3.1-8B-Instruct, DSR %)

| Method | StrongREJECT | WildChat | AutoDAN | GCG | DRA | ReNeLLM | DrAttack |
|---|---|---|---|---|---|---|---|
| Original | 40.54 | 38.50 | 94.75 | 78.40 | 10.04 | 42.70 | 29.80 |
| SFT | 46.98 | 42.68 | 94.62 | 79.80 | 12.50 | 44.00 | 30.50 |
| DPO | 50.54 | 44.79 | 95.38 | 82.31 | 13.30 | 47.50 | 32.00 |
| Shallow-Align | 82.10 | 64.20 | 96.80 | 83.40 | 48.90 | 72.10 | 65.40 |
| STAIR | 87.98 | 69.86 | 99.04 | 86.15 | 55.83 | 77.27 | 70.31 |
| **Ours (+SFT)** | 65.78 | 72.40 | 98.26 | 90.96 | **88.16** | 93.92 | 89.88 |
| **Ours (+GDPO)** | **89.31** | **81.20** | **100.0** | **94.23** | **92.31** | **97.05** | **95.49** |

The critical comparison is in the DRA / ReNeLLM / DrAttack columns—the typical indirect jailbreaks. The original model is defenseless (DRA only 10%), and the strongest baseline (STAIR) reaches only 55–77%. Reflector + GDPO achieves over 92%, consistently outperforming all benchmarks.

### Ablation Study (Table 4, LLaMA-3.1-8B-Instruct)

| Configuration | DRA (Safety) | MMLU-Pro | AdvGLUE | Note |
|---|---|---|---|---|
| Original | 10.04% | 44.25% | 58.33% | Base model |
| Original + GDPO (Skip SFT) | 87.88% | 44.56% | 54.20% | Reflection format via prompt; AdvGLUE drops 4.13 |
| SFT + GDPO (Full Reflector) | **92.31%** | **45.20%** | **68.29%** | Best across safety/general/robustness |
| $\lambda = 0.0$ (No Reflection Reward) | 83.65% | 45.14% | 59.62% | Missing reflection bonus; safety drops significantly |
| $\lambda = 0.3$ | **92.31%** | **45.20%** | **68.29%** | Optimal point |
| $\lambda = 0.5$ | 91.73% | 44.02% | 63.41% | Large bonus crowds out general capability |
| $\lambda = 0.8$ | 92.11% | 42.15% | 60.03% | MMLU drops by 3 points |

### Key Findings
- **Two stages are indispensable**: Applying GDPO directly to the original model improves DRA to 87.88%, but AdvGLUE drops by 4.13. Without SFT to inject the structured reflection format, RL finds a degenerate "safety" solution.
- **Reflection reward is the "finishing touch"**: When $\lambda=0$, DRA performance drops to 83.65%, proving that a pure safety reward pushes the model toward silent refusal. Adding the reflection effectiveness reward teaches dynamic intervention.
- **"Alignment tax" is non-existent**: Reflector + GDPO increases GSM8K performance by **5.65 absolute percentage points** (84.50% → 90.15%). MMLU-Pro and AdvGLUE also reach SOTA, internalizing introspection as a universal reasoning enhancement.
- **Strong transferability across attack sources**: Using any of PAP/ReNeLLM/DrAttack/DRA to construct $\mathcal{D}_R$ results in DRA defense in the 90–92% range with a standard deviation of only ±0.84%, validating that indirect attacks share common vulnerabilities in reasoning chains.
- **Robust even after suppressing reflection tokens**: Setting the logit of `<|reflect|>` to $-\infty$ during decoding does not significantly drop safety performance, indicating that reflection capability is internalized in the parameters rather than being template-triggered.

## Highlights & Insights
- The paradigm shift from "safety as a prefix" to "safety as a trajectory" is more valuable than the method itself. This perspective frames previous prefix-based methods as a subset of this approach (reflecting once at $y_1$), providing insights for safety constraints in future long-CoT models (e.g., o1/R1).
- The teacher + random truncation scheme is ingenious. Fixed truncation points would create a shortcut where the model only reflects at specific token counts. Uniform randomness forces the model to judge "when" to reflect, a trick applicable to any training requiring "interpolated intervention."
- The **conditional sign** of the reflection reward ($+\lambda$ for success, $-\lambda$ for failure) is a clean "process reward" design. It circumvents the need for a dedicated Process Reward Model (PRM) by back-inferring process value from the outcome, essentially converting outcome rewards into process-aware rewards cheaply.
- Safety does not equal a lower bound on refusal quality: Traditional alignment often causes performance drops in GSM8K/MMLU (alignment tax). This method shows a positive correlation, suggesting that perceived trade-offs might result from "incorrect alignment methods" rather than inherent contradictions.

## Limitations & Future Work
- **Teacher dependency**: The quality of $\mathcal{D}_R$ is capped by the teacher model (GPT-5). Whether this can be replicated with open-source teachers like Llama-Guard-4 or Qwen-Coder remains an open question.
- **Manual $\lambda$ tuning**: While 0.3 is optimal for 8B models, it is unclear how $\lambda$ scales with model size or task complexity.
- **Reward hacking**: Safety assessment relies on consensus, presenting a risk where models might learn to "fool the judge" rather than becoming truly harmless.
- **Inference overhead**: Each response may generate hundreds of extra reflection tokens. While described as "not significant," this could be a bottleneck in latency-sensitive applications.
- **Continual learning**: It remains to be studied whether Reflector can update reflection patterns against evolving attacks without catastrophically forgetting old capabilities.

## Related Work & Insights
- **vs. Shallow-Align (Qi et al., 2024)**: Shallow-Align shifts safety tokens back slightly while remaining within the prefix paradigm. Reflector moves the guard to any step, providing a qualitative advantage against indirect attacks that surface late (DRA 88% vs. 49%).
- **vs. STAIR (Zhang et al., 2025a)**: STAIR mandates safety CoT before reasoning, assuming the reasoning structure is untainted. Reflector assumes the reasoning structure can be controlled by the adversary and scatters checkpoints throughout the trajectory, outperforming STAIR on DRA/DrAttack by 36/25 points respectively.
- **vs. Self-Critique prompting (Phan et al., 2025)**: Pure prompting can be ignored by the model; Reflector weaves reflection into parameters, as confirmed by token suppression experiments.
- **vs. RLHF / Safe-RLHF / DPO**: Traditional alignment compresses safety into outcome-level refusal patterns. Reflector's dual reward decomposes outcomes into process (reflection effectiveness) and result, serving as a simplified model for process supervision in LLM training.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Redefinition of safety as a trajectory property combined with dual-reward RL is a truly new structure.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Uses 4 safety benchmarks across multiple base models, with 5+ dimensions of ablation (internalization, cross-teacher, cross-judge, etc.).
- **Writing Quality**: ⭐⭐⭐⭐ Clear arguments; the statistical hook in Figure 1 is powerful.
- **Value**: ⭐⭐⭐⭐⭐ In the long-CoT era, "trajectory-level safety" will likely become a standard paradigm; this work is a foundational reference for this direction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Internalizing Safety Understanding in Large Reasoning Models via Verification](internalizing_safety_understanding_in_large_reasoning_models_via_verification.md)
- [\[ICML 2026\] Watermarking LLM Agent Trajectories (ACTHOOK)](watermarking_llm_agent_trajectories.md)
- [\[ICML 2026\] Jailbreaking Vision-Language Models Through the Visual Modality](jailbreaking_vision-language_models_through_the_visual_modality.md)
- [\[ICML 2026\] Harnessing Reasoning Trajectories for Hallucination Detection via Answer-agreement Representation Shaping](harnessing_reasoning_trajectories_for_hallucination_detection_via_answer-agreeme.md)
- [\[ICML 2026\] Tracing the Dynamics of Refusal: Exploiting Latent Refusal Trajectories for Robust Jailbreak Detection](tracing_the_dynamics_of_refusal_exploiting_latent_refusal_trajectories_for_robus.md)

</div>

<!-- RELATED:END -->
