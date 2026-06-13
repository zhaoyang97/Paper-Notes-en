---
title: >-
  [Paper Note] Bad Seeing or Bad Thinking? Rewarding Perception for Vision-Language Reasoning
description: >-
  [ICML 2026][Multimodal VLM][VLM] This paper forcibly decomposes VLM outputs into `<recognition>` perception blocks and `<think>` reasoning blocks. It introduces a "blindfolded" text reasoning agent (which has no access t…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "VLM"
  - "RL"
  - "Modality Credit Assignment"
  - "GRPO"
  - "Structured Verbal Verification"
date: 2026-05-08
content_hash: fc2bb241585c9fec
---

# Bad Seeing or Bad Thinking? Rewarding Perception for Vision-Language Reasoning

**Conference**: ICML 2026 Oral  
**arXiv**: [2605.14054](https://arxiv.org/abs/2605.14054)  
**Code**: To be open-sourced (authors promise release of data/code/models)  
**Area**: Multimodal VLM / Vision-Language Reasoning / Reinforcement Learning  
**Keywords**: VLM, RL, Modality Credit Assignment, GRPO, Structured Verbal Verification

## TL;DR
This paper forcibly decomposes VLM outputs into `<recognition>` perception blocks and `<think>` reasoning blocks. It introduces a "blindfolded" text reasoning agent (which has no access to the image and only sees the VLM's perception text) to determine a perception reward $R_P$ based on answer correctness, paired with Structured Verbal Verification (SVV) as an outcome reward $R_O$. MoCA utilizes $R_P$ as a gate for modality-level credit assignment, allowing a 7B model to improve across 9 perception, reasoning, and rich-modality benchmarks simultaneously, outperforming GPT-4o on multiple metrics.

## Background & Motivation

**Background**: Advanced VLMs aim to achieve "perception-reasoning synergy" primarily through two approaches: (a) implicit fusion of visual tokens and text embeddings in latent space (e.g., Qwen-VL), handled by static text reasoning; (b) agentic "thinking with images" workflows (e.g., Pixel Reasoner, DeepEyes) that use multi-round function-calling to invoke external tools for active re-observation.

**Limitations of Prior Work**: (a) Static reasoning approaches fail when encountering fine-grained details; (b) Agentic workflows are engineering-heavy—involving multi-round RL, asynchronous long-tail episodes, and external tool integration—and often suffer from the "seesaw effect," where gains in perception metrics lead to drops in reasoning, and vice versa. These methods involve high investment with low returns due to inter-modality competition.

**Key Challenge**: The authors attribute the core issue to a neglected problem—**ambiguity in modality credit assignment**. When a VLM provides a wrong answer, is it due to "bad seeing" (incorrect visual evidence) or "bad thinking" (errors in the logical chain)? Existing training paradigms provide only outcome rewards, distributing penalties uniformly across the entire trajectory, making them unable to distinguish the source of error. Consequently, a model might "unlearn" correct perception behaviors due to reasoning failures.

**Goal**: (1) Extract "perception" from the latent black box into explicit, supervisable tokens; (2) Design a perception reward that does not rely on manual annotations or ground-truth captions; (3) Provide a low-variance, high-fidelity outcome verifier for free-form answers; (4) Decouple penalties for "seeing" and "thinking" via modality-level credit assignment.

**Key Insight**: The core observation is that **in explicit visual reasoning, the product of perception serves as the "discrete premises" required for logical derivation**. Therefore, the "sufficiency" of perception does not require a ground-truth caption; it can be tested by whether a text-only reasoner—given only the VLM's perception text and the question—can answer correctly. This approach bypasses the bottleneck of lacking perception labels.

**Core Idea**: Replace holistic outcome supervision with a framework consisting of a **"Blindfolded Reasoner + Structured Verbal Verification (SVV) + Modality-Aware Credit Assignment (MoCA),"** upgrading VLM training from vague end-to-end supervision to precise module-level credit assignment.

## Method

### Overall Architecture
The training objective is based on GRPO (Group Relative Policy Optimization). Under system prompt instructions, the VLM alternately outputs `<recognition>...</recognition>` (perception action $a_p$) and `<think>...</think>` (reasoning action). For each trajectory $\tau$, two rewards are computed: (i) **Perception Verification (PV)**: The $\{a_p\}$ and the question $Q$ are fed to a strong text-only reasoner (e.g., Qwen2.5-Instruct-14B, without the image) to see if it can produce the correct answer, yielding a binary $R_P\in\{0,1\}$; (ii) **Structured Verbal Verification (SVV)**: The same LLM executes a universal "verification protocol" (identifying answer type $\rightarrow$ extracting content $\rightarrow$ reconstructing reference $\rightarrow$ semantic comparison by type) to output an outcome reward $R_O$. The total return is $R(\tau)=R_O(\tau)+\lambda R_P(\tau)$, and the advantage is calculated via group normalization as $A_{\tau,t}=R(\tau)-\frac1k\sum R(\tau_j)$. **MoCA** reroutes the advantage on failed trajectories ($R_O=0$) based on $R_P$, delivering precise gradients for "seeing" vs. "thinking" to the corresponding tokens.

### Key Designs

1.  **Perception Verification via "Blindfolded Reasoner" (PV)**:
    - **Function**: Assigns a binary "sufficiency" reward to the VLM's `<recognition>` block without requiring perception ground truth.
    - **Mechanism**: All perception text $\{a_p\}$ written by the VLM is sent with the original question $Q$ to a text reasoner (blindfolded, image withheld). If the agent answers correctly, $R_P=1$; otherwise $R_P=0$. Theoretically, this is equivalent to an Information Bottleneck implementation $\min_{p(A_p|V)} I(V;A_p)-\beta I(A_p;Y)$, rewarding perception that is most informative for the answer and minimally redundant for the image. Explicit penalties are added for perception blocks exceeding 800 tokens to enforce minimalism.
    - **Design Motivation**: The essence of perception is to provide "discrete premises" for downstream reasoning. Thus, whether a proxy can independently complete the reasoning serves as a natural functional metric for perception sufficiency. This functional proxy requires no manual labels and is zero-tolerant toward "hallucinated captions"—no matter how detailed a caption is, it receives no reward if the information is incorrect. Human experiments show the PV matches human majority voting with an 86.31% agreement rate ($Cohen's \kappa=0.707$), with failures primarily being conservative False Negatives, which are mitigated by MoCA's protection mechanism.

2.  **Structured Verbal Verification (SVV)**:
    - **Function**: Replaces "rigid regex" and "subjective LLM judges" in free-form answer scenarios to provide low-variance, high-accuracy rewards for $R_O$.
    - **Mechanism**: Instead of asking an LLM judge "are these equivalent?", the judge is given a universal "verbal verification algorithm" and required to **execute it step-by-step**: (1) identify answer type (number/set/expression/multiple choice/free text), (2) extract content, (3) reconstruct reference form, (4) perform semantic comparison by type. An execution-based rather than estimation-based approach significantly reduces randomness.
    - **Design Motivation**: Rigid rules suffer from low recall on semantic paraphrasing. Subjective LLM judges have low consistency ($78.6\%$ at $T=0.7$ over five trials), making them vulnerable to reward hacking. Decomposing judgment into execution steps yields $92.3\%$ consistency, $91.9\%$ accuracy, and $92.7\%$ F1.

3.  **MoCA: Modality-Aware Credit Assignment**:
    - **Function**: Allocates penalties or protection to the "responsible" module based on $R_P$ when a trajectory fails ($R_O=0$).
    - **Mechanism**: Two modifications are applied to perception tokens $\tau_P$ in failed trajectories: **Case 1 (bad thinking)**: $R_O=0$ but $R_P=1$ (perception correct, reasoning wrong). A positive "protection" term is added to the advantage $A_{\tau,t}+\alpha_{\text{protect}}\cdot|A_{\tau,t}|$ to prevent gradients from destroying correct visual grounding. **Case 2 (bad seeing)**: $R_O=0$ and $R_P=0$ (perception also wrong). The penalty is amplified to $A_{\tau,t}-\alpha_{\text{punish}}\cdot|A_{\tau,t}|$. Reasoning tokens follow the standard GRPO process.
    - **Design Motivation**: Standard GRPO applies negative advantages uniformly to all tokens in a failed trajectory. This is the root of the seesaw effect—correct perception paired with failed reasoning leads to the unlearning of correct perception. MoCA's gate mechanism maps modality error sources to module gradients, and the protection term inherently buffers against False Negative noise from the PV oracle.

### Loss & Training
The framework is built on GRPO with a group baseline, using the modality-aware advantage modifications. Qwen2.5-VL-Instruct-7B serves as the base model. Both the PV reasoner and SVV judge use Qwen2.5-Instruct-14B. Training data is a mixture of ViRL39K (STEM reasoning), VisualWebInstruct-Verified (general visual instructions), Pixel Reasoner data (perception-intensive), and rich-modality data crawled from arXiv, newspapers, and infographics.

## Key Experimental Results

### Main Results
Across 9 benchmarks (perception-intensive, rich-modality, and reasoning-intensive), MoCA-7B achieves universal improvements, surpassing GPT-4o on several metrics.

| Model | V* | HRBench | DUDE | MMLong | MMMU | EMMA | MathVista |
|------|-----|---------|------|--------|------|------|-----------|
| GPT-4o | 45.0 | 65.0 | 52.7 | 42.3 | 51.9 | 32.7 | 63.4 |
| Qwen2.5-VL-Instruct 72B | 81.2 | 73.4 | 44.5 | 24.9 | 67.0 | 38.5 | 74.8 |
| Qwen2.5-VL-Instruct 7B (base) | 71.4 | 69.2 | 41.8 | 21.2 | 54.3 | 21.5 | 68.2 |
| Pixel Reasoner 7B | 84.3 | 72.8 | 44.5 | 22.0 | 50.8 | 19.8 | 65.3 |
| DeepEyes 7B | 88.9 | 73.1 | 35.2 | 17.5 | 45.2 | 18.1 | 64.9 |
| **MoCA 7B** | **86.6** | **74.2** | **45.1** | **33.1** | **54.8** | **31.3** | **73.8** |

Highlights: Relative to the base model, MoCA gains $+15.2$ on V*, $+5.0$ on HRBench, $+3.3$ on DUDE, $+11.9$ on MMLong, and $+9.8$ on EMMA—improving across all 9 benchmarks without triggering the seesaw effect.

### Ablation Study

| Configuration | V* | HRBench | DUDE | MMMU | MathVista | Description |
|------|-----|---------|------|------|-----------|------|
| Full MoCA | 86.6 | 74.2 | 45.1 | 54.8 | 73.8 | Full model |
| Instruction-Only (No RL) | 68.3 | 66.5 | 37.7 | 49.9 | 65.7 | Forced decomposition via prompt alone (performance drops) |
| w/o PV (Only $R_O$) | 79.7 | 70.1 | 42.5 | 55.3 | 74.4 | Perception tasks drop significantly (V* -6.9) |
| w/o MoCA ($R_O+\lambda R_P$ naive sum) | 83.1 | 72.5 | 43.7 | 54.6 | 74.1 | Gating is necessary; approx. -3 drop |
| w/o SVV+PV (Standard LLM Judge) | 78.4 | 69.7 | 38.9 | 52.3 | 72.1 | High-variance rewards lead to hacking |

PV oracle vs. human majority (N=979): 86.31% accuracy, $Cohen's \kappa=0.707$ (substantial agreement). SVV on VP-Challenge-Set (N=273): 91.9% accuracy, 92.7% F1, 92.3% consistency, outperforming both Rigid Rule and standard LLM Prompting.

### Key Findings
- **PV is the primary source of gains for perception tasks**: Removing $R_P$ leads to significant drops solely on perception-intensive benchmarks (e.g., V* -6.9), while reasoning tasks remain largely unchanged, indicating that PV provides a highly targeted reward signal.
- **The MoCA gate is not just a minor addition**: Naive reward summation still results in a ~3 point drop on perception tasks because negative gradients from reasoning tokens in failed trajectories contaminate perception tokens. The gate mechanism is essential to eliminate the seesaw effect.
- **Structured execution > Subjective judgment**: SVV increases consistency from 78.6% to 92.3% compared to standard LLM judges, preventing instability from reward hacking.
- **MoCA enables the 7B model to outperform GPT-4o and Qwen2.5-VL-72B on multiple metrics**: It surpasses the 72B version of its own lineage on DUDE (45.1 vs 44.5) and HRBench (74.2 vs 73.4), proving that paradigm advantages can compensate for parameter scale deficiency.

## Highlights & Insights
- The "blindfolded reasoning proxy" resolves the no-label perception supervision dilemma: functional sufficiency testing requires no captions or external tools, just a text LLM, making it inexpensive and theoretically supported by IB.
- It internalizes perception-reasoning synergy from an "external agentic workflow" into "alternating blocks within a single autoregressive generation," avoiding the pitfalls of multi-round RL and asynchronous engineering—a rare simplified implementation of complex capabilities.
- The MoCA gate mechanism essentially reduces "trajectory-level coarse signals" into "token-segment-level precise signals." This decoupled credit assignment approach can be generalized to training any modular policy (e.g., code, tool, or memory modules).

## Limitations & Future Work
- Text reasoners are "near-sighted"—certain visual features requiring spatial relationships (e.g., complete mazes, complex geometric relations) are difficult to compress into text, exceeding the coverage of the System-2 hypothesis used here.
- The PV oracle’s 9.19% False Negative rate may "wrongly accuse" good perception. Although MoCA's protection buffers most negative impacts, it still relies on an imperfect oracle whose scale/capability limits the model's ceiling.
- The 800-token perception limit is anecdotal and may be too tight for truly complex scenes (multi-page documents, multiple images); a more dynamic sufficiency measure is needed.
- Both training and inference depend on a second 14B reasoner, nearly doubling the deployment cost compared to a standalone 7B VLM.

## Related Work & Insights
- **vs. Pixel Reasoner / DeepEyes (agentic)**: These rely on external tool calls for active perception, requiring multi-round RL and tool protocols. MoCA internalizes the same "see-think" loop into alternating blocks in a single generation, increasing efficiency by an order of magnitude.
- **vs. VL-Rethinker / R1-VL (RL-based VLM)**: These use a single outcome reward + GRPO and suffer from the seesaw effect. MoCA resolves credit assignment ambiguity through modular rewards and gating, achieving consistent improvements across tasks.
- **vs. RLHF / DPO**: Traditional alignment is outcome-only. MoCA draws on the spirit of process supervision but solves the difficulty of obtaining process labels through a functional proxy, which can be transferred to other scenarios where process labels are scarce (e.g., code, agents).

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The "blindfolded reasoner" as a functional proxy for perception sufficiency is ingenious.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 9 benchmarks, component ablations, human validation, and verifier comparisons.
- Writing Quality: ⭐⭐⭐⭐ The logic from problem definition to IB theory and MoCA gating is flow-oriented; framework diagrams are clear.
- Value: ⭐⭐⭐⭐⭐ Provides a universal recipe for "internalizing agentic capabilities + modality-level RL," potentially resetting the VLM training paradigm.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] From Seeing to Thinking: Decoupling Perception and Reasoning Improves Post-Training of Vision-Language Models](from_seeing_to_thinking_decoupling_perception_and_reasoning_improves_post-traini.md)
- [\[ICML 2026\] 3ViewSense: Spatial and Mental Perspective Reasoning from Orthographic Views in Vision-Language Models](3viewsense_spatial_and_mental_perspective_reasoning_from_orthographic_views_in_v.md)
- [\[ICML 2026\] Efficient Reasoning with Hidden Thinking](efficient_reasoning_with_hidden_thinking.md)
- [\[ICML 2026\] Active Exploring like a Pigeon: Reinforcing Spatial Reasoning via Agentic Vision-Language Models](active_exploring_like_a_pigeon_reinforcing_spatial_reasoning_via_agentic_vision-.md)
- [\[ICML 2026\] Med-Scout: Curing MLLMs' Geometric Blindness in Medical Perception via Geometry-Aware RL Post-Training](med-scout_curing_mllms_geometric_blindness_in_medical_perception_via_geometry-aw.md)

</div>

<!-- RELATED:END -->
