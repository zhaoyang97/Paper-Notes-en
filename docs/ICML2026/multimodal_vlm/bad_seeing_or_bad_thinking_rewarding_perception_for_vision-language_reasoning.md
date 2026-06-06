---
title: >-
  [Paper Note] Bad Seeing or Bad Thinking? Rewarding Perception for Vision-Language Reasoning
description: >-
  [ICML 2026][Multimodal VLM][VLM] This work enforces VLM outputs to be split into `<recognition>` perception blocks and `<think>` reasoning blocks…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "VLM"
  - "RL"
  - "Modal Credit Assignment"
  - "GRPO"
  - "Structured Verbal Verification"
date: 2026-05-08
content_hash: 3f3d3e6370392db7
---

# Bad Seeing or Bad Thinking? Rewarding Perception for Vision-Language Reasoning

**Conference**: ICML 2026  
**arXiv**: [2605.14054](https://arxiv.org/abs/2605.14054)  
**Code**: To be released (authors promise to release data/code/models)  
**Area**: Multimodal VLM / Vision-Language Reasoning / Reinforcement Learning  
**Keywords**: VLM, RL, Modal Credit Assignment, GRPO, Structured Verbal Verification

## TL;DR
This work enforces VLM outputs to be split into `<recognition>` perception blocks and `<think>` reasoning blocks, then uses a "blindfolded" text-only reasoning agent (which cannot access the image, only the perception text written by the VLM) to determine if the question can be answered correctly, serving as the perception reward $R_P$, paired with structured verbal verification (SVV) as the outcome reward $R_O$. MoCA uses $R_P$ as a gate for modality-level credit assignment, enabling a 7B model to achieve simultaneous improvements across 9 perception/reasoning/rich-modality benchmarks, surpassing GPT-4o on multiple metrics.

## Background & Motivation

**Background**: Advanced VLMs aim for "perception-reasoning synergy," mainly following two paths: (a) models like Qwen-VL that implicitly fuse visual tokens and text embeddings in latent space, relying on static text-based reasoning; (b) agentic workflows such as Pixel Reasoner and DeepEyes, which use multi-round function-calling to actively re-observe images via external tools.

**Limitations of Prior Work**: (a) The first approach is limited by static reasoning and fails on fine-grained details; (b) The second is engineering-heavy—requiring multi-round RL, asynchronous long-tail episodes, and external tool integration—and often suffers from the notorious "seesaw effect": perception metrics improve at the expense of reasoning, and vice versa. High investment, low return, and intense competition between modalities.

**Key Challenge**: The authors attribute the core issue to a neglected root problem—**ambiguity in modality credit assignment**. When a VLM makes a mistake, is it due to "bad seeing" (misinterpreting visual evidence) or "bad thinking" (faulty logic)? Existing training paradigms only provide outcome rewards, distributing penalties uniformly across the entire trajectory, making it impossible to distinguish the source of error. As a result, the model may "unlearn" correct perception due to reasoning failures, and vice versa.

**Goal**: (1) Extract "perception" from the latent black box into explicit, supervised tokens; (2) Design a perception reward that does not rely on human annotation or ground-truth captions; (3) Provide a low-variance, high-fidelity outcome verifier for free-form answers; (4) Separate penalties for "bad seeing" and "bad thinking" via modality-level credit assignment.

**Key Insight**: The core observation is that **in explicit visual reasoning, the product of perception is the set of discrete premises required for logical inference**. Thus, perception sufficiency does not require ground-truth captions; it suffices to test whether a pure text reasoner (given only the VLM's perception text and the question, without the image) can answer correctly. If so, perception is sufficient; otherwise, it is "bad seeing." This approach circumvents the lack of perception labels.

**Core Idea**: Replace holistic outcome supervision with **"Blindfolded Reasoner + Structured Verbal Verification (SVV) + Modality-Aware Credit Assignment (MoCA)"**, upgrading VLM training from end-to-end fuzzy supervision to module-level precise credit assignment.

## Method

### Overall Architecture
The training objective is based on GRPO (Group Relative Policy Optimization). Under system prompt instructions, the VLM alternately outputs `<recognition>...</recognition>` (perception action $a_p$) and `<think>...</think>` (reasoning action). For each trajectory $\tau$, two rewards are computed: (i) **Perception Verification (PV)**: Feed $\{a_p\}$ and question $Q$ to a strong text-only reasoner (e.g., Qwen2.5-Instruct-14B, without the image) to check if it can produce the correct answer, yielding a binary $R_P\in\{0,1\}$; (ii) **Structured Verbal Verification (SVV)**: The same LLM executes a general "verification protocol" step-by-step (identify answer type → extract content → reconstruct reference → semantic comparison by type), outputting the outcome reward $R_O$. The total return is $R(\tau)=R_O(\tau)+\lambda R_P(\tau)$, with advantage normalized within the group: $A_{\tau,t}=R(\tau)-\frac1k\sum R(\tau_j)$. **MoCA** reroutes the advantage on failed trajectories ($R_O=0$) based on $R_P$, precisely directing gradients for "bad seeing" and "bad thinking" to the corresponding tokens.

### Key Designs

1. **Perception Verification via "Blindfolded Reasoner" (PV)**:

    - **Function**: Provides a binary "sufficiency" reward for the VLM's `<recognition>` block without perception ground truth.
    - **Mechanism**: All perception texts $\{a_p\}$ written by the VLM, together with the original question $Q$, are given to a pure text reasoner (blindfolded, i.e., image withheld). If the agent answers correctly, $R_P=1$; otherwise, $R_P=0$. Theoretically, this is equivalent to the Information Bottleneck objective $\min_{p(A_p|V)} I(V;A_p)-\beta I(A_p;Y)$—rewarding perception expressions that are maximally informative for the answer and minimally redundant for the image. To enforce minimalism, an explicit penalty is applied to perception blocks exceeding 800 tokens.
    - **Design Motivation**: The essence of perception is to provide "discrete premises" for downstream reasoning, so "whether the agent can independently reason" is a natural functional indicator of perception sufficiency. This functional proxy requires no human annotation and is intolerant of "hallucinated captions"—no reward for beautiful but uninformative captions. Human annotation experiments on 979 samples show PV achieves 86.31% agreement with human majority, Cohen's κ=0.707, with failure modes dominated by "overly conservative false negatives" (9.19% vs 4.49% FP)—FN is buffered by MoCA's protect mechanism, ensuring safety.

2. **Structured Verbal Verification (SVV)**:

    - **Function**: Replaces "rigid regex" and "subjective LLM judge" in free-form answer scenarios, providing $R_O$ with low-variance, high-accuracy rewards.
    - **Mechanism**: Instead of subjective equivalence judgments by LLMs, a general "language-based verification algorithm" is provided, requiring stepwise execution: (1) identify answer type (number / set / expression / multiple choice / free text), (2) extract content, (3) reconstruct reference form, (4) perform semantic comparison by type. This execution-based approach greatly reduces randomness.
    - **Design Motivation**: Rigid rules suffer drastic recall drops on semantic rewrites; subjective LLM judges achieve only 78.6% consistency under $T=0.7$ with five repetitions, and are vulnerable to reward hacking during RL training. Decomposing judgment into execution steps raises consistency to 92.3%, accuracy to 91.9%, and F1 to 92.7% (VP-Challenge-Set, N=273).

3. **MoCA: Modality-Aware Credit Assignment**:

    - **Function**: On failed trajectories ($R_O=0$), assigns penalties/protection to the responsible module based on $R_P$.
    - **Mechanism**: For perception tokens $\tau_P$ in failed trajectories, two corrections are applied—**Case 1 (bad thinking)**: $R_O=0$ but $R_P=1$ (perception correct, reasoning wrong), add a positive "protection" term $A_{\tau,t}+\alpha_{\text{protect}}\cdot|A_{\tau,t}|$ to the advantage of perception tokens, preventing gradients from destroying correct visual grounding; **Case 2 (bad seeing)**: $R_O=0$ and $R_P=0$ (perception also wrong), amplify the penalty for perception tokens as $A_{\tau,t}-\alpha_{\text{punish}}\cdot|A_{\tau,t}|$. Reasoning tokens follow the standard GRPO process.
    - **Design Motivation**: Standard GRPO applies negative advantage uniformly to all tokens on failed trajectories, which is the root of the seesaw effect—when perception is correct but reasoning fails, correct perception is unlearned. MoCA's gate mechanism aligns modality errors with module gradients, and the protection term naturally buffers PV oracle's false negative noise (even if the oracle misjudges $R_P=0$, protection is simply not triggered, avoiding harm to perception ability).

### Loss & Training

The foundation is GRPO with group baseline, with advantage modified as above for modality-aware perception. Qwen2.5-VL-Instruct-7B serves as the base; both the PV reasoner and SVV judge use Qwen2.5-Instruct-14B (dual use of the same model for deployment efficiency). Training data is a mixture: ViRL39K (STEM reasoning), VisualWebInstruct-Verified (general visual instructions), Pixel Reasoner data (perception-intensive), and rich-modality data crawled by the authors from arXiv/newspapers/infographics.

## Key Experimental Results

### Main Results

Across 9 benchmarks (perception-intensive / rich-modality / reasoning-intensive), MoCA-7B achieves consistent improvements, surpassing GPT-4o on multiple metrics.

| Model | V* | HRBench | DUDE | MMLong | MMMU | EMMA | MathVista |
|-------|-----|---------|------|--------|------|------|-----------|
| GPT-4o | 45.0 | 65.0 | 52.7 | 42.3 | 51.9 | 32.7 | 63.4 |
| Qwen2.5-VL-Instruct 72B | 81.2 | 73.4 | 44.5 | 24.9 | 67.0 | 38.5 | 74.8 |
| Qwen2.5-VL-Instruct 7B (base) | 71.4 | 69.2 | 41.8 | 21.2 | 54.3 | 21.5 | 68.2 |
| Pixel Reasoner 7B | 84.3 | 72.8 | 44.5 | 22.0 | 50.8 | 19.8 | 65.3 |
| DeepEyes 7B | 88.9 | 73.1 | 35.2 | 17.5 | 45.2 | 18.1 | 64.9 |
| **MoCA 7B** | **86.6** | **74.2** | **45.1** | **33.1** | **54.8** | **31.3** | **73.8** |

Highlights: Relative to the base, V* +15.2, HRBench +5.0, DUDE +3.3, MMLong +11.9, EMMA +9.8—improvements across all 9 benchmarks, with no seesaw effect observed.

### Ablation Study

| Configuration | V* | HRBench | DUDE | MMMU | MathVista | Notes |
|---------------|-----|---------|------|------|-----------|-------|
| Full MoCA | 86.6 | 74.2 | 45.1 | 54.8 | 73.8 | Complete model |
| Instruction-Only (no RL) | 68.3 | 66.5 | 37.7 | 49.9 | 65.7 | Prompt-based forced decomposition only, performance drops |
| w/o PV (only $R_O$) | 79.7 | 70.1 | 42.5 | 55.3 | 74.4 | Perception tasks drop sharply (V* -6.9, HRBench -4.1) |
| w/o MoCA ($R_O+\lambda R_P$ naive sum) | 83.1 | 72.5 | 43.7 | 54.6 | 74.1 | Gate mechanism is essential, about -3 points |
| w/o SVV+PV (standard LLM Judge) | 78.4 | 69.7 | 38.9 | 52.3 | 72.1 | High-variance reward leads to hacking |

PV oracle vs human majority (N=979): accuracy 86.31%, Cohen's κ=0.707 (substantial agreement). SVV on VP-Challenge-Set (N=273): accuracy 91.9%, F1 92.7%, consistency 92.3%, comprehensively surpassing Rigid Rule (67/58.7/100) and LLM Prompting (79.1/82.4/78.6).

### Key Findings
- **PV is the main source of gains for perception tasks**: Removing $R_P$ causes significant drops only on perception-intensive benchmarks (V* -6.9), with little effect on reasoning tasks, indicating PV's reward signal is well-targeted.
- **MoCA gate is essential**: Naively summing rewards still results in a ~3 point drop on perception tasks, as negative gradients from reasoning tokens in failed trajectories contaminate perception tokens; the gate mechanism is necessary to eliminate the seesaw effect.
- **Structured execution > subjective judgment**: SVV vs standard LLM Judge raises consistency from 78.6→92.3, avoiding reward hacking instability.
- **MoCA enables the 7B model to surpass GPT-4o and Qwen2.5-VL-72B on multiple metrics**: Outperforms the 72B model on DUDE (45.1 vs 44.5) and HRBench (74.2 vs 73.4), demonstrating that the paradigm advantage can compensate for smaller model size.

## Highlights & Insights
- The "blindfolded reasoner" breaks the label-free perception supervision dilemma in one move: functional sufficiency testing requires no caption annotation or external tools, only a text LLM, making it extremely cost-effective and theoretically grounded in IB.
- Perception-reasoning synergy is internalized from "external agentic workflows" to "single-pass autoregressive alternation," avoiding all the pitfalls of multi-round RL and asynchronous engineering—a rare "complex capability, simplified implementation."
- MoCA's gate mechanism essentially reduces "trajectory-level coarse signals" to "token-segment-level precise signals"; this decoupled credit assignment approach can be generalized to any modular policy training (e.g., code/tool/memory modules).

## Limitations & Future Work
- The text reasoner is "near-sighted"—certain visual features that require spatial relationships (e.g., complete mazes, complex geometry) are hard to compress into text, exceeding the coverage of the System-2 assumption; the paper acknowledges this as future work.
- PV oracle's 9.19% false negative rate may "penalize good perception," though MoCA's protect mechanism buffers most negative effects; ultimately, reliance on an imperfect oracle means the oracle's scale/capability limits the final ceiling.
- The 800-token perception length limit is empirical and may be too restrictive for truly complex scenarios (multi-page documents, multiple images); more dynamic sufficiency metrics are needed in the future.
- Both training and inference depend on a second 14B reasoner, nearly doubling deployment cost compared to a pure 7B VLM.

## Related Work & Insights
- **vs Pixel Reasoner / DeepEyes (agentic)**: These rely on external tool calls for active perception, requiring multi-round RL and tool protocols; MoCA internalizes the same "see-think" loop into alternating blocks within a single autoregressive generation, achieving an order-of-magnitude efficiency gain.
- **vs VL-Rethinker / R1-VL (RL-based VLM)**: These use a single outcome reward + GRPO, suffering from the seesaw effect; MoCA resolves credit assignment ambiguity with module rewards + gate, achieving consistent multi-task improvements.
- **vs RLHF / DPO**: Traditional alignment is also outcome-only; MoCA draws on process supervision concepts, but solves the "process label scarcity" problem via functional proxy, making it transferable to code, agent, and other process-label-scarce scenarios.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The "blindfolded reasoner" as a functional proxy for perception sufficiency is extremely ingenious
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 9 benchmarks, component ablations, human validation, verifier comparisons—all included
- Writing Quality: ⭐⭐⭐⭐ Problem definition → IB theory → MoCA gate logic is very coherent, with clear framework diagrams
- Value: ⭐⭐⭐⭐⭐ Provides a general recipe for "internalizing agentic capability + modality-level RL," potentially resetting VLM training paradigms

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] ChemVLR: Prioritizing Reasoning in Perception for Chemical Vision-Language Understanding](../../ACL2026/multimodal_vlm/chemvlr_prioritizing_reasoning_in_perception_for_chemical_vision-language_unders.md)
- [\[CVPR 2026\] Seeing Clearly, Reasoning Confidently: Plug-and-Play Remedies for Vision Language Model Blindness](../../CVPR2026/multimodal_vlm/seeing_clearly_reasoning_confidently_plug-and-play_remedies_for_vision_language_.md)
- [\[ICLR 2026\] Seeing Across Views: Benchmarking Spatial Reasoning of Vision-Language Models in Robotic Scenes](../../ICLR2026/multimodal_vlm/seeing_across_views_benchmarking_spatial_reasoning_of_vision-language_models_in_.md)
- [\[ACL 2026\] Addressing Overthinking in Large Vision-Language Models via Gated Perception-Reasoning Optimization](../../ACL2026/multimodal_vlm/addressing_overthinking_in_large_vision-language_models_via_gated_perception-rea.md)
- [\[CVPR 2026\] CropVLM: Learning to Zoom for Fine-Grained Vision-Language Perception](../../CVPR2026/multimodal_vlm/cropvlm_learning_to_zoom_for_fine_grained_vision_language_perception.md)

</div>

<!-- RELATED:END -->
