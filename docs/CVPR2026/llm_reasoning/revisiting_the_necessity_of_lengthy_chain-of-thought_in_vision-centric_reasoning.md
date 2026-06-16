---
title: >-
  [Paper Note] Revisiting the Necessity of Lengthy Chain-of-Thought in Vision-centric Reasoning Generalization
description: >-
  [CVPR 2026][LLM Reasoning][SFT-then-RL] The authors systematically compare three "think with image" supervision formats—Language CoT, Grounding CoT, and Visual CoT—using a controlled maze navigation task. They find that longer or more elaborate Visual CoTs only accelerate convergence without raising the final performance ceiling. Conversely, a **minimalist C
tags:
  - CVPR 2026
  - LLM Reasoning
  - SFT-then-RL
date: 2026-05-08
content_hash: 7fcf3e77354a3377
---
# Revisiting the Necessity of Lengthy Chain-of-Thought in Vision-centric Reasoning Generalization

**Conference**: CVPR 2026  
**Paper**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Du_Revisiting_the_Necessity_of_Lengthy_Chain-of-Thought_in_Vision-centric_Reasoning_Generalization_CVPR_2026_paper.html)  
**Code**: https://github.com/RUCAIBox/Revisiting-Visual-CoT  
**Area**: LLM Reasoning / Multimodal VLM  
**Keywords**: Visual Chain-of-Thought, Visual CoT, Generalization, SFT-then-RL, Maze Navigation

## TL;DR
The authors systematically compare three "think with image" supervision formats—Language CoT, Grounding CoT, and Visual CoT—using a controlled maze navigation task. They find that longer or more elaborate Visual CoTs only accelerate convergence without raising the final performance ceiling. Conversely, a **minimalist CoT preserving only essential grounding information (a single coordinate path) achieves the best generalization**. The paper proposes the "short is long" effect and provides a practical guide for constructing generalizable visual reasoning SFT data.

## Background & Motivation
**Background**: Visual reasoning is becoming a critical capability for Vision-Language Models (VLMs). The industry commonly uses CoT data for supervised fine-tuning (SFT) to teach models to "think before answering." The mainstream belief is "the longer, the better"—longer CoTs provide multi-step deduction and self-reflection, while o3-style Visual CoT (cropping, drawing, and labeling on images before feeding them back) is considered closer to human visual cognition and capable of further improving various visual reasoning benchmarks.

**Limitations of Prior Work**: Most of these conclusions are derived from real-world benchmarks prone to interference from pre-training priors and data contamination. **It remains unclear which CoT design is effective, why it works, and which type truly supports "generalizable" reasoning**. Language, spatial coordinates, and image operations are fundamentally different mechanisms for "externalizing" intermediate reasoning, yet they are often lumped together under the assumption that "adding CoT improves performance."

**Key Challenge**: There is not necessarily a positive correlation between "enriching supervision signals (longer, image-based operations)" and "enabling the model to learn transferable abstract rules." Rich trajectories might merely help the model fit a specific layout more quickly rather than internalizing scale-invariant navigation laws.

**Goal**: In a clean, controlled, and difficulty-adjustable environment, the authors decouple language, grounding, and visual CoT to answer: ① What specific benefits does each provide? ② Through what capability does CoT function in vision-centric tasks? ③ Which format generalizes best?

**Key Insight**: Maze navigation is selected as the testbed because its reasoning rules are entirely expressed by visual input, its difficulty is smoothly adjustable by grid size (from $4\times4$ to $10\times10$), current VLMs perform poorly on it (Qwen2.5-VL-7B success rate $<10\%$ on $4\times4$, meaning it won't be masked by pre-trained saturated capabilities), and both solutions and intermediate steps can be automatically synthesized and filtered via rule functions, naturally avoiding data contamination.

**Core Idea**: Under a unified SFT-then-RL pipeline, the study fairly compares four CoT formats, using "cross-maze-size generalization" rather than "training set success rate" as the criterion for excellence. It was ultimately discovered that **stripping CoT down to minimal grounding information is most conducive to generalization**.

## Method

### Overall Architecture
This is a **mechanism analysis and data construction study** rather than a proposal for a new model. The "method" is a rigorously controlled experimental protocol: using Qwen2.5-VL-7B as the unified base, 8K cold-start trajectories are synthesized for each of the four CoT formats for SFT. Subsequently, the models are trained to convergence (up to 1000 steps) using RL (GRPO) on maze data, then tested for generalization on **unseen larger mazes**. The pipeline consists of: "Rule-based maze synthesis → CoT trajectory formatting → Independent SFT for each format to obtain policy models → RLVR reinforcement → Cross-scale generalization evaluation." The four formats differ only in how the "intermediate reasoning is externalized," isolating the impact of format on learning and generalization.

The input consists of an $N\times N$ maze image $I$ and an instruction $Q$ (requiring a coordinate path from start $S$ to end $E$ without passing through walls, with the final path placed in `\boxed{}`). The output is the model-generated reasoning process `<think>…</think>` plus the path. Walls are defined between adjacent cells rather than occupying cells; the path must satisfy the condition that no wall exists between two adjacent cells $w_{(i_k,j_k)\to(i_{k+1},j_{k+1})}=0$.

### Key Designs

**1. Four CoT Formats: Isolating "Reasoning Externalization" as the Sole Variable**

This is the experimental backbone of the paper, addressing the pain point that "prior work conflates different CoTs." The formats are ordered from "verbose" to "minimalist":

- **Language CoT (L-CoT)**: Pure text, using "north/south/west/east" to describe each step. Trajectories $R^{lang}_T=r^{(l)}_1,\dots,r^{(l)}_T$ where $r^{(l)}_t\in V_{text}$. A rule function converts paths to directional sequences, followed by Gemini-2.5-Pro synthesizing natural language reasoning.
- **Grounding CoT (G-CoT)**: Explicitly binds language references to spatial coordinates on the image at each step. Elements are represented as $g_k=(G_k,C_k)$ where $G_k\in\{point,line,region\}$. **Reflection patterns** (deliberately creating incorrect paths hitting walls/dead ends + error correction reasoning) are injected during synthesis to deepen reasoning.
- **Visual CoT (V-CoT)**: Allows "image modification" on top of grounding—using line-drawing operations $I_{t+1}=\phi_t(I_t,g_t)$ to render the current partial path onto the image, which is then fed back to the model, forming interleaved image-text reasoning.
- **G-CoT-least (Minimal Grounding)**: Directly uses the final path coordinate sequence as the answer, providing no extra text explanation or absolute coordinate systems. Since the target output of the maze task is itself a sequence of visited grid points, **reasoning is implicitly embedded in the path**. This represents the extreme of "minimal grounding information."

By placing all four in the same SFT-then-RL pipeline, the study asks whether more externalization is better.

**2. SFT-then-RL Training Protocol: Cold-start Shaping followed by Verifiable Reward Reinforcement**

This addresses the issue that "current VLMs cannot even generate decent maze logic, causing direct RL to collapse." The process has two stages: SFT wraps synthesized reasoning in `<think></think>` and answers in `\boxed{}`, with 8K samples per format. Visual CoT uses interleaved data where cross-entropy is only calculated on text tokens. The RL stage synthesizes an additional 20K maze samples, optimized using GRPO with the reward:

$$r=\alpha\cdot r_{acc}+(1-\alpha)\cdot r_{format},\quad \alpha=0.9$$

where $r_{acc}$ is determined by a rule function verifying if the predicted path connects the start and end without hitting walls, and $r_{format}$ constrains the output format. A key methodological contribution is **training to true convergence**: while prior visual RL work often trains for only a few hundred steps, leading to undertraining, this study trains all models to 1000 steps to compare "final ceilings" rather than "early speeds." The vision encoder is frozen during SFT and unfrozen during RL.

**3. Using "Cross-scale Generalization" rather than "Training Success Rate" as the Criterion: Revealing "Short is Long"**

This is the criterion design that establishes the conclusion. The limitation is that "training sets can always be pushed to 100%, making it impossible to see who truly learned the rules." The authors examine two types of generalization: **single-scale generalization** (SFT+RL on $6\times6$, testing on unseen $7\times7$) and **cross-scale generalization** (SFT on $4\times4$–$6\times6$, RL on $7\times7$–$9\times9$, testing on unseen $10\times10$). The results show G-CoT-least robustly maintains high success rates, while V-CoT saturates after ~800 steps and lags behind. Mechanism explanation: minimalist grounding forces the model to internalize **scale-invariant local navigation rules** (follow hallways, backtrack at dead ends), whereas Visual CoT tends to overfit specific visual layouts and operation patterns. This leads to "short is long"—concise but well-grounded supervision is better for learning reusable reasoning patterns than verbose, heavy supervision.

### Loss & Training
SFT uses standard cross-entropy (V-CoT only calculates loss on text tokens). RL uses GRPO with the reward formula mentioned above ($\alpha=0.9$). SFT: 3 epochs, learning rate $1\times10^{-5}$, warm-up ratio 0.1, batch 64. RL: rollout batch 128, mini-batch 32, 8 rollouts per sample, trained to convergence ($\le1000$ steps).

## Key Experimental Results

### Main Results
The core conclusions stem from training dynamics on mazes (Figures 2–5) and cross-task validation (Table 1). Three key observations on mazes:

| Observation Dimension | L-CoT | G-CoT | V-CoT | G-CoT-least |
|---------|-------|-------|-------|-------------|
| RL Convergence Speed | Slowest | Medium | Fast (≈half the steps of L-CoT) | **Fastest**, exceeding V-CoT |
| Final Train Success Rate | →100% | →100% | →100% | →100% (never saw explicit coords) |
| $7\times7$ Unseen Gen. | Average | Good | Saturates @ 800 steps, lower | **Best and stable** |

Key Point: Visual / Longer CoT only **accelerates convergence, not the ceiling**; G-CoT-least, stripped to minimal grounding, starts higher, converges faster, and generalizes strongest.

Extrapolating conclusions to other vision-centric tasks (Table 1, Accuracy %):

| Model | V*Bench Overall | HR-Bench 4K Overall | FrozenLake | Jigsaw |
|------|------|------|------|------|
| Qwen2.5-VL-7B | 72.25 | 72.50 | 20.00 | 0.00 |
| + V-CoT RL | 83.25 | 72.00 | - | - |
| + G-CoT-least RL | **85.86** | **74.12** | **90.33** | **75.60** |

Jigsaw improved from 0% to 70%+, FrozenLake from 20% to 90%+; on V*Bench / HR-Bench real high-res VQA, G-CoT-least (without cropping or drawing) outperformed explicit V-CoT across the board, proving models can perform visual reasoning **implicitly**.

### Ablation Study
The structure of this paper is unique—it is essentially a set of "ablation-style" comparisons, where the CoT format is the variable being ablated or replaced:

| Configuration | Key Phenomenon | Explanation |
|------|---------|------|
| Zero RL (No SFT cold-start) | Training collapse | Proves SFT cold-start is necessary to stabilize RL |
| L-CoT / G-CoT / V-CoT | Train sets all →100%, but ceilings are similar | Visual/Longer CoT is fast but not stronger |
| G-CoT → G-CoT-least | Higher start, faster convergence, still hits 100% | Removing explicit coordinate systems is actually better |
| V-CoT vs G-CoT-least (Cross-scale) | V-CoT saturates at 800 steps, lags behind | Minimal grounding generalizes better |

### Key Findings
- **"short is long"**: The biggest contribution is not a specific module but the counter-intuitive choice to "strip grounding information to its minimum." This avoids overfitting to specific coordinate systems/layouts and provides a more compact, transferable inductive bias.
- **Mechanism**: In vision-centric tasks, RL primarily reinforces the model's existing **grounding capability**. Once grounding is aligned with the visual environment, the model can complete tasks with extremely short CoT or even implicit reasoning without needing to output coordinates or modify images.
- **Cold-start is indispensable**: RL from scratch fails; SFT shapes the policy space first to alleviate exploration and reward sparsity issues.

## Highlights & Insights
- **Isolating variables using controlled mazes** is clever: pure visual rules, adjustable difficulty, and automatically synthesizable solutions suppress the major interference factors—"data contamination" and "pre-training priors"—allowing the impact of "format" to be cleanly measured for the first time. This testbed approach is transferable to any mechanism analysis in visual reasoning.
- **Training to true convergence** is a frequently overlooked methodological detail. Many conclusions claiming "Visual CoT is stronger" are actually comparing early speeds in an undertrained stage. This paper shows that once trained to 1000 steps, the speed advantage disappears and the ceilings converge, reminding the community to control for training sufficiency when comparing RL methods.
- **Minimal grounding as an inductive bias** directly informs SFT data construction: instead of stacking long CoTs, it is better to provide a clean grounded answer and let RL reinforce the model's own implicit spatial representations.

## Limitations & Future Work
- The authors acknowledge that validation was primarily on maze-like tasks ("vision-centric + automatically synthesizable rules"). While extrapolated to FrozenLake / Jigsaw / V*Bench, they plan to expand to richer task families and more VLMs.
- The conclusion depends on the task structure where "the answer itself is a grounding sequence" (mazes/paths/puzzles). For tasks where answers are not spatial sequences and require heavy linguistic deduction (e.g., visual math, chart understanding), whether "short is long" holds remains unverified—language CoT might still be irreplaceable there.
- Only a single base model (Qwen2.5-VL-7B) was used; the "optimal minimal grounding" threshold might vary for models of different scales or different pre-trained grounding capabilities.

## Related Work & Insights
- **vs Long CoT / Visual CoT Mainstream Narratives (o3 "think with image", etc.)**: Mainstream belief holds that more externalization and longer chains are better. This paper provides a counterexample via controlled experiments—Visual CoT only accelerates but does not improve final effectiveness, and verbosity can harm generalization, correcting the over-optimism that "Visual CoT is universally stronger."
- **vs Grounding CoT Work (Binding language to visual evidence via bbox/point/line)**: This study goes beyond using grounding to pushing it to the limit (G-CoT-least), proving explicit coordinate systems are not required as models can perform spatial reasoning in implicit latent spaces.
- **vs Vision-centric RL Work**: These studies observed that "RL-induced CoT in visual tasks is often very short." This paper further reveals the mechanism—RL is mainly strengthening existing grounding capabilities; once grounding is strong enough, extremely short CoT suffices.

## Rating
- Novelty: ⭐⭐⭐⭐ The counter-intuitive "short is long" conclusion + clean variable control design provide solid mechanism insights, though no new model is proposed.
- Experimental Thoroughness: ⭐⭐⭐⭐ Fair comparison trained to convergence + multi-task extrapolation (Mazes/Games/Real VQA), though task families are still biased toward "spatial sequence answers."
- Writing Quality: ⭐⭐⭐⭐ Clear problem-hypothesis-verification structure with well-extracted take-aways.
- Value: ⭐⭐⭐⭐ Directly provides a practical guide for generalizable visual reasoning SFT data, offering high practical significance for data construction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Improving Vision-language Models with Perception-centric Process Reward Models](improving_vision-language_models_with_perception-centric_process_reward_models.md)
- [\[CVPR 2025\] Argus: Vision-Centric Reasoning with Grounded Chain-of-Thought](../../CVPR2025/llm_reasoning/argus_vision-centric_reasoning_with_grounded_chain-of-thought.md)
- [\[CVPR 2026\] Think-as-You-See: Streaming Chain-of-Thought Reasoning for Large Vision-Language Models](think-as-you-see_streaming_chain-of-thought_reasoning_for_large_vision-language_.md)
- [\[CVPR 2026\] Think 360°: Beyond Depth — Evaluating the Width-centric Reasoning Capability of MLLMs](think_360deg_beyond_depth_evaluating_the_width-centric_reasoning_capability_of_m.md)
- [\[ICML 2026\] Diversity Matters: Revisiting Test-Time Compute in Vision-Language Models](../../ICML2026/llm_reasoning/diversity_matters_revisiting_test-time_compute_in_vision-language_models.md)

</div>

<!-- RELATED:END -->
