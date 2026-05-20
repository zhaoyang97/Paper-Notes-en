---
title: >-
  [Paper Note] What You Think is What You See: Driving Exploration in VLM Agents via Visual-Linguistic Curiosity (GLANCE)
description: >-
  [ICML 2026][Multimodal VLM][VLM agent] GLANCE introduces a "think-see alignment" self-supervised head into VLM agent reinforcement learning: the LLM’s "next state prediction" in CoT is projected via a lightweight project…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "VLM agent"
  - "curiosity-driven exploration"
  - "cross-modal alignment"
  - "internal world model"
  - "curiosity drain"
date: 2026-05-08
content_hash: 501fef441ba459c9
---

# What You Think is What You See: Driving Exploration in VLM Agents via Visual-Linguistic Curiosity (GLANCE)

**Conference**: ICML 2026  
**arXiv**: [2605.03782](https://arxiv.org/abs/2605.03782)  
**Code**: Not yet released (none)  
**Area**: Multimodal VLM / Reinforcement Learning / Agent Exploration  
**Keywords**: VLM agent, curiosity-driven exploration, cross-modal alignment, internal world model, curiosity drain

## TL;DR
GLANCE introduces a "think-see alignment" self-supervised head into VLM agent reinforcement learning: the LLM’s "next state prediction" in CoT is projected via a lightweight projector to the real next-frame representation encoded by an EMA target vision encoder. The prediction gap serves simultaneously as intrinsic curiosity reward, training signal for the vision encoder, and an alignment loss to ground the internalized world model. Combined with a curriculum exploration mechanism that periodically resets the projector to counteract curiosity drain, GLANCE consistently outperforms existing exploitation-only VLM-RL methods across five agentic tasks.

## Background & Motivation

**Background**: Current VLM agents (e.g., VAGEN, InternVL-Agent) increasingly internalize "world modeling" into the policy: using explicit CoT to structure each round as `<Obs>s_t</Obs><Res>z_t</Res><Pred>s_{t+1}</Pred>`, then outputting an action `<Ans>a_t</Ans>`, and learning integrated reasoning-action policies via PPO-like algorithms with sparse external rewards.

**Limitations of Prior Work**: These methods essentially "passively refine visited states"—the agent only polishes reasoning on states it has already encountered, lacking any mechanism to actively seek out "areas its model cannot explain." In sparse reward tasks (e.g., Sokoban, 3D navigation), this leads to "perfectly describing dead ends without ever trying other paths." Traditional curiosity methods (ICM, BYOL-Explore, Latent Curiosity) only consider vision-to-vision prediction errors, remaining disconnected from the language-based world model internalized in VLMs—so even with well-learned visual representations, language reasoning may still hallucinate.

**Key Challenge**: The world model in VLM agents has moved from "external RNN/Transformer" into the LLM’s CoT, but curiosity mechanisms remain external and vision-only; this misalignment leads to two failure modes: (a) language-level speculation ungrounded in perception; (b) rich visual representations but language reasoning detached from physical reality.

**Goal**: Construct a unified objective so that "what the agent thinks" must predict "what the agent sees," and convert prediction failures into active exploration incentives; also address the "curiosity drain" problem where intrinsic signals decay too quickly during long training.

**Key Insight**: The authors leverage the cross-modal signal of "linguistic prediction vs. visual reality"—the hidden state at the `<Pred>` token in VLM is a semantic guess of the future, which is projected into visual space and aligned with the actual next-frame representation. The prediction error naturally serves as (i) alignment loss (training the vision encoder for semantically actionable features), (ii) supervision to ground the world model, and (iii) intrinsic curiosity reward—three objectives in one.

**Core Idea**: Replace "visual past ↔ visual future" with cross-modal prediction error between "linguistic hypothesis ↔ visual reality," making curiosity an "active falsification" rather than random search.

## Method

### Overall Architecture
GLANCE treats the VLM agent as a policy under a partially observable MDP $(\mathcal{S}, \mathcal{A}, \mathcal{O}, T, R, Z, \gamma)$, constructing an online-target dual-stream architecture: (i) **Online VLM agent** $\boldsymbol{\theta} = (\mathbf{v}, \boldsymbol{\ell})$ includes a trainable vision encoder $f_\mathbf{v}$, frozen LLM backbone $\Lambda_\boldsymbol{\ell}$, and lightweight projector $g_\boldsymbol{\psi}$; (ii) **Momentum target network** $\boldsymbol{\phi}$ is an EMA copy of the vision encoder. At each turn $t$: the online agent generates CoT $\Phi_t$ including state prediction $s_{t+1}$, takes the hidden state $h_{t+1} \in \mathbb{R}^d$ at the end of the `<Pred>` segment as the "linguistic hypothesis"; after executing action $a_t$ and obtaining $o_{t+1}$, the target encodes $y_{t+1} = \text{sg}(f_\phi(o_{t+1}))$; the online agent projects to $\hat{y}_{t+1} = g_\psi(h_{t+1})$, and the normalized MSE between the two yields $\mathcal{L}_\text{explore}$. This loss is backpropagated to update $g_\psi$ and $f_\mathbf{v}$ (LLM frozen), and also serves as intrinsic reward $r_t^i = \beta \cdot \mathcal{L}_\text{explore}$, added to external reward $r_t^e = r_t^\text{task} + r_t^\text{reason} + r_t^\text{format}$.

### Key Designs

1. **Linguistic-to-Visual Cross-modal Alignment**:

    - **Function**: Translates the agent’s linguistic "future state guess" into the visual feature space in real time and aligns it with the actual next frame, forcing "thinking" and "seeing" to refer to the same physical event.
    - **Mechanism**: The online VLM takes the last-layer hidden state $h_{t+1}$ at the end of `<Pred>s_{t+1}</Pred>` in CoT as the "language-encoded future state"; a lightweight projector maps it to the visual representation space: $\hat{y}_{t+1} = g_\psi(h_{t+1})$. The target vision encoder $f_\phi$ is updated via EMA $\phi \leftarrow \alpha \phi + (1-\alpha) \mathbf{v}$, encoding the actual next frame as $y_{t+1}$. The alignment loss adopts BYOL-style normalized MSE: $\mathcal{L}_\text{explore} = \|\frac{\hat{y}_{t+1}}{\|\hat{y}_{t+1}\|_2} - \text{sg}(\frac{y_{t+1}}{\|y_{t+1}\|_2})\|_2^2$, with stop-gradient on the target side to prevent collapse. Crucially, "selective gradient routing" allows gradients from $\mathcal{L}_\text{explore}$ to pass through the frozen LLM but only update the projector $g_\psi$ and online vision encoder $f_\mathbf{v}$, avoiding language drift while enabling the vision encoder to learn semantically actionable features.
    - **Design Motivation**: Traditional BYOL/SPR visual self-supervision only predicts "vision to vision," failing to ensure language reasoning and visual perception refer to the same entity. GLANCE treats the language hidden state as the query and the visual EMA as the answer, making cross-modal alignment a single self-supervised objective, naturally grounding the world model in physical reality.

2. **Cross-modal Curiosity as Intrinsic Reward**:

    - **Function**: Reuses the alignment loss as intrinsic reward, driving the agent to actively visit states where "its language reasoning cannot explain the visual outcome."
    - **Mechanism**: The current turn’s $\mathcal{L}_\text{explore}$ directly becomes the intrinsic reward $r_t^i = \beta \cdot \mathcal{L}_\text{explore}(\mathbf{v}, \boldsymbol{\psi}, t)$, which is combined with external reward $r_t = r_t^e + r_t^i$ and fed into PPO-style Bi-Level GAE for hierarchical credit assignment from token to turn. Intuitively, a large $\mathcal{L}_\text{explore}$ means "my language-based next-state prediction differs greatly from what I actually see"—a known unknown, and thus worth exploring; conversely, low-loss regions are familiar states.
    - **Design Motivation**: Standard ICM-style curiosity in VLMs is blind to LLM reasoning, possibly resulting in "well-learned visual representations but persistent language hallucinations." Using "language-visual alignment error" as reward unifies curiosity and world model grounding, forcing the agent to improve both reasoning and perception to reduce loss.

3. **Curriculum Exploration: Periodic Projector Reset to Counter Curiosity Drain**:

    - **Function**: Addresses the issue where the lightweight projector quickly overfits in front of a semantically rich LLM backbone, causing intrinsic rewards to decay prematurely.
    - **Mechanism**: The authors observe that the LLM backbone is already highly semantic, so the projector easily fits the language hidden state to "surface-level visual features" early in training, causing $\mathcal{L}_\text{explore}$ to approach zero and intrinsic rewards to vanish, misleading the agent into thinking it has "mastered the environment." GLANCE periodically reinitializes the projector $g_\psi$ weights while retaining the evolving vision encoder $f_\mathbf{v}$; the new projector, stripped of "old tricks," is forced to recalibrate using the richer features learned by the vision encoder, re-exposing fine-grained differences previously smoothed out by the old projector, forming a "self-paced curriculum."
    - **Design Motivation**: BYOL/SimSiam-style self-supervision also faces collapse, but stop-gradient + EMA suffices there; in GLANCE, the collapse is a false sense of "fully learned," so stop-gradient is insufficient—actively "resetting one side’s fitter" forces the model to realign, a clever adaptation of curriculum learning to self-supervised curiosity.

### Loss & Training

The overall optimization objective has two branches:
- **Self-supervised branch**: $\min_{\mathbf{v}, \boldsymbol{\psi}} \mathcal{L}_\text{explore}$, with LLM frozen.
- **RL branch**: PPO maximizes $\mathcal{J}(\boldsymbol{\theta}) = \mathbb{E}[\sum_t \gamma^t r_t]$ where $r_t = r_t^e + r_t^i$, using Bi-Level GAE to backpropagate turn-level rewards to token-level.
The LLM remains frozen throughout; only the projector, vision encoder, and trainable adapters in RL are updated. The target network follows EMA $\phi \leftarrow \alpha \phi + (1-\alpha) \mathbf{v}$. The curriculum step resets the projector every $K$ iterations.

## Key Experimental Results

### Main Results
Five agentic tasks (Grid Puzzles, Sokoban, 3D Navigation, Object Manipulation/PrimitiveSkill, Geometric/SVG Reconstruction) are evaluated, all using Qwen2.5-VL-3B as backbone. Main metrics: average success rate for puzzle/embodied tasks, DINO+DreamSim mean perceptual similarity for SVG.

| Benchmark | VAGEN (exploitation-only) | GLANCE (Ours) | Notes |
|---|---|---|---|
| Sokoban | baseline | significant improvement | Sparse rewards + long horizon, curiosity yields largest gains |
| 3D Navigation | baseline | significant improvement | Strong visual partial observability |
| PrimitiveSkill | baseline | significant improvement | Requires predicting physical outcomes like stacking/moving |
| Grid Puzzles | baseline | significant improvement | Tight reasoning-perception coupling |
| SVG Reconstruction | baseline | significant improvement | DINO+DreamSim mean |

The authors also report a "zero external reward" experiment: setting $r_t^e$ to 0, leaving only $r_t^i$. GLANCE still learns meaningful exploration strategies, while the exploitation-only baseline fails to learn—directly demonstrating that cross-modal curiosity alone can drive exploration.

### Ablation Study

| Configuration | Phenomenon | Notes |
|---|---|---|
| Full GLANCE | Training is stable, $\mathcal{L}_\text{explore}$ decays then is reignited by curriculum | Full model |
| w/o Curriculum Exploration | $\mathcal{L}_\text{explore}$ collapses to ≈0 early, agent stops exploring | Validates reality of curiosity drain and necessity of curriculum |
| w/o cross-modal alignment (vision-only BYOL) | Visual representations learned but language reasoning still hallucinates, performance drops on long-horizon tasks | Validates that querying with language hidden state is necessary to ground the world model |
| w/o EMA target (online network as target) | Representation collapse | Validates that stop-gradient + EMA is necessary to prevent collapse |

### Key Findings
- Cross-modal curiosity can operate independently of external rewards: with zero $r_t^e$, GLANCE still steadily learns exploration strategies, which vision-only ICM cannot achieve, proving that the "linguistic-visual" signal alone contains sufficient semantic information about "where the problem is."
- Curriculum projector reset is indispensable: without it, $\mathcal{L}_\text{explore}$ approaches zero within the first 10–20% of training, and the agent enters a false equilibrium of "thinking it knows everything"—a new collapse mode not previously discussed in BYOL-style self-supervision for VLM-RL.
- Selective gradient routing (freezing LLM, updating vision encoder and projector) balances "no language drift" and "vision learns semantics," and is a key engineering trick for integrating self-supervised loss into VLMs.

## Highlights & Insights
- **"Think-see alignment" as a unified objective**: The authors use a single $\mathcal{L}_\text{explore}$ to simultaneously address intrinsic reward, visual representation learning, and world model grounding—this "one loss solves multiple problems" design is minimalistic and elegant, a model of "main idea condensation" for academic writing.
- **Using `<Pred>` token hidden state as semantic query**: This step naturally connects the "VLM internal world model" with "self-supervised alignment," a key observation that leverages the architectural difference between VLM agents and traditional RL agents.
- **Curiosity Drain concept**: The authors name and systematically discuss the real but underexplored failure mode where a lightweight projector quickly overfits in front of a semantically rich backbone, causing premature curiosity decay. The proposed curriculum solution (simply resetting projector weights) is cheap yet empirically effective, and the idea is transferable to any "small adapter + large frozen model" self-supervised alignment architecture.
- **Learning with zero external reward**: This "pure curiosity-driven" result is striking—it means VLM agents can bootstrap themselves in new environments via cross-modal alignment, then layer on external rewards once the task structure is discovered.

## Limitations & Future Work
- The code is not yet public (as of arXiv v1), so specific hyperparameters ($\beta$, curriculum period $K$, EMA $\alpha$) can only be inferred from descriptions, raising the bar for reproduction.
- Main experiments use only Qwen2.5-VL-3B as backbone; it is unverified whether larger models (7B/72B) or stronger vision encoders (SigLIP-2, InternViT) require retuning the curriculum period.
- The definition "linguistic hypothesis = `<Pred>` token final hidden state" is somewhat rigid; the model could hide key information in intermediate tokens. Attention pooling or averaging over the entire `<Pred>` segment could be considered.
- The current architecture updates the vision encoder online during RL training, with no quantification of the impact on vision-language alignment pretraining; some downstream zero-shot visual capabilities may degrade.
- All five tasks are still relatively controlled simulator environments; generalization to real web/robot agents in long-horizon, open-domain scenarios remains untested.

## Related Work & Insights
- **vs ICM / RND / Latent Curiosity (Pathak et al., 2017; Burda et al., 2018; Ermolov & Sebe, 2020)**: Classic curiosity only considers "vision→vision" or "action→vision" prediction errors; GLANCE switches to "language→vision" alignment, directly taking over the VLM-internalized world model.
- **vs BYOL-Explore (Guo et al., 2022)**: BYOL-Explore uses BYOL-style visual bootstrapping for intrinsic reward but is disconnected from LLM reasoning; GLANCE retains BYOL’s stop-gradient + EMA + normalized MSE framework but replaces the query side with LLM hidden state.
- **vs VAGEN / InternVL-Agent (Wang et al., 2025; Chen et al., 2025)**: These VLM-RL frameworks internalize the world model but only perform exploitation; GLANCE adds a cross-modal curiosity loss on top, serving as a plug-and-play exploration enhancement.
- **Insights**: The paradigm of using "modality A’s intermediate representation → modality B’s real representation" as both alignment loss and intrinsic reward can be generalized to other multimodal agents—e.g., audio-visual, text-mechanical control; as long as there is an internalized world model in language/intermediate representations, this approach applies.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The unified objective of "linguistic prediction vs. visual reality," the curiosity drain concept, and curriculum projector are all new and elegantly combined.
- Experimental Thoroughness: ⭐⭐⭐⭐ Five tasks cover puzzle/navigation/manipulation/reconstruction; the zero external reward ablation is compelling, though only one backbone is used.
- Writing Quality: ⭐⭐⭐⭐ Motivation and derivation are clear, with a consistent "thinking ↔ seeing" narrative; formula formatting is affected by arXiv HTML noise.
- Value: ⭐⭐⭐⭐ Provides a plug-and-play exploration module for VLM agent RL training, with clear benefits for long-horizon/sparse-reward tasks, and direct reference value for embodied/web agent training paradigms.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Ground What You See: Hallucination-Resistant MLLMs via Caption Feedback, Diversity-Aware Sampling, and Conflict Regularization](../../AAAI2026/multimodal_vlm/ground_what_you_see_hallucination-resistant_mllms_via_caption_feedback_diversity.md)
- [\[ACL 2025\] I See What You Mean: Co-Speech Gestures for Reference Resolution in Multimodal Dialogue](../../ACL2025/multimodal_vlm/i_see_what_you_mean_co-speech_gestures_for_reference_resolution_in_multimodal_di.md)
- [\[CVPR 2026\] Aligning What Vision-Language Models See and Perceive with Adaptive Information Flow](../../CVPR2026/multimodal_vlm/aif_adaptive_information_flow_vlm.md)
- [\[CVPR 2026\] See, Think, Act: Teaching Multimodal Agents to Effectively Interact with GUI by Identifying Toggles](../../CVPR2026/multimodal_vlm/see_think_act_teaching_multimodal_agents_to_effectively_interact_with_gui_by_ide.md)
- [\[ICLR 2026\] Empowering Small VLMs to Think with Dynamic Memorization and Exploration](../../ICLR2026/multimodal_vlm/empowering_small_vlms_to_think_with_dynamic_memorization_and_exploration.md)

</div>

<!-- RELATED:END -->
