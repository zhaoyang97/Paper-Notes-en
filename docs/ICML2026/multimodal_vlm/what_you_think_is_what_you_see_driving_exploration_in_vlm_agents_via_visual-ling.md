---
title: >-
  [Paper Note] What You Think is What You See: Driving Exploration in VLM Agents via Visual-Linguistic Curiosity (GLANCE)
description: >-
  [ICML 2026][Multimodal VLM][VLM agent] GLANCE introduces a self-supervised "think-see alignment" head into the reinforcement learning of VLM agents: the "next-state prediction" generated in the LLM's CoT is mapped throug…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "VLM agent"
  - "curiosity-driven exploration"
  - "cross-modal alignment"
  - "internal world model"
  - "curiosity drain"
date: 2026-05-08
content_hash: d7388b7f242c65cc
---

# What You Think is What You See: Driving Exploration in VLM Agents via Visual-Linguistic Curiosity (GLANCE)

**Conference**: ICML 2026  
**arXiv**: [2605.03782](https://arxiv.org/abs/2605.03782)  
**Code**: Not yet public (None)  
**Area**: Multimodal VLM / Reinforcement Learning / Agent Exploration  
**Keywords**: VLM agent, curiosity-driven exploration, cross-modal alignment, internal world model, curiosity drain

## TL;DR
GLANCE introduces a self-supervised "think-see alignment" head into the reinforcement learning of VLM agents: the "next-state prediction" generated in the LLM's CoT is mapped through a lightweight projector to the actual next-frame representation encoded by an EMA target visual encoder. The gap between prediction and reality simultaneously serves as an intrinsic curiosity reward, a training signal for the visual encoder, and an alignment loss to "ground" the internalized world model. Combined with a curriculum exploration mechanism that periodically resets the projector to combat curiosity drain, GLANCE consistently outperforms existing exploitation-only VLM-RL methods across 5 agentic tasks.

## Background & Motivation

**Background**: Current VLM agents (such as VAGEN, InternVL-Agent, etc.) increasingly tend to internalize "world modeling" within their policies. They use explicit CoT to write structured reasoning like `<Obs>s_t</Obs><Res>z_t</Res><Pred>s_{t+1}</Pred>` in each turn, followed by an action `<Ans>a_t</Ans>`, learning an integrated reasoning-action policy through PPO-like algorithms and sparse extrinsic rewards.

**Limitations of Prior Work**: These methods essentially perform "passive exploitation of visited states"—the agent only refines its reasoning on states it has already encountered, lacking any mechanism to actively seek out "places where its model is unclear." In sparse-reward tasks (Sokoban puzzles, 3D navigation), this often leads to pseudo-successes where an agent "perfectly describes a dead end but never tries another path." Meanwhile, traditional curiosity methods (ICM, BYOL-Explore, Latent Curiosity) only focus on vision-to-vision prediction errors, which are completely decoupled from the VLM's internalized linguistic world model—even if visual representations are learned well, linguistic reasoning may continue to hallucinate.

**Key Challenge**: The world model of VLM agents has moved from "external RNN/Transformers" into the internal CoT of the LLM, but curiosity mechanisms remain in an external vision-only form. When these two are misaligned, the agent falls into one of two failure modes: (a) linguistic-level fantasy that the visual representation cannot perceive; (b) rich visual representations while linguistic reasoning remains detached from physical reality.

**Goal**: Construct a unified objective where "what the agent thinks" must predict "what the agent sees," and transform this prediction failure into active exploration motivation; simultaneously address the "curiosity drain" problem where curiosity signals decay too quickly during long-term training.

**Key Insight**: The authors leverage the cross-modal signal of "linguistic prediction vs. visual reality"—the hidden state of the VLM at the `<Pred>` token is itself the model's semantic-level guess about the future. Projecting it into the visual space and aligning it with the real next-frame representation makes the prediction error naturally serve as (i) an alignment loss (to learn semantically actionable features for the vision encoder), (ii) supervision for grounding the world model, and (iii) an intrinsic curiosity reward—killing three birds with one stone.

**Core Idea**: Replace "visual past ↔ visual future" with the cross-modal prediction error between "linguistic hypothesis ↔ visual reality," making curiosity an "active falsification" rather than a "random search."

## Method

### Overall Architecture
GLANCE treats the VLM agent as a policy under a partially observable MDP $(\mathcal{S}, \mathcal{A}, \mathcal{O}, T, R, Z, \gamma)$, constructing an online-target dual-stream architecture: (i) **Online VLM agent** $\boldsymbol{\theta} = (\mathbf{v}, \boldsymbol{\ell})$ includes a trainable vision encoder $f_\mathbf{v}$, a frozen LLM backbone $\Lambda_\boldsymbol{\ell}$, and a lightweight projector $g_\boldsymbol{\psi}$; (ii) **Momentum target network** $\boldsymbol{\phi}$ is an EMA copy of the vision encoder. Each turn $t$: the online agent generates CoT $\Phi_t$ containing state prediction $s_{t+1}$, taking the last hidden state $h_{t+1} \in \mathbb{R}^d$ of the `<Pred>` segment as the "linguistic hypothesis"; the action $a_t$ is executed to obtain $o_{t+1}$, and the target encodes $y_{t+1} = \text{sg}(f_\phi(o_{t+1}))$; the online agent uses the projector to get $\hat{y}_{t+1} = g_\psi(h_{t+1})$, and the normalized MSE between the two yields $\mathcal{L}_\text{explore}$. This loss is backpropagated to update $g_\psi$ and $f_\mathbf{v}$ (LLM is frozen) and is added as an intrinsic reward $r_t^i = \beta \cdot \mathcal{L}_\text{explore}$ to the extrinsic rewards $r_t^e = r_t^\text{task} + r_t^\text{reason} + r_t^\text{format}$.

### Key Designs

1.  **Linguistic-to-Visual Cross-modal Alignment**:
    - **Function**: Real-time translation of the "future state guess" written in language by the agent into the visual feature space and aligning it with the actual next frame, thereby forcing "thinking" and "seeing" to point to the same physical event.
    - **Mechanism**: The final layer hidden state $h_{t+1}$ of the Transformer at the end of the `<Pred>s_{t+1}</Pred>` CoT segment in the online VLM is treated as the "linguistically implicitly encoded future state"; a lightweight projector projects it into the visual representation space: $\hat{y}_{t+1} = g_\psi(h_{t+1})$. Meanwhile, the target visual encoder $f_\phi$ is updated via EMA $\phi \leftarrow \alpha \phi + (1-\alpha) \mathbf{v}$, encoding the truly occurring next frame to obtain $y_{t+1}$. The alignment loss uses a BYOL-style normalized MSE: $\mathcal{L}_\text{explore} = \|\frac{\hat{y}_{t+1}}{\|\hat{y}_{t+1}\|_2} - \text{sg}(\frac{y_{t+1}}{\|y_{t+1}\|_2})\|_2^2$, with a stop-gradient applied to the target side to prevent representation collapse. A critical "selective gradient routing": gradients from $\mathcal{L}_\text{explore}$ pass through the frozen LLM but only update the projector $g_\psi$ and the online vision encoder $f_\mathbf{v}$, avoiding language drift while allowing the vision encoder to learn "semantically actionable" features.
    - **Design Motivation**: Traditional BYOL/SPR-style visual self-supervision can only predict "vision-to-vision," failing to ensure that linguistic reasoning and visual perception refer to the same thing. By using linguistic hidden states as queries and visual EMA as answers, GLANCE makes cross-modal alignment a single self-supervised goal, naturally "pulling" the world model from language into physical reality.

2.  **Cross-modal Curiosity as Intrinsic Reward**:
    - **Function**: "Re-purposing" the alignment loss as an intrinsic reward to drive the agent to actively visit states where its "linguistic reasoning cannot explain the visual results."
    - **Mechanism**: The $\mathcal{L}_\text{explore}$ of the current turn directly becomes the intrinsic reward $r_t^i = \beta \cdot \mathcal{L}_\text{explore}(\mathbf{v}, \boldsymbol{\psi}, t)$, which is combined with extrinsic rewards into $r_t = r_t^e + r_t^i$ and fed into a PPO-style Bi-Level GAE for token-to-turn hierarchical credit assignment. Intuitively, a large $\mathcal{L}_\text{explore}$ means "my linguistic next-state prediction differs significantly from what I actually saw"—i.e., a known unknown, which is exactly where exploration is valuable; conversely, low-loss regions are states the model is already familiar with.
    - **Design Motivation**: The blind spot of standard ICM-style curiosity in VLMs is its decoupling from LLM reasoning, which can lead to a pseudo-equilibrium where "visual representations are learned but linguistic reasoning still hallucinates." By using the "language-visual alignment error" as a reward, curiosity and world model grounding share the same objective; the agent must improve both reasoning and perception to reduce loss.

3.  **Curriculum Exploration: Periodic projector resets to combat curiosity drain**:
    - **Function**: Solves the problem where a "lightweight projector converges too quickly in the presence of a pre-trained semantically rich LLM, leading to premature decay of intrinsic rewards."
    - **Mechanism**: The authors found that because the LLM backbone is already semantically strong, the projector easily fits linguistic hidden states to "surface visual features" early in training. This causes $\mathcal{L}_\text{explore}$ to rapidly approach zero, causing intrinsic rewards to disappear and making the agent falsely believe it has "mastered the environment." GLANCE periodically re-initializes the weights of the projector $g_\psi$ while keeping the evolving vision encoder $f_\mathbf{v}$. The new projector, stripped of "old tricks," is forced to recalibrate using the richer features already learned by the vision encoder, re-exposing fine-grained differences previously smoothed out by the old projector, forming a "self-paced curriculum."
    - **Design Motivation**: BYOL/SimSiam-style self-supervision faces similar collapse issues, but stop-gradients + EMA are usually sufficient. In GLANCE, the collapse is not a breakdown of representations but a "premature learning" illusion; thus, stop-gradients cannot save it, requiring an active "reset of one negotiator" to force the model to compare again—a clever adaptation of curriculum learning for self-supervised curiosity.

### Loss & Training
The total optimization objective is split into two paths:
- **Self-supervised Path**: $\min_{\mathbf{v}, \boldsymbol{\psi}} \mathcal{L}_\text{explore}$, with LLM frozen.
- **RL Path**: PPO maximizes $\mathcal{J}(\boldsymbol{\theta}) = \mathbb{E}[\sum_t \gamma^t r_t]$ where $r_t = r_t^e + r_t^i$, using Bi-Level GAE to backpropagate turn-level rewards to the token-level.
The LLM remains frozen throughout, updating only the projector + vision encoder + trainable adaptation layers in RL; the target network follows EMA $\phi \leftarrow \alpha \phi + (1-\alpha) \mathbf{v}$; the Curriculum step resets the projector every $K$ iterations.

## Key Experimental Results

### Main Results
Five agentic tasks (Grid Puzzles, Sokoban, 3D Navigation, Object Manipulation/PrimitiveSkill, Geometric/SVG Reconstruction), all using Qwen2.5-VL-3B as the backbone. Primary metrics: average success rate for puzzle/embodied, and average perceptual similarity using DINO+DreamSim for SVG.

| Benchmark | VAGEN (exploitation-only) | GLANCE (Ours) | Description |
|---|---|---|---|
| Sokoban | baseline | Significant Gain | Sparse reward + long horizon, maximum curiosity benefit |
| 3D Navigation | baseline | Significant Gain | Strong visual partial observability |
| PrimitiveSkill | baseline | Significant Gain | Needs to predict physical consequences like "stacking/moving" |
| Grid Puzzles | baseline | Significant Gain | Tight reasoning-perception coupling |
| SVG Reconstruction | baseline | Significant Gain | Averaged via DINO+DreamSim |

Authors also reported "zero extrinsic reward" experiments: setting $r_t^e$ to 0 while keeping $r_t^i$, GLANCE still learns meaningful exploration strategies, whereas the exploitation-only baseline fails to start learning entirely—directly proving that cross-modal curiosity can independently drive exploration.

### Ablation Study

| Configuration | Phenomena | Description |
|---|---|---|
| Full GLANCE | Stable training, $\mathcal{L}_\text{explore}$ re-excited by curriculum after decay | Full model |
| w/o Curriculum Exploration | $\mathcal{L}_\text{explore}$ rapidly collapses to ≈0 early, agent exploration stops | Verifies curiosity drain exists and curriculum is necessary |
| w/o cross-modal alignment (Pure BYOL) | Visual features learned but reasoning still hallucinates, performance drops in long-horizon tasks | Verifies hidden state as query is necessary to ground world model |
| w/o EMA target (Online as target) | Representation collapse | Verifies stop-gradient + EMA is required |

### Key Findings
- Cross-modal curiosity can function independently of extrinsic rewards: With zero $r_t^e$, GLANCE consistently learns exploration strategies, which vision-only ICM cannot do, proving "linguistic-visual" signals contain sufficient semantics regarding "where the problems are."
- Curriculum resetting of the projector is indispensable: Without it, $\mathcal{L}_\text{explore}$ tends toward 0 in the first 10%–20% of training, and the agent enters a pseudo-equilibrium of "thinking it knows everything"—a new collapse mode for BYOL-style self-supervision in VLM-RL that hasn't been discussed before.
- Selective gradient routing (freezing LLM, releasing vision encoder and projector) balances "no language drift" with "learning visual semantics," serving as a key engineering trick for putting self-supervised loss into VLMs.

## Highlights & Insights
- **"Think-See Alignment" as a Unified Objective**: The authors use a single $\mathcal{L}_\text{explore}$ to solve three problems simultaneously—intrinsic reward, visual representation learning, and world model grounding. This "one loss for multiple tasks" design is minimalist and elegant, a core "concentrated idea" worth learning in paper writing.
- **Using `<Pred>` Token Hidden State as Semantic Query**: This step naturally connects the "VLM's internal world model" with "self-supervised alignment," capturing a key structural difference between VLM agents and traditional RL agents.
- **Concept of Curiosity Drain**: The authors systematically discuss how "lightweight projectors quickly overfit in front of rich semantic backbones, leading to curiosity decay." This is a real but rarely discussed failure mode, and the proposed curriculum solution is very cheap (simply resetting weights) but effective, transferable to any "small adapter + large frozen model" self-supervised architecture.
- **Learnable with Zero Extrinsic Rewards**: This "pure curiosity-driven" result is significant—it means a VLM agent can self-warm-start in a new environment using cross-modal alignment before superiterating with extrinsic rewards once the task structure is found.

## Limitations & Future Work
- Code is not yet public (as of arXiv v1); specific hyperparameters ($\beta$, curriculum period $K$, EMA $\alpha$) can only be guessed from descriptions, making reproduction difficult.
- Main experiments only use Qwen2.5-VL-3B; it is not verified if larger models (7B/72B) or stronger vision encoders (SigLIP-2, InternViT) require retuning the curriculum period.
- The definition of "linguistic hypothesis = last hidden state of `<Pred>` token" is quite rigid; the model might hide critical info in middle tokens; attention pooling or averaging the whole `<Pred>` segment could be considered.
- The online update of the vision encoder in the RL loop may damage the original vision-language alignment pre-training, which is not quantified; some downstream zero-shot visual capabilities might degrade.
- All five tasks are in controlled simulator environments; generalization to long-horizon, open-domain scenarios like real web/robot agents is not yet verified.

## Related Work & Insights
- **vs ICM / RND / Latent Curiosity (Pathak et al., 2017; Burda et al., 2018; Ermolov & Sebe, 2020)**: Classic curiosity focuses on "vision→vision" or "action→vision" prediction errors; GLANCE changes this to "language→vision" alignment, directly taking over the VLM's internalized world model.
- **vs BYOL-Explore (Guo et al., 2022)**: BYOL-Explore uses BYOL-style visual bootstrapping for intrinsic rewards but is decoupled from LLM reasoning; GLANCE adopts the BYOL framework (stop-gradient + EMA + normalized MSE) but uses the LLM hidden state as the query side.
- **vs VAGEN / InternVL-Agent (Wang et al., 2025; Chen et al., 2025)**: These VLM-RL frameworks internalize world models but focus only on exploitation; GLANCE adds a plug-and-play exploration enhancement via the cross-modal curiosity loss.
- **Insight**: The paradigm of "Intermediate representation of modality A → Actual representation of modality B" as an alignment loss + intrinsic reward can be extended to other multimodal agents—such as audio-visual or text-mechanical control; as long as an internalized world model's linguistic/intermediate representation exists, this can be implemented.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The trio of "linguistic prediction vs. visual reality" as a unified objective + curiosity drain + curriculum projector are all fresh and combined concisely.
- Experimental Thoroughness: ⭐⭐⭐⭐ Coverage of five tasks (puzzle/nav/manipulation/recon) is wide, and zero-extrinsic-reward ablation is powerful; however, using only one backbone is a slight drawback.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clearly derived, and the "thinking ↔ seeing" narrative is consistent; formula layout was somewhat affected by arXiv HTML noise.
- Value: ⭐⭐⭐⭐ Provides a plug-and-play exploration module for VLM-RL training with clear gains in long-horizon/sparse-reward tasks, serving as a direct reference for embodied/web agent training paradigms.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Revisit What You See: Revealing Visual Semantics in Vision Tokens to Guide LVLM Decoding](../../ACL2026/multimodal_vlm/revisit_what_you_see_revealing_visual_semantics_in_vision_tokens_to_guide_lvlm_d.md)
- [\[ACL 2026\] "I See What You Did There": Can Large Vision-Language Models Understand Multimodal Puns?](../../ACL2026/multimodal_vlm/i_see_what_you_did_there_can_large_vision-language_models_understand_multimodal_.md)
- [\[ICML 2026\] Visual Persuasion: What Influences the Decision-Making of Vision-Language Models?](visual_persuasion_what_influences_decisions_of_vision-language_models.md)
- [\[CVPR 2026\] See, Think, Act: Teaching Multimodal Agents to Effectively Interact with GUI by Identifying Toggles](../../CVPR2026/multimodal_vlm/see_think_act_teaching_multimodal_agents_to_effectively_interact_with_gui_by_ide.md)
- [\[CVPR 2026\] Aligning What Vision-Language Models See and Perceive with Adaptive Information Flow](../../CVPR2026/multimodal_vlm/aif_adaptive_information_flow_vlm.md)

</div>

<!-- RELATED:END -->
