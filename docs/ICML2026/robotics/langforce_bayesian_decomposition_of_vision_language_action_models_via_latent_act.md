---
title: >-
  [Paper Note] LangForce: Bayesian Decomposition of Vision-Language-Action Models via Latent Action Queries
description: >-
  [ICML 2026][Robotics][VLA] LangForce reformulates the VLA policy as a Bayesian decomposition $\pi(a\mid v,\ell)=p(\ell\mid a,v)\…
tags:
  - "ICML 2026"
  - "Robotics"
  - "VLA"
  - "Visual shortcuts"
  - "Bayesian decomposition"
  - "Latent Action Queries"
  - "Pointwise Mutual Information"
date: 2026-05-08
content_hash: e47c38b3c75a5d80
---

# LangForce: Bayesian Decomposition of Vision-Language-Action Models via Latent Action Queries

**Conference**: ICML 2026  
**arXiv**: [2601.15197](https://arxiv.org/abs/2601.15197)  
**Code**: The paper mentions "Code and videos are available," but no repository URL is provided.  
**Area**: Robotics / Embodied AI / VLA  
**Keywords**: VLA, Visual shortcuts, Bayesian decomposition, Latent Action Queries, Pointwise Mutual Information

## TL;DR
LangForce reformulates the VLA policy as a Bayesian decomposition $\pi(a\mid v,\ell)=p(\ell\mid a,v)\,p(a\mid v)/p(\ell\mid v)$. It introduces learnable Latent Action Queries to execute "vision-only" and "vision+language" branches simultaneously on shared VLM weights. By maximizing the log-likelihood ratio between actions and instructions to penalize "visual shortcuts," it achieves an 11.3 absolute point improvement over the QwenGR00T baseline on SimplerEnv.

## Background & Motivation

**Background**: Current mainstream VLAs (OpenVLA, π0, GR00T, StarVLA series) attach diffusion action heads to pretrained VLMs and perform imitation learning on large-scale human demonstrations. They aim to ground natural language instructions into continuous actions using the world knowledge within the VLM.

**Limitations of Prior Work**: The authors find that these models frequently collapse in OOD scenarios and multi-task ambiguous settings. In the LIBERO Goal benchmark (multiple tasks for the same tabletop), a "vision-only" model—fed only visual input without instructions—achieves success rates close to the full model (e.g., 44.6 vs. 47.8 on RoboCasa; action loss of 0.13 vs. 0.08 on BridgeData+Fractal). This suggests that instructions play a minimal role during training.

**Key Challenge**: Goal-driven data collection makes the mapping $v\to\ell$ almost injective—seeing a cabinet implies "open the cabinet." This results in a conditional entropy $H(\ell\mid v)\approx 0$, which further drives the conditional mutual information $I(\ell;a\mid v)\le H(\ell\mid v)$ towards zero. The model learns $\pi(a\mid v,\ell)\approx p(a\mid v)$, a phenomenon the authors call "information collapse."

**Goal**: Force the policy to truly depend on language during training without re-collecting data or increasing inference computation.

**Key Insight**: Rewrite the posterior using Bayes' theorem $\pi(a\mid v,\ell)=p(\ell\mid a,v)\,p(a\mid v)/p(\ell\mid v)$ and use the log-likelihood ratio $\log p(\ell\mid a,v)-\log p(\ell\mid v)$ (i.e., conditional PMI) as a regularizer to reward policies where instructions can be inferred from actions.

**Core Idea**: Utilize a shared set of VLM weights and learnable Latent Action Query tokens. Using decoder-only causal masking, the model simultaneously simulates the "vision prior branch" and the "vision+language posterior branch." The difference between the language log-probabilities of these two branches is explicitly optimized as a PMI reward.

## Method

### Overall Architecture
LangForce adds three components to a native VLA (specifically QwenGR00T based on StarVLA, using Qwen3-VL-4B and a DiT action head): (1) $K=64$ new tokens $\mathcal{Q}=\{\langle\text{action}_1\rangle,\dots,\langle\text{action}_K\rangle\}$ in the vocabulary as Latent Action Queries; (2) A dual-branch setup within the same batch using $[v, \mathcal{Q}, \ell]$ and $[v, \ell, \mathcal{Q}]$ token sequences; (3) A total loss combining flow-matching action loss from both branches and a Language Log-Likelihood Ratio (LLR) term. Only the posterior branch is used during inference, maintaining the same overhead as a standard VLA.

### Key Designs

1.  **Latent Action Queries as Information Bottleneck**:
    - **Function**: Condenses action-related semantics from VLM outputs into $K$ fixed-length latent states $\mathbf{H}_{\mathcal{Q}}\in\mathbb{R}^{K\times D}$ before feeding them to the DiT head.
    - **Mechanism**: Adds $K=64$ learnable tokens to the VLM vocabulary and embedding table. The DiT iterates only on $\mathbf{H}_{\mathcal{Q}}$ rather than the full hidden states. By placing $\mathcal{Q}$ before $\ell$ (prior) or after $\ell$ (posterior) with causal masking, the model instantiates two conditional distributions using the same weights.
    - **Design Motivation**: Reduces DiT attention complexity from $O(N^2)$ to $O(K^2)$ and provides a structural basis for the switchable conditions in Bayesian decomposition.

2.  **Dual-branch Bayesian Training**:
    - **Function**: Estimates the visual prior $p(a\mid v)$ and the full conditional posterior $\pi(a\mid v,\ell)$ simultaneously.
    - **Mechanism**: The prior branch takes $\text{Input}_{\text{prior}}=[v,\mathcal{Q},\ell]$, where $\mathcal{Q}$ only sees $v$ due to causal masking, resulting in $\mathbf{H}_{\mathcal{Q}}^{\text{prior}}$. To prevent the VLM backbone from internalizing visual shortcuts, a stop-gradient is applied to $\mathbf{H}_{\mathcal{Q}}^{\text{prior}}$ so the prior loss only updates the DiT. The posterior branch takes $\text{Input}_{\text{post}}=[v,\ell,\mathcal{Q}]$, resulting in $\mathbf{H}_{\mathcal{Q}}^{\text{post}}$.
    - **Design Motivation**: Direct training on goal-driven data collapses $\pi$ into $p(a\mid v)$. Placing these distributions in explicit contrast allows the use of PMI to widen the gap. Shared weights and prefix prefilling minimize training overhead.

3.  **LLR / PMI Language Reward**:
    - **Function**: Penalizes actions that contain no language information, forcing $\mathbf{H}_{\mathcal{Q}}$ to carry features capable of inferring $\ell$.
    - **Mechanism**: Uses the VLM's own LM loss as an approximation for $\log p(\ell\mid\cdot)$. The objective is $\mathcal{L}_{\text{LLR}}=\log p(\ell\mid v,\mathbf{H}_{\mathcal{Q}}^{\text{prior}})-\mathrm{sg}(\log p(\ell\mid v))$. The final loss is $\mathcal{L}_{\text{total}}=(1-\lambda)\mathcal{L}_{\text{FM}}^{\text{post}}+\lambda\mathcal{L}_{\text{FM}}^{\text{prior}}-\beta\mathcal{L}_{\text{LLR}}$, with $\lambda=0.3$ and $\beta=0.1$.
    - **Design Motivation**: Maximizing PMI $\log[\pi(a\mid v,\ell)/p(a\mid v)]$ is implementationally equivalent to this log-likelihood difference. The stop-gradient prevents the model from "gaming" the reward by degrading the denominator.

### Loss & Training
Trained on 8 H100 GPUs using AdamW (lr=1e-5 + cosine), DeepSpeed ZeRO-2, and grad clip 1.0. Fine-tuned for 50k steps on BridgeData V2 + Fractal on SimplerEnv (batch 16/GPU). 

## Key Experimental Results

### Main Results

SimplerEnv (WidowX, Avg@480 Success Rate, %):

| Method | Put Spoon | Put Carrot | Stack Block | Eggplant | Average |
| :--- | :--- | :--- | :--- | :--- | :--- |
| OpenVLA-OFT | 34.2 | 30.0 | 30.0 | 72.5 | 41.8 |
| CogACT | 71.7 | 50.8 | 15.0 | 67.5 | 51.3 |
| π0 | 29.2 | 62.5 | 29.2 | 91.6 | 53.1 |
| π0.5 | 49.3 | 64.7 | 44.7 | 69.7 | 57.1 |
| Isaac-GR00T-N1.6 | 64.5 | 65.5 | 5.5 | 93.0 | 57.1 |
| QwenGR00T (baseline) | 87.5 | 50.0 | 29.2 | 54.2 | 55.2 |
| **Ours (LangForce)** | **89.6** | **63.8** | **33.3** | **79.2** | **66.5** |

RoboCasa GR1 Tabletop (24 tasks): LangForce 52.6% vs. QwenGR00T 47.8%.

Real-world Franka Pick-and-Place (OOD red cube): LangForce 9/30 vs. QwenGR00T 2/30. Vegetable sorting total: LangForce 97/120 (80.8%) vs. QwenGR00T 71/120 (59.2%).

### Ablation Study

| Configuration | Spoon | Carrot | Stack | Eggplant | Avg |
| :--- | :--- | :--- | :--- | :--- | :--- |
| QwenGR00T (Baseline) | 87.5 | 50.0 | 29.2 | 54.2 | 55.2 |
| + Latent Action Queries | 74.6 | 58.3 | 29.2 | 67.9 | 57.5 |
| Full LangForce | 89.6 | 63.8 | 33.3 | 79.2 | 66.5 |

### Key Findings
- The dual-branch + LLR setup provides the primary gain: Adding queries alone improves the average by 2.3 points, but PMI adds another 9.0 points.
- LangForce preserves the VLM's general capabilities: While QwenGR00T outputs gibberish on text math problems after fine-tuning, LangForce remains communicative, as the LLR term protects language representations.
- Improvements are concentrated in tasks requiring language-based target disambiguation (Eggplant +15, Carrot +13.6), rather than fine manipulation (Stack Block).

## Highlights & Insights
- Formulates the engineering issue of "ignored instructions" through information theory: $H(\ell\mid v)\approx 0\Rightarrow I(\ell;a\mid v)=0$, identifying PMI as a mathematically sound target.
- Uses decoder-only causal masking and token reordering to switch conditional distributions "for free" during training.
- Utilizing stop-gradients to fix the language baseline $p(\ell\mid v)$ is a critical trick to prevent the model from cheating the contrastive reward.

## Limitations & Future Work
- Dual-branch training increases memory and compute costs, though mitigated by prefix prefilling.
- Real-world experiments focus on pick-and-place; the method's advantages might be less pronounced in high-contact or dexterous manipulation tasks that rely more on low-level control than language grounding.
- The LLR loss depends on the VLM's LM head; $\beta$ might need re-tuning if the instruction style or tokenizer changes significantly.

## Related Work & Insights
- **vs BayesVLA**: Both use Bayesian decomposition, but BayesVLA requires a two-stage process (train prior, freeze, train posterior). LangForce is single-stage end-to-end.
- **vs π0 / GR00T**: Instead of feeding all hidden states ($O(N^2)$), LangForce compresses info into 64 queries and emphasizes "signal distillation" through contrastive regularization.
- **vs ChatVLA**: While ChatVLA uses task-routing for decoupling, LangForce embeds language dependency directly into the loss, providing inherent protection against forgetting.

## Rating
- Novelty: ⭐⭐⭐⭐ Applies information collapse and PMI to VLA via dual-branch training; solid logic though individual tricks (stop-gradient, shared weights) are known.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive coverage spanning SimplerEnv, RoboCasa, LIBERO, and real-world Franka tasks.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation supported by pilot experiments; methodology is logically organized.
- Value: ⭐⭐⭐⭐ Provides a practical training-side fix for the common "VLA ignoring language" problem with low implementation cost for engineering teams.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Latent Reasoning VLA: Latent Thinking and Prediction for Vision-Language-Action Models](latent_reasoning_vla_latent_thinking_and_prediction_for_vision-language-action_m.md)
- [\[ICML 2026\] Neural Implicit Action Fields: From Discrete Waypoints to Continuous Functions for Vision-Language-Action Models](neural_implicit_action_fields_from_discrete_waypoints_to_continuous_functions_fo.md)
- [\[ICML 2026\] StableVLA: Towards Robust Vision-Language-Action Models without Extra Data](stablevla_towards_robust_vision-language-action_models_without_extra_data.md)
- [\[ICML 2026\] Contrastive Representation Regularization for Vision-Language-Action Models](contrastive_representation_regularization_for_vision-language-action_models.md)
- [\[ICML 2026\] Spatial Memory for Out-of-Vision Manipulation in Vision-Language-Action](spatial_memory_for_out-of-vision_manipulation_in_vision-language-action.md)

</div>

<!-- RELATED:END -->
