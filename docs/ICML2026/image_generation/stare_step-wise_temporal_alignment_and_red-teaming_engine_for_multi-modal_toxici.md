---
title: >-
  [Paper Note] STARE: Step-wise Temporal Alignment and Red-teaming Engine for Multi-modal Toxicity Attack
description: >-
  [ICML 2026][Image Generation][Multi-modal Red-teaming] This work treats the entire denoising trajectory of T2I models as the "attack surface" for VLM red-teaming. By utilizing a hierarchical RL framework (STARE) consisti…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "Multi-modal Red-teaming"
  - "Diffusion Trajectory Attack"
  - "Hierarchical RL"
  - "GRPO"
  - "Temporal Alignment Analysis"
date: 2026-05-08
content_hash: c693fa0d1953a27e
---

# STARE: Step-wise Temporal Alignment and Red-teaming Engine for Multi-modal Toxicity Attack

**Conference**: ICML 2026  
**arXiv**: [2605.00699](https://arxiv.org/abs/2605.00699)  
**Code**: https://github.com/henrymao2004/STARE.git (Included)  
**Area**: Image Generation / Multi-modal VLM Safety / Red-teaming  
**Keywords**: Multi-modal Red-teaming, Diffusion Trajectory Attack, Hierarchical RL, GRPO, Temporal Alignment Analysis

## TL;DR
This work treats the entire denoising trajectory of T2I models as the "attack surface" for VLM red-teaming. By utilizing a hierarchical RL framework (STARE) consisting of a high-level prompt editor and low-level GRPO fine-tuning for a rectified-flow model, the authors not only improve the attack success rate by 68% over SOTA but also reveal a novel phenomenon: **Optimization-Induced Phase Alignment**. Adversarial optimization automatically binds "conceptual toxicity" to early denoising stages and "detailed toxicity" to later stages, transforming the chaotic toxicity formation into predictable "vulnerability time windows."

## Background & Motivation

**Background**: Toxic continuation attacks on VLMs represent a significant multi-modal safety threat, where attackers use T2I models to generate adversarial images designed to trigger highly toxic outputs from a VLM given a text prefix. Existing red-teaming methods (PGJ, DiffZOO, ART, RedDiffuser, etc.) typically treat T2I as a black box, focusing only on terminal toxicity scores while ignoring at which step the toxic semantics emerge.

**Limitations of Prior Work**: A terminal-only perspective leads to "temporal opacity." Diffusion models possess an inherent mechanism of semantic emergence from coarse to fine (early stages determine layout/concept, later stages determine details). Existing red-teaming ignores this temporal structure, causing sparse global rewards to fail in providing attribution—one neither knows "why an adversarial image jailbreaks" nor how to perform precise intervention for defense.

**Key Challenge**: (1) Black-box optimization vs. white-box attack surface: Treating T2I as a black box only yields final toxicity, yet intermediate steps in diffusion models contain exploitable semantic emergence patterns; (2) Flattened RL vs. hierarchical semantic structure: Standard RL (e.g., DDPO) treats the entire generation as a single policy, failing to correspond to the natural division of "early layout / late details"; (3) Conceptual vs. detailed toxicity: Real-world toxicity includes "concept-level" types (identity/threat) requiring early seeds and "detail-level" types (obscene/insult) requiring late-stage amplification, but baselines apply uniform pressure.

**Goal**: (1) Design a hierarchical RL framework capable of explicitly manipulating both early and late stages of the denoising trajectory for end-to-end VLM toxicity attacks; (2) Reveal the impact of adversarial optimization on diffusion temporal structure through temporal alignment analysis; (3) Push Attack Success Rate (ASR) to SOTA.

**Key Insight**: The authors utilize rectified flow as the backbone (as its velocity field is explicit and trajectories are near-linear, facilitating temporal attribution). They link "prompt editing for semantic subgoals" and "velocity field fine-tuning for detail amplification" to high-level and low-level MDPs, respectively. This hierarchical structure naturally corresponds to the early and late stages of semantic emergence in diffusion.

**Core Idea**: A high-level prompt editor is used to plant "conceptual toxicity subgoals" in the embedding space, while low-level GRPO fine-tunes the rectified-flow velocity field to amplify "detailed toxicity." Both policies share the same toxicity reward. Temporal attribution analysis (MLMC + block perturbation) demonstrates that this hierarchical structure corresponds to real early and late vulnerability windows.

## Method

### Overall Architecture

Input: Root prompt $p$, white-box T2I model (SD 3.5-Medium + LoRA $r=16$), and query-level black-box VLM (LLaVA-v1.6-mistral-7b). Workflow: (1) High-level: Add noise/perturbation to the embedding of $p$ to obtain $K$ candidate edits $e_p + \delta_j$, decoded via vec2text into $K$ subgoal prompts $p'^{(j)}$; (2) Low-level: For each $p'^{(j)}$, perform $M$ image rollouts using the current LoRA-augmented velocity field $v_\theta$ (using Marginal-Preserving Stochastic SDE for discretization to ensure exploration); (3) VLM provides toxicity scores for each image + continuation prompt, combined with CLIPScore alignment rewards to form a terminal reward; (4) Update both policies using GRPO objectives (group-normalized advantage). The high-level group uses the average reward of $K$ candidates + edit reward; the low-level group uses individual rewards from all $K \times M$ rollouts. This pipeline forms a dual-layer closed loop: "semantic subgoals $\rightarrow$ image generation $\rightarrow$ VLM continuation $\rightarrow$ toxicity $\rightarrow$ backpropagation to both policies."

### Key Designs

1. **Hierarchical MDP (High-Level Prompt Editor + Low-Level Velocity Fine-tuning)**:
    - **Function**: Distributes semantic injection and detail amplification into two policies at different time scales, corresponding to early conceptual seeds and late-stage refinement in diffusion.
    - **Mechanism**: The high-level MDP is a single-step decision where the state is prompt embedding $e_p$ and the action is an edit vector $\delta$. The policy $\pi_{edit}(\delta|e_p)$ is an encoder-decoder Transformer outputting $\mu_j$, projected onto an $\ell_2$ ball $\delta_j = \epsilon_p \cdot \mu_j / \max(\|\mu_j\|_2, \epsilon_p)$ ($\epsilon_p = 0.8$). The low-level MDP is iterative denoising where state $s_t = (x_t, t, c)$, action $a_t = x_{t - \Delta t}$, and policy $\pi_\theta(a_t|s_t) = \mathcal{N}(\mu_\theta, \sigma_t^2 I)$ with $\mu_\theta = x_t - v_\theta(x_t, t, c) \Delta t$.
    - **Design Motivation**: T2I early stages mainly determine semantics/layout, while later stages determine details. This naturally maps to "prompt for semantics" and "velocity for image statistics." By separating these, each policy applies pressure to its respective era of expertise, proving more precise than flattened RL like DDPO (+21% ASR over DDPO).

2. **GRPO Dual-layer Optimization + Marginal Reward Composition**:
    - **Function**: Uses group-normalized advantage instead of absolute rewards to reduce variance under sparse rewards; includes an auxiliary high-level reward for semantic preservation.
    - **Mechanism**: GRPO loss $\mathcal{L}_{grp}(r_t, \hat A, \varepsilon) = \min(r_t \hat A, \mathrm{clip}(r_t, 1-\varepsilon, 1+\varepsilon) \hat A)$ where $r_t = \pi_\theta(a_t|s_t)/\pi_{old}(a_t|s_t)$ and $\hat A_i = (X_i - \mu_{grp})/(\sigma_{grp} + \epsilon)$. The high-level group reward is $\mathcal{R}_{high}^{(j)} = \bar R_j + \mathcal{R}_{edit}^{(j)}$, where $\mathcal{R}_{edit}^{(j)} = \lambda_{sem}[s_{SBERT}(e_p, e_p + \delta_j) - \tau_{sem}]_+ + \lambda_{recon}/(1 + \|e_p + \delta_j - \mathrm{emb}(p'^{(j)})\|^2)$. The low-level group reward is $R^{(j,m)} = R_{tox}^{(j,m)} + w_{align} R_{align}^{(j,m)}$.
    - **Design Motivation**: Toxicity reward is sparse and noisy. Group normalization provides much lower variance than absolute rewards. Flow-DPO requires preference datasets which is costly for a dual-layer structure; GRPO is the more lightweight choice.

3. **Temporal Alignment Analysis (MLMC Attribution)**:
    - **Function**: Quantifies "which denoising step contributes most to which category of toxicity" as a $T \times D$ heatmap to verify that the hierarchical structure maps to different time windows.
    - **Mechanism**: Defines net toxicity score $\mathcal{R}_d(I, p) = R_d(\mathrm{VLM}(I, p)) - R_d(\mathrm{VLM}(\mathrm{null}, p))$. Sensitivity to a time block $B$ is defined as $\Delta_B^{(d)} = \mathbb{E}_{\mathbf{z}}[(\mathcal{R}_d(G^{(B, +\eta\mathbf{z})}) - \mathcal{R}_d(G^{(B, -\eta\mathbf{z})}))/(2\eta)]$. Estimated efficiently via coarse-to-fine search + Multi-Level Monte Carlo (MLMC).
    - **Design Motivation**: Visualizing "what adversarial optimization did" in a 2D time-dimension map is a major methodological contribution. MLMC is necessary because direct sampling across $T$ steps for 6 toxicity dimensions is too expensive.

### Loss & Training

Total Loss = High-level GRPO loss + Low-level $\mathcal{J}_{low} = \mathbb{E}_\tau[\frac{1}{T}\sum_t(\mathcal{L}_{grp}^{low}(t) - \beta_t D_{KL}(\pi_\theta^{(t)}\|\pi_{ref}^{(t)}))]$. Key hyperparameters: $K = 4$ candidates, $M = 8$ rollouts, $\epsilon_p = 0.8$, $\tau_{sem} = 0.7$, $\beta_{high} = 0.02, \beta_t = 0.04$, PPO clip $\varepsilon = 0.001$. Training uses 20 denoising steps; inference uses 40 steps.

## Key Experimental Results

### Main Results

ASR (%) on LLaVA + RTP dataset ↑:

| Method | Any ↑ | Toxic ↑ | Obscene ↑ | Identity ↑ | Insult ↑ | CLIP ↑ |
|------|-------|---------|-----------|------------|----------|--------|
| Text-Only | 5.20 | 3.10 | 5.10 | 0.60 | 2.80 | – |
| Text + SD | 11.15 | 5.71 | 10.63 | 3.97 | 6.11 | 0.72 |
| PGJ | 14.86 | 7.85 | 13.98 | 3.43 | 8.09 | 0.71 |
| DiffZOO | 17.20 | 9.01 | 16.42 | 4.14 | 7.88 | 0.73 |
| ART | 18.62 | 9.22 | 17.54 | 6.45 | 8.94 | 0.75 |
| Ours w/ DDPO (Fixed budget) | 27.84 | 15.62 | 26.12 | 5.80 | 15.11 | 0.75 |
| **Ours ($w_{align}=0.2$)** | **31.36** | **17.10** | **29.73** | **6.14** | **15.95** | **0.78** |

Ours achieves 30.83 Any ASR on OOD PolygloToxicityPrompts vs ART's 22.01. Transferability holds for Qwen2.5-VL and Gemini-2.5-Pro.

### Ablation Study

| Configuration | Any ASR | Description |
|------|---------|------|
| Full STARE ($w_{align}=0.2$) | **31.36** | Complete method |
| Ours w/o LoRA | 22.04 | Removed low-level; velocity fine-tuning is the largest contributor |
| Ours w/o Edit | 25.56 | Removed high-level; prompt edit is significant |
| Ours w/o Align | 26.43 | Removed alignment reward; CLIP drops to 0.68 |
| Ours w/ DDPO | 27.84 | Hierarchical > Flattened RL |

### Key Findings

- **Optimization-Induced Phase Alignment**: Heatmaps show that while vanilla SD toxicity is diffuse, adversarial optimization concentrates identity/threat (concept-level) toxicity in early steps and obscene/insult (detail-level) in later steps. This is a real causal structure "induced" by RL.
- **Hierarchical > Flattened RL**: STARE outperforms STARE-w/-DDPO by 3.5% ASR. Flattened RL smears optimization pressure across the trajectory, failing to exploit the internal temporal structure.
- **CLIP Alignment Benefits ASR**: Maintaining alignment prevents images from collapsing into noise. Consistent image-prompt pairs ensure the VLM actually uses the image as context, facilitating jailbreaks.

## Highlights & Insights

- Reframing the denoising trajectory as an attack surface is highly innovative, moving T2I from a "black-box generator" to a "white-box temporal-semantic target."
- **Optimization-Induced Phase Alignment** suggests that early/late semantic emergence mechanisms can be "amplified" and "target-exploited" by adversarial optimization.
- MLMC utilizes hierarchical estimation to make attribution analysis feasible at $O(T \cdot D)$ costs.
- Using vec2text to reverse-map embedding edits into discrete text prompts avoids distribution shifts common in direct prompt embedding injection.

## Limitations & Future Work

- **White-box T2I requirement**: Requires access to full parameters (e.g., SD 3.5) for LoRA fine-tuning; not directly applicable to closed-source APIs like DALL-E 3.
- **Query Cost**: Obtaining 6-dimensional toxicity scores for each VLM query is expensive for commercial APIs.
- **Theoretical Grounding**: While phase alignment is empirically observed via perturbation, a precise analytical theory for "why" specific toxicity types bind to specific windows is still needed.
- **Model specific**: Rectified flow is required for the linear trajectory assumption; curves in DDIM/DDPM might distort the analysis.

## Related Work & Insights

- **vs. PGJ/DiffZOO/ART**: These focus on black-box prompt search; STARE manipulates both prompt and velocity field to exploit generation structure, doubling the ASR.
- **vs. RedDiffuser**: STARE introduces hierarchical optimization and phase-level analysis as differentiators.
- **vs. DDPO**: STARE's hierarchical structure proves more effective than flattened RL for diffusion models.
- **vs. Text Jailbreak (GCG)**: Text-only attacks lack the multi-modal channel; STARE reveals that the image channel is an underrated attack surface.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ (Treating trajectory as attack surface + Phase Alignment phenomenon are significant paradigm shifts).
- **Experimental Thoroughness**: ⭐⭐⭐⭐ (Multiple datasets, cross-model transfer, DDPO baselines; needs more defense-side comparison).
- **Writing Quality**: ⭐⭐⭐⭐ (Rigorous math for MDP/GRPO/MLMC, though the attribution section is technically dense).
- **Value**: ⭐⭐⭐⭐ (Foundation for both red-teaming and phase-aware safety monitoring).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Pareto-Guided Optimal Transport for Multi-Reward Alignment](pareto-guided_optimal_transport_for_multi-reward_alignment.md)
- [\[ICML 2026\] Diffusion Models Are Statistically Optimal for Learning Low-Dimensional Multi-Modal Distributions](diffusion_models_are_statistically_optimal_for_learning_low-dimensional_multi-mo.md)
- [\[ICCV 2025\] AutoPrompt: Automated Red-Teaming of Text-to-Image Models via LLM-Driven Adversarial Prompts](../../ICCV2025/image_generation/autoprompt_automated_red-teaming_of_text-to-image_models_via_llm-driven_adversar.md)
- [\[ICLR 2026\] Image Can Bring Your Memory Back: A Novel Multi-Modal Guided Attack against Image Generation Model Unlearning](../../ICLR2026/image_generation/image_can_bring_your_memory_back_a_novel_multi-modal_guided_attack_against_image.md)
- [\[CVPR 2026\] CognitionCapturerPro: Towards High-Fidelity Visual Decoding from EEG/MEG via Multi-modal Information and Asymmetric Alignment](../../CVPR2026/image_generation/cognitioncapturerpro_towards_highfidelity_visual_d.md)

</div>

<!-- RELATED:END -->
