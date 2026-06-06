---
title: >-
  [Paper Note] STARE: Step-wise Temporal Alignment and Red-teaming Engine for Multi-modal Toxicity Attack
description: >-
  [ICML 2026][LLM Safety][Multi-modal red-teaming] This paper treats the entire denoising trajectory of T2I models as the "attack surface" for VLM red-teaming attacks. It proposes a hierarchical RL framework (STARE) combin…
tags:
  - "ICML 2026"
  - "LLM Safety"
  - "Multi-modal red-teaming"
  - "diffusion trajectory attack"
  - "hierarchical RL"
  - "GRPO"
  - "temporal alignment analysis"
date: 2026-05-08
content_hash: 07a320bd324bc57b
---

# STARE: Step-wise Temporal Alignment and Red-teaming Engine for Multi-modal Toxicity Attack

**Conference**: ICML 2026  
**arXiv**: [2605.00699](https://arxiv.org/abs/2605.00699)  
**Code**: https://github.com/henrymao2004/STARE.git (available)  
**Area**: Image Generation / Multi-modal VLM Security / Red-teaming Attacks  
**Keywords**: Multi-modal red-teaming, diffusion trajectory attack, hierarchical RL, GRPO, temporal alignment analysis

## TL;DR
This paper treats the entire denoising trajectory of T2I models as the "attack surface" for VLM red-teaming attacks. It proposes a hierarchical RL framework (STARE) combining a high-level prompt editor and low-level GRPO fine-tuning of rectified-flow models. This approach not only improves attack success rate by 68% over SOTA, but also reveals a novel phenomenon—Optimization-Induced Phase Alignment: adversarial optimization automatically binds "conceptual toxicity" to early denoising and "detail toxicity" to later stages, transforming the chaotic toxicity formation process into several predictable "vulnerability time windows."

## Background & Motivation

**Background**: Toxic continuation attacks on VLMs represent a major multi-modal security threat—attackers use T2I models to generate adversarial images, paired with a text prefix, to induce highly toxic VLM continuations. Existing red-teaming methods (PGJ, DiffZOO, ART, RedDiffuser, etc.) treat T2I as a black box—focusing only on terminal toxicity scores, without considering at which step toxic semantics emerge.

**Limitations of Prior Work**: The terminal-only perspective leads to a "temporal opacity" problem. Diffusion models inherently exhibit a coarse-to-fine semantic emergence mechanism (early stages determine layout/concept, later stages determine details), but current red-teaming ignores this temporal structure, resulting in sparse global rewards that lack attribution—making it unclear "why" an adversarial image can jailbreak, and preventing precise defense interventions.

**Key Challenge**: (1) Black-box optimization vs. white-box attack surface: treating T2I as a black box only yields final toxicity, but intermediate steps in diffusion models have exploitable semantic emergence patterns; (2) Flat RL vs. hierarchical semantic structure: standard RL (e.g., DDPO) treats the entire generation as a single policy, unable to match the natural division between "early layout / late details"; (3) Conceptual toxicity vs. detail toxicity: real-world toxicity includes "concept-level" (identity/threat, requiring early seeds) and "detail-level" (obscene/insult, requiring late amplification), but baselines apply uniform pressure.

**Goal**: (1) Design a hierarchical RL framework that can explicitly manipulate early and late stages of the denoising trajectory for end-to-end toxicity attacks on VLMs; (2) Use temporal alignment analysis to reveal the impact of adversarial optimization on diffusion temporal structure; (3) Push ASR to SOTA.

**Key Insight**: The authors use rectified flow as the base (its velocity field is explicit and trajectories are nearly linear, facilitating temporal attribution analysis). "Prompt editing for semantic subgoals" and "velocity field fine-tuning for detail amplification" are mapped to high-level/low-level MDPs—naturally corresponding to early/late semantic emergence in diffusion.

**Core Idea**: The high-level prompt editor seeds "conceptual toxicity subgoals" in embedding space, while the low-level GRPO fine-tunes the rectified-flow velocity field to amplify "detail toxicity." Both policies share the same toxicity reward. Temporal attribution analysis (MLMC + block perturbation) demonstrates that this hierarchical structure aligns with real early/late vulnerability windows.

## Method

### Overall Architecture

Input: root prompt $p$, white-box T2I model (SD 3.5-Medium + LoRA $r=16$), query-level black-box VLM (LLaVA-v1.6-mistral-7b). Pipeline: (1) High-level adds noise perturbations to $p$ embedding to obtain $K$ candidate edits $e_p + \delta_j$, decoded via vec2text into $K$ subgoal prompts $p'^{(j)}$; (2) Low-level, for each $p'^{(j)}$, uses the current LoRA-augmented velocity field $v_\theta$ to run $M$ image rollouts (using Marginal-Preserving Stochastic SDE discretization for exploration); (3) VLM evaluates each image + continuation prompt for toxicity score, combines with CLIPScore alignment reward to form terminal reward; (4) Both policies are updated with GRPO objective (group-normalized advantage): high-level group is the average reward + edit reward over $K$ candidates, low-level group is the individual reward over all $K \times M$ rollouts. The pipeline forms a dual-loop: "semantic subgoal → image generation → VLM continuation → toxicity → backprop to both policies."

### Key Designs

1. **Hierarchical MDP: High-Level Prompt Editor + Low-Level Velocity Fine-tuning**:

    - **Function**: Separates semantic injection and detail amplification into two policies at different temporal scales, corresponding to early concept seeding and late detail refinement in diffusion.
    - **Mechanism**: High-level MDP is a single-step decision—state is prompt embedding $e_p$, action is edit vector $\delta$, policy $\pi_{edit}(\delta|e_p)$ is an encoder-decoder Transformer outputting $\mu_j$, projected onto an $\ell_2$ ball $\delta_j = \epsilon_p \cdot \mu_j / \max(\|\mu_j\|_2, \epsilon_p)$ ($\epsilon_p = 0.8$). Low-level MDP is iterative denoising—state $s_t = (x_t, t, c)$, action $a_t = x_{t - \Delta t}$, policy $\pi_\theta(a_t|s_t) = \mathcal{N}(\mu_\theta, \sigma_t^2 I)$, where $\mu_\theta = x_t - v_\theta(x_t, t, c) \Delta t$; MPS SDE discretization $x_{t - \Delta t} = x_t - v_\theta \Delta t + \sigma_t \varepsilon$ ensures exploration.
    - **Design Motivation**: Early T2I stages mainly determine semantics/layout, later stages determine details—naturally matching "prompt edits for semantics" and "velocity edits for image statistics." Assigning to two policies allows each to focus on its optimal temporal segment, outperforming flat RL (e.g., DDPO) by 21% ASR in experiments.

2. **Dual-level GRPO Optimization + Marginal Reward Combination**:

    - **Function**: Uses group-normalized advantage instead of absolute reward to reduce variance under sparse rewards; high-level includes an auxiliary "edit semantic preservation" reward.
    - **Mechanism**: GRPO loss $\mathcal{L}_{grp}(r_t, \hat A, \varepsilon) = \min(r_t \hat A, \mathrm{clip}(r_t, 1-\varepsilon, 1+\varepsilon) \hat A)$, where $r_t = \pi_\theta(a_t|s_t)/\pi_{old}(a_t|s_t)$, group-normalized advantage $\hat A_i = (X_i - \mu_{grp})/(\sigma_{grp} + \epsilon)$. High-level group uses average reward over $K$ candidates plus edit reward $\mathcal{R}_{high}^{(j)} = \bar R_j + \mathcal{R}_{edit}^{(j)}$, where $\mathcal{R}_{edit}^{(j)} = \lambda_{sem}[s_{SBERT}(e_p, e_p + \delta_j) - \tau_{sem}]_+ + \lambda_{recon}/(1 + \|e_p + \delta_j - \mathrm{emb}(p'^{(j)})\|^2)$, encouraging both semantic similarity to the original prompt and consistency between embedding edit and vec2text-decoded text. Low-level group is all $K \times M$ rollout rewards $R^{(j,m)} = R_{tox}^{(j,m)} + w_{align} R_{align}^{(j,m)}$, with per-step KL $D_{KL}(\pi_\theta^{(t)} \| \pi_{ref}^{(t)}) = \tfrac{1}{2\sigma_t^2}\|\mu_\theta - \mu_{ref}\|^2$ to stabilize mean drift.
    - **Design Motivation**: Toxicity reward is a sparse and noisy terminal reward; group normalization greatly reduces variance compared to absolute reward. Methods like Flow-DPO requiring preference datasets are too costly for the dual-level structure; GRPO is the most lightweight choice. Edit reward prevents high-level from editing prompts into irrelevant directions.

3. **Temporal Alignment Analysis (MLMC Temporal Attribution)**:

    - **Function**: Quantifies "which denoising step contributes most to which toxicity type" as a $T \times D$ heatmap, verifying that the hierarchical structure indeed corresponds to different temporal windows.
    - **Mechanism**: Define net toxicity score $\mathcal{R}_d(I, p) = R_d(\mathrm{VLM}(I, p)) - R_d(\mathrm{VLM}(\mathrm{null}, p))$ to isolate the image's marginal contribution; define sensitivity to time block $B$ as $\Delta_B^{(d)} = \mathbb{E}_{\mathbf{z}}[(\mathcal{R}_d(G^{(B, +\eta\mathbf{z})}) - \mathcal{R}_d(G^{(B, -\eta\mathbf{z})}))/(2\eta)]$ (finite difference with symmetric perturbation within the block). Use coarse-to-fine search + Multi-Level Monte Carlo $\hat\Delta_B^{MLMC} = \tfrac{1}{M_0}\sum \hat\Delta_B^{(0)} + \sum_\ell \tfrac{1}{M_\ell}\sum(\hat\Delta_B^{(\ell)} - \hat\Delta_B^{(\ell-1)})$ for efficient estimation; finally, for singleton $B = \{t\}$, obtain TemporalScore$(t, d) = \hat\Delta_{\{t\}}^{(d), MLMC}$, rescaled to $[-1, 1]$ for heatmap visualization.
    - **Design Motivation**: Converts "what adversarial optimization did" from a black-box reward number into a 2D time-dimension visualization—this is the paper's main methodological contribution. MLMC is necessary—since 6 toxicity dimensions × $T$ steps is too expensive for direct sampling, MLMC uses low-fidelity hierarchical estimates plus a few high-fidelity corrections to significantly reduce variance.

### Loss & Training

Total loss = High-level GRPO loss + Low-level $\mathcal{J}_{low} = \mathbb{E}_\tau[\tfrac{1}{T}\sum_t(\mathcal{L}_{grp}^{low}(t) - \beta_t D_{KL}(\pi_\theta^{(t)}\|\pi_{ref}^{(t)}))]$. Key hyperparameters: $K = 4$ candidates, $M = 8$ rollouts, $\epsilon_p = 0.8$, $\tau_{sem} = 0.7$, $\lambda_{sem} = 1.0, \lambda_{recon} = 0.1$, $\beta_{high} = 0.02, \beta_t = 0.04$, PPO clip $\varepsilon_{low} = \varepsilon_{high} = 0.001$. Training uses 20 denoising steps, inference 40 steps.

## Key Experimental Results

### Main Results

On LLaVA + RTP dataset, ASR (%) ↑:

| Method | Any ↑ | Toxic ↑ | Obscene ↑ | Identity ↑ | Insult ↑ | CLIP ↑ |
|--------|-------|---------|-----------|------------|----------|--------|
| Text-Only | 5.20 | 3.10 | 5.10 | 0.60 | 2.80 | – |
| Text + SD | 11.15 | 5.71 | 10.63 | 3.97 | 6.11 | 0.72 |
| PGJ | 14.86 | 7.85 | 13.98 | 3.43 | 8.09 | 0.71 |
| DiffZOO | 17.20 | 9.01 | 16.42 | 4.14 | 7.88 | 0.73 |
| ART | 18.62 | 9.22 | 17.54 | 6.45 | 8.94 | 0.75 |
| STARE w/ DDPO (same white-box budget) | 27.84 | 15.62 | 26.12 | 5.80 | 15.11 | 0.75 |
| **STARE (Ours, $w_{align}=0.2$)** | **31.36** | **17.10** | **29.73** | 6.14 | 15.95 | 0.78 |

On OOD PolygloToxicityPrompts test, STARE Any 30.83 vs ART 22.01, demonstrating generalization. Transfer to Qwen2.5-VL and Gemini-2.5-Pro also maintains significant lead.

### Ablation Study

| Configuration | Any ASR | Notes |
|---------------|---------|-------|
| Full STARE ($w_{align}=0.2$) | **31.36** | Full method |
| STARE w/o LoRA (remove low-level) | 22.04 | -9.32, showing velocity fine-tuning is the largest contributor |
| STARE w/o Edit (remove high-level) | 25.56 | -5.80, prompt edit is the next largest contributor |
| STARE w/o Align (remove alignment reward) | 26.43 | -4.93, CLIP drops to 0.68 |
| STARE w/ DDPO (replace with flat RL) | 27.84 | -3.52, showing hierarchical > flat |

### Key Findings

- **Optimization-Induced Phase Alignment**: Temporal attribution heatmaps show that vanilla SD's toxicity contribution is diffuse over time, but after adversarial optimization, identity/threat (concept-level) toxicity concentrates in early timesteps, obscene/insult (detail-level) toxicity in late timesteps, with almost no overlap. This is not a side effect of the hierarchical design, but a real temporal pattern "induced" by RL optimization—targeting early windows only suppresses concept toxicity, late windows only suppress detail toxicity, confirming causality.
- **Hierarchical > Flat RL**: STARE outperforms STARE w/ DDPO (same white-box budget but flat) by 3.5% ASR, and the temporal window structure is clearer; DDPO's optimization pressure is smeared across the trajectory, failing to exploit diffusion's intrinsic temporal structure.
- **Strong Transfer**: Maintains ASR lead across different VLMs (Qwen2.5-VL, Gemini-2.5-Pro, GPT-5.4) and T2I generators (FLUX.1-dev), showing the attack is not overfitting to specific victims or "trick prompts."
- **CLIP align actually improves ASR**: Intuitively, "preserving CLIP alignment" might restrict adversarial freedom, but $w_{align}=0.2$ yields the highest ASR. The authors explain that alignment prevents images from collapsing into meaningless noise (which is actually harder to trigger toxic continuation), and maintaining image-prompt consistency ensures VLMs use the image as real context.

## Highlights & Insights

- The reframing of "treating the denoising trajectory itself as the attack surface" is highly innovative—transforming T2I from a "black-box image generator" into a "white-box temporal-semantic structure exploit target," opening new directions for attacks and defenses based on diffusion temporal structure.
- The Optimization-Induced Phase Alignment phenomenon is more valuable than the attack numbers—it shows that the early/late semantic emergence mechanism of diffusion models is not just empirical, but a real causal structure that can be "amplified" and "exploited" by adversarial optimization. For defense, this suggests phase-specific monitoring (concept-level filters in early timesteps, detail-level filters in late timesteps), greatly reducing defense costs.
- MLMC's hierarchical estimation reduces variance, making attribution analysis feasible at $O(T \cdot D \cdot M)$ cost—this is the engineering key to scaling perturbation analysis.
- Using vec2text to invert embedding edits back to text prompts is an ingenious engineering solution: it allows high-level optimization in continuous embedding space, but the final T2I input remains discrete text, avoiding distribution mismatch from direct prompt embedding injection.

## Limitations & Future Work

- The white-box T2I assumption is strong (requires access to all SD 3.5 parameters for LoRA fine-tuning), not applicable to fully black-box T2I (DALL-E 3, Midjourney); transfer experiments were done on FLUX.1-dev but not on true black-box commercial APIs.
- The VLM is assumed to be query-only black-box, but the reward signal requires 6-dimensional toxicity per query, making commercial API usage costly (no query budget analysis provided).
- Causal evidence for Optimization-Induced Phase Alignment comes from perturbation experiments, but the authors do not provide an analytic theory for "alignment strength"—why 6 toxicity dimensions, why early/late stages correspond to concept/detail, remain empirical observations.
- The rectified flow assumption is necessary (explicit velocity + near-linear trajectories); for models with high trajectory curvature (DDIM/DDPM), temporal attribution may be distorted.
- Red-teaming ethics: a 31% success rate poses a significant threat to LLaVA; the paper includes a content warning but no disclosure timeline.
- Group sizes $K = 4, M = 8$ are limited by GPU; larger groups could further reduce GRPO variance but would sharply increase training cost.

## Related Work & Insights

- **vs PGJ / DiffZOO / ART**: All are prompt-side black-box search + frozen T2I, unable to exploit generation temporal structure; STARE manipulates both prompt edit and velocity field, doubling ASR.
- **vs RedDiffuser (Wang et al. 2025a)**: Also steers diffusion but lacks hierarchy and phase-level analysis; STARE's hierarchical structure and temporal attribution are differentiators.
- **vs DDPO (Black et al. 2024)**: DDPO is flat diffusion RL; STARE-w/-DDPO ablation shows hierarchy outperforms flat by 3.5% ASR and yields clearer temporal structure.
- **vs Flow-GRPO (Liu et al. 2025)**: The low-level GRPO idea is similar, but Flow-GRPO is single-level; STARE embeds it within a high-level prompt editor for a dual-level approach.
- **vs Text jailbreak (GCG, etc.)**: Pure text jailbreak lacks the image channel and cannot trigger multi-modal toxic-continuation attacks; STARE reveals the "image channel as an underestimated attack surface."

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Treating diffusion trajectory as attack surface + Phase Alignment phenomenon are paradigm-shifting innovations.
- Experimental Thoroughness: ⭐⭐⭐⭐ Dual datasets + three VLM transfers + DDPO compute-matched baseline + full ablation + MLMC temporal attribution; lacks query budget and defense comparison.
- Writing Quality: ⭐⭐⭐⭐ Hierarchical MDP / GRPO / MLMC formulas are rigorous, threat model is clear; temporal attribution is math-heavy but Figures 1/3 aid readability.
- Value: ⭐⭐⭐⭐ Provides both an attack tool and a foundation for defense design (phase-aware monitoring) for the multi-modal safety community, but responsible disclosure is needed.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Stable-GFlowNet: Toward Diverse and Robust LLM Red-Teaming via Contrastive Trajectory Balance](stable-gflownet_toward_diverse_and_robust_llm_red-teaming_via_contrastive_trajec.md)
- [\[CVPR 2026\] Multi-Paradigm Collaborative Adversarial Attack Against Multi-Modal Large Language Models](../../CVPR2026/llm_safety/multi-paradigm_collaborative_adversarial_attack_against_multi-modal_large_langua.md)
- [\[ICML 2026\] OTora: A Unified Red Teaming Framework for Reasoning-Level Denial-of-Service in LLM Agents](otora_a_unified_red_teaming_framework_for_reasoning-level_denial-of-service_in_l.md)
- [\[ICLR 2026\] Supervised Reinforcement Learning: From Expert Trajectories to Step-wise Reasoning](../../ICLR2026/llm_safety/supervised_reinforcement_learning_from_expert_trajectories_to_step-wise_reasonin.md)
- [\[ACL 2026\] STAR-Teaming: A Strategy-Response Multiplex Network Approach to Automated LLM Red Teaming](../../ACL2026/llm_safety/star-teaming_a_strategy-response_multiplex_network_approach_to_automated_llm_red.md)

</div>

<!-- RELATED:END -->
