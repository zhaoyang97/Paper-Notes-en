---
title: >-
  [Paper Note] VLM-Guided Group Preference Alignment for Diffusion-based Human Mesh Recovery
description: >-
  [CVPR 2026][Multimodal VLM][Human Mesh Recovery] This paper proposes a VLM-guided dual-memory self-reflective Critique Agent that generates group-level preference signals for diffusion-based human mesh recovery (HMR), followed by Group Preference Alignment fine-tuning of the diffusion model. The approach substantially improves in-the-wild HMR accuracy without requiring any 3D annotations.
tags:
  - CVPR 2026
  - Multimodal VLM
  - Human Mesh Recovery
  - diffusion model
  - VLM
  - GRPO
  - Preference Alignment
  - Critique Agent
date: 2026-05-08
content_hash: 2ffaf60727ee91a4
---

# VLM-Guided Group Preference Alignment for Diffusion-based Human Mesh Recovery

**Conference**: CVPR 2026
**arXiv**: [2602.19180](https://arxiv.org/abs/2602.19180)
**Institution**: Nanyang Technological University, HKUST(GZ), SenseTime Research, A*STAR
**Code**: To be confirmed
**Area**: Multimodal VLM
**Keywords**: Human Mesh Recovery, diffusion model, VLM, GRPO, Preference Alignment, Critique Agent

## TL;DR

This paper proposes a VLM-guided dual-memory self-reflective Critique Agent that generates group-level preference signals for diffusion-based human mesh recovery (HMR), followed by Group Preference Alignment fine-tuning of the diffusion model. The approach substantially improves in-the-wild HMR accuracy without requiring any 3D annotations.

## Background & Motivation

Monocular human mesh recovery (HMR) is an inherently ill-posed problem: a single 2D image can correspond to multiple valid 3D poses. Existing methods fall into three categories:

- **Optimization-based methods** (e.g., SMPLify): iterative optimization prone to local optima
- **Regression-based methods** (e.g., HMR, HybrIK): directly predict a single result, unable to handle depth/occlusion ambiguity
- **Probabilistic methods** (e.g., ScoreHypo, ADHMR): generate multiple hypotheses but often at the cost of accuracy

Diffusion-based HMR methods can generate diverse hypotheses but suffer from two critical deficiencies:

1. **Prediction–input inconsistency**: generated 3D meshes frequently deviate from 2D image evidence, particularly under occlusion and complex scenes
2. **Unreliable DPO supervision**: the HMR-Scorer used in ADHMR scores based solely on 2D joint features and can be fooled by poses that match silhouettes but are physically implausible; moreover, DPO only performs pairwise comparisons, ignoring the relative quality relationships among multiple predictions within a group

## Core Problem

How to provide high-quality preference supervision signals for diffusion-based HMR, enabling the model to learn to generate physically plausible and image-consistent human meshes on in-the-wild data without 3D ground truth?

## Method

### Overall Architecture

The framework consists of three core components:

1. **VLM-guided HMR Critique Agent** (Sec. 3.3): generates semantically-aware quality scores for predicted meshes
2. **HMR Group Preference Dataset Construction** (Sec. 3.4): automatically annotates group-level preferences using the Critique Agent
3. **Group Preference Alignment Training** (Sec. 3.5): introduces the GRPO paradigm into diffusion model fine-tuning

### 3.3 VLM-Guided HMR Critique Agent

#### Mechanism

Unlike conventional regression-based scorers that predict scores from 2D joint data, the proposed Critique Agent operates directly on rendered overlay images, simulating the judgment of a human expert. Given an RGB image $I$ and $n$ overlay images $\{\hat{I}_j\}_{j=1}^n$ of mesh predictions, the agent outputs a score $s_j \in [0, 100]$ and a one-sentence commentary $c_j$ for each overlay.

**Qwen3-VL-32B** is used as the VLM backbone.

#### 3.3.1 Dual-Memory Mechanism

Two complementary memory stores are designed:

| Memory Type | Stored Content | Data Structure | Role |
|-------------|---------------|----------------|------|
| **Rule Memory** $\mathcal{M}_R$ | Evaluation rule texts | $(t_i, T_i, N_i^u, N_i^s)$: rule text, semantic label, usage count, success count | Provides general evaluation criteria |
| **Prototype Memory** $\mathcal{M}_P$ | Previously evaluated exemplar cases | $(v_i, r_i, T_i)$: CLIP visual embedding, critique rationale with score, semantic label | Provides similar-case references |

**Dual-memory augmented scoring pipeline** (three steps):

**Step 1 — Prototype Retrieval**: The CLIP embedding $v_q$ of the query image is used to retrieve the top-$K$ historical cases with the highest cosine similarity from $\mathcal{M}_P$ as references.

**Step 2 — Rule Retrieval**: The most effective evaluation rules are selected via a hybrid score $\Psi_i$:

$$\Psi_i = \mathrm{R}(T_q, T_i) + \mathrm{U}_i$$

where semantic relevance $\mathrm{R}(T_q, T_i) = |T_q \cap T_i|$ rewards rules matching query labels. The UCB exploration score is:

$$\mathrm{U}_i = \rho_i + C\sqrt{\frac{\log N_{\text{total}}}{N_i^u + 1}}$$

Here $\rho_i = N_i^s / N_i^u$ is the historical success rate and $C$ is the exploration constant. This design balances exploitation of high-success-rate rules with exploration of infrequently used ones.

**Step 3 — Contextualized Scoring**: Retrieved rules and prototype rationales are dynamically assembled into a prompt, which is fed to the VLM to produce the final score and commentary.

#### 3.3.2 Reflective Knowledge Construction

Directly prompting the VLM to score produces unstable and inconsistent results. An **exploration phase** is therefore introduced to allow the agent to autonomously construct domain knowledge:

1. **Dual-memory augmented scoring**: score a batch of data and increment $N_i^u$ for used rules
2. **Prototype write-back**: store representative cases into $\mathcal{M}_P$
3. **Rule update**: compare the agent's score ranking with GT metrics via Spearman rank correlation; if the correlation exceeds threshold $\tau$, increment $N_i^s$ for the corresponding rules
4. **New rule mining (core)**: instruct the VLM to examine discrepancies between its outputs and GT metrics, proposing 1–2 new testable rules to be added to $\mathcal{M}_R$

**Evaluation phase**: memories and the learning loop are frozen; only dual-memory augmented scoring is performed to ensure consistency.

### 3.4 HMR Group Preference Dataset Construction

Dataset construction proceeds in two steps:

1. **Group generation**: for each training image $I$, the frozen pretrained diffusion reference model $\epsilon_{\text{ref}}$ samples $G$ times with different initial noise, yielding a diverse group of human mesh predictions $\{\mathbf{m}^i\}_{i=1}^G$
2. **Group-level scoring**: image $I$ and all $G$ predictions (rendered as 2D overlay images) are simultaneously fed to the Critique Agent to obtain intra-group consistent relative quality scores:

$$\{s^1, \ldots, s^G\} = \mathcal{C}_{\text{VLM}}(I, \mathbf{m}^1, \ldots, \mathbf{m}^G)$$

The resulting dataset is $\mathcal{G}_{\text{HMR}} = \{(I, (\mathbf{m}^1, s^1), \ldots, (\mathbf{m}^G, s^G))\}$. The entire process is fully automated with no human annotation.

### 3.5 Group Preference Alignment Training

#### From GRPO to Diffusion Models

GRPO was originally designed for alignment via stochastic decoding in LLMs, whereas diffusion models typically employ deterministic ODE samplers (e.g., DDIM). Introducing stochasticity via SDE sampling requires training along the entire diffusion trajectory, which is computationally expensive and degrades output quality.

The key innovation of this paper: **retain ODE sampling efficiency while extracting only the group-level preference signal from GRPO**.

#### Training Objective Derivation

**Step 1 — Compute intra-group advantages**: normalize scores $\{s^i\}_{i=1}^G$ from the preference dataset:

$$A_i = \frac{s_i - \text{mean}(\{s_i\}_{i=1}^G)}{\text{std}(\{s_i\}_{i=1}^G)}$$

**Step 2 — Advantage-weighted log-likelihood ratio**: treating the diffusion sampler as a conditional policy $p_\theta(\mathbf{m} | \mathbf{c})$, the optimization objective is:

$$\mathcal{L}(\theta) = -\mathbb{E}_{\mathbf{c}, \{\mathbf{m}^i\}} \left[\sum_{i=1}^G A(\mathbf{m}^i) \log \frac{p_\theta(\mathbf{m}^i | \mathbf{c})}{p_{\text{ref}}(\mathbf{m}^i | \mathbf{c})}\right]$$

**Step 3 — Diffusion surrogate loss**: using the reparameterization from Diffusion-DPO, the log-likelihood ratio is converted into a difference of denoising losses:

$$\log \frac{p_\theta(\mathbf{m}^i | \mathbf{c})}{p_{\text{ref}}(\mathbf{m}^i | \mathbf{c})} \approx T\lambda_t \mathbb{E}_{t, \epsilon}[L_{\text{DM}}^{\text{ref}}(\mathbf{x}_t^i, \epsilon) - L_{\text{DM}}^\theta(\mathbf{x}_t^i, \epsilon)]$$

**Final training loss**:

$$\mathcal{L}(\theta) = \mathbb{E}_{\mathbf{m} \sim \mathcal{G}_{\text{HMR}}, t, \epsilon} \; \beta T \lambda_t \sum_{i=1}^G \left[A(\mathbf{m}^i)(L_{\text{DM}}^\theta(\mathbf{x}_t^i, \epsilon) - L_{\text{DM}}^{\text{ref}}(\mathbf{x}_t^i, \epsilon))\right]$$

**Intuition**: meshes with high scores (positive advantage) are encouraged to achieve lower denoising loss than the reference model; meshes with low scores (negative advantage) are pushed in the opposite direction. No 3D ground-truth annotations are required throughout this process.

## Key Experimental Results

### Main Results (Tab. 1, selected)

| Method | Type | M | 3DPW MPJPE↓ | 3DPW PA-MPJPE↓ | H36M MPJPE↓ | H36M PA-MPJPE↓ |
|--------|------|---|------------|----------------|------------|----------------|
| ScoreHypo | Probabilistic | 100 | 63.0 | 37.6 | 38.4 | 26.0 |
| ADHMR | Probabilistic | 100 | 57.2 | 33.5 | 36.9 | 24.8 |
| **Ours** | Probabilistic | 100 | **52.5** | **31.5** | **35.0** | **23.9** |
| **Ours†** | Probabilistic | 100 | **49.9** | **31.9** | **34.3** | **23.5** |

- Ours vs. ADHMR (M=100): 3DPW MPJPE reduced by 8.2% (57.2→52.5)
- Ours† additionally uses InstaVariety in-the-wild data (preference signals only, no 3D labels), further reducing 3DPW MPJPE to 49.9

### Ablation Study (Tab. 2)

| Configuration | 3DPW PVE↓ | MPJPE↓ | PA-MPJPE↓ |
|---------------|-----------|--------|-----------|
| Base diffusion model | 73.4 | 63.0 | 37.6 |
| + Supervised fine-tuning | 70.2 | 61.3 | 36.5 |
| DPO + Critique Agent | 63.9 | 53.1 | 33.4 |
| Ours w/o Critique Agent (HMR-Scorer) | 65.4 | 54.9 | 34.7 |
| **Ours (full)** | **59.5** | **49.9** | **31.9** |

- Group preference alignment vs. DPO: MPJPE reduced by 6.0% (53.1→49.9), demonstrating the superiority of group-level signals over pairwise comparison
- Replacing the Critique Agent with HMR-Scorer leads to a notable performance drop, validating the importance of high-quality preference signals
- Supervised fine-tuning on noisy pseudo-labels yields only marginal improvement

### Critique Agent Evaluation

Removing the self-reflection mechanism (w/o self-reflection) causes the largest performance degradation across all metrics, demonstrating that reflective knowledge construction is critical to the stability of agent rankings.

## Highlights & Insights

1. **First VLM Critique Agent for HMR**: the dual-memory (rule + prototype) and self-reflection mechanism provide stronger 3D-aware evaluation than conventional 2D joint scorers, enabling detection of self-intersection, depth relation errors, and similar artifacts
2. **Elegant transfer of GRPO to diffusion models**: avoids the need for SDE sampling to introduce stochasticity, preserving ODE efficiency while extracting group-level preference signals with a clean and intuitive loss derivation
3. **In-the-wild fine-tuning without 3D ground truth**: effective fine-tuning on in-the-wild data such as InstaVariety using only relative preference signals from the Critique Agent, breaking the bottleneck of HMR's dependence on high-quality 3D annotations
4. **UCB exploration strategy**: rule retrieval draws on the UCB strategy from multi-armed bandits, automatically balancing exploitation of validated rules with exploration of novel ones

## Limitations & Future Work

1. **VLM inference cost**: using Qwen3-VL-32B as the Critique Agent incurs high inference overhead during preference dataset construction, limiting scalability
2. **Exploration phase dependency on GT**: rule learning and validation still require 3D ground truth from synthetic/laboratory data; the agent's evaluation capability may be affected by the distribution of exploration data
3. **Impact of group size**: training uses $G=20$; whether larger groups yield better preference signals remains insufficiently explored
4. **Single-person limitation**: the framework is based on the SMPL single-person model and does not address extension to multi-person scenarios

## Related Work & Insights

- **vs. ADHMR**: ADHMR applies DPO with an HMR-Scorer for pairwise preference learning, where the scorer relies on 2D joint features and is susceptible to occlusion-induced errors; the proposed VLM Critique Agent provides more reliable 3D-aware scores, and group preference alignment outperforms pairwise DPO
- **vs. ScoreHypo**: ScoreHypo uses an auxiliary selection network to pick the best hypothesis but does not improve the generative distribution; this work directly optimizes the diffusion model's sampling strategy
- **vs. GRPO-based diffusion methods** (DAPO, D-GRPO): these methods introduce stochasticity via SDE sampling and require training along entire trajectories; the proposed offline GRPO with ODE sampling is more efficient

**Broader implications**:
- The dual-memory + self-reflective VLM Critique Agent is a general paradigm transferable to other 3D tasks requiring automatic quality assessment (e.g., hand reconstruction, scene reconstruction)
- The group preference alignment framework is agnostic to the specific scorer and can in principle be combined with any quality assessment method
- Leveraging VLM 3D semantic priors for evaluation represents the first successful application of LLM-as-a-Judge to visual 3D tasks

## Rating

- Novelty: ⭐⭐⭐⭐ — Dual innovation of VLM Critique Agent and group preference alignment; the offline GRPO-to-diffusion transfer is elegantly designed
- Experimental Thoroughness: ⭐⭐⭐⭐ — Multi-benchmark comparisons, detailed ablations, qualitative analysis, and independent Critique Agent evaluation
- Writing Quality: ⭐⭐⭐⭐ — Clear structure with complete mathematical derivations
- Value: ⭐⭐⭐⭐ — Annotation-free in-the-wild fine-tuning capability carries significant practical importance

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] GLYPH-SR: Can We Achieve Both High-Quality Image Super-Resolution and High-Fidelity Text Recovery via VLM-Guided Latent Diffusion Model?](../../ICLR2026/multimodal_vlm/glyph-sr_can_we_achieve_both_high-quality_image_super-resolution_and_high-fideli.md)
- [\[CVPR 2026\] DSCA: Dynamic Subspace Concept Alignment for Lifelong VLM Editing](dsca_dynamic_subspace_concept_alignment_for_lifelong_vlm_editing.md)
- [\[AAAI 2026\] Towards Human-AI Accessibility Mapping in India: VLM-Guided Annotations and POI-Centric Analysis in Chandigarh](../../AAAI2026/multimodal_vlm/towards_human-ai_accessibility_mapping_in_india_vlm-guided_annotations_and_poi-c.md)
- [\[CVPR 2026\] AGFT: Alignment-Guided Fine-Tuning for Zero-Shot Adversarial Robustness of Vision-Language Models](agft_alignment-guided_fine-tuning_for_zero-shot_adversarial_robustness_of_vision.md)
- [\[NeurIPS 2025\] Guiding Cross-Modal Representations with MLLM Priors via Preference Alignment](../../NeurIPS2025/multimodal_vlm/guiding_cross-modal_representations_with_mllm_priors_via_preference_alignment.md)

<!-- RELATED:END -->
