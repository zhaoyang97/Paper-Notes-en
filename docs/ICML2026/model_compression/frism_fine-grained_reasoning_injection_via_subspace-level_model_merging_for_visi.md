---
title: >-
  [Paper Note] FRISM: Fine-Grained Reasoning Injection via Subspace-Level Model Merging for Vision–Language Models
description: >-
  [ICML 2026][Model Compression][Model Merging] FRISM refines "VLM × LRM merging" from layer-level granularity to SVD subspace-level granularity. It utilizes the SVD subspaces of LRM task vectors as reasoning priors and em…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "Model Merging"
  - "SVD Subspace"
  - "Reasoning Injection"
  - "Vision Preservation"
  - "Unlabeled Self-Distillation"
date: 2026-05-08
content_hash: ff7ac246acf9401f
---

# FRISM: Fine-Grained Reasoning Injection via Subspace-Level Model Merging for Vision–Language Models

**Conference**: ICML 2026  
**arXiv**: [2601.21187](https://arxiv.org/abs/2601.21187)  
**Code**: None  
**Area**: Multimodal VLM / Model Merging / Reasoning Injection  
**Keywords**: Model Merging, SVD Subspace, Reasoning Injection, Vision Preservation, Unlabeled Self-Distillation

## TL;DR
FRISM refines "VLM × LRM merging" from layer-level granularity to SVD subspace-level granularity. It utilizes the SVD subspaces of LRM task vectors as reasoning priors and employs an unlabeled self-distillation process (preserving vision via KL-divergence + maximizing spectral magnitude for reasoning absorption) with learnable gates to find optimal injection intensities, significantly enhancing VL reasoning performance without substantial vision degradation.

## Background & Motivation
**Background**: VLMs (Qwen2.5-VL, LLaVA, InternVL, etc.) possess strong general capabilities but exhibit clear reasoning shortcomings. LRMs (DeepSeek-R1, OpenAI-o1) excel in math, logic, and programming tasks. Transferring reasoning from LRMs to VLMs follows two paths: ① Large-scale retraining based on RL/SFT; ② Model Merging, which has near-zero training cost and requires no labeled data, thus being widely explored (e.g., BR2V, FRANK, IP-Merging).

**Limitations of Prior Work**: Existing merging methods primarily operate at the "layer" granularity—applying a unified mixing coefficient per layer like $\lambda_{\text{vlm}}\tau_{\text{vlm}}+\lambda_{\text{lrm}}\tau_{\text{lrm}}$. Experiments in Figure 2 show that tuning a single coefficient always results in a vision–reasoning trade-off: "either vision drops, or reasoning remains weak."

**Key Challenge**: By performing SVD on the task vectors of DeepSeek-R1-Distill-Qwen-7B and injecting them into Qwen2.5-VL rank-by-rank, the authors found that "optimal scaling coefficients vary significantly across different rank subspaces" (Figure 3). Some subspaces peak at $\lambda=0.1$, while others require much higher values. A layer-wide $\lambda$ inevitably entangles this heterogeneity, introducing harmful vision noise alongside useful reasoning. In other words, **the layer is not the atomic unit of capability**; the subspace is.

**Goal**: To refine merging granularity to the SVD subspace level, allowing the model to automatically determine which subspaces should be strongly injected and which should be suppressed, without relying on any VL reasoning labels.

**Key Insight**: Treat the SVD decomposition of LRM task vectors directly as "reasoning prior subspaces." Freeze $\mathbf{U}, \mathbf{S}, \mathbf{V}$ and learn only a per-rank gate vector $\mathbf{g}^l$. Use "unlabeled self-distillation + spectral magnitude maximization" to allow the gates to automatically find the equilibrium between "maximum injection" and "minimum vision loss."

**Core Idea**: Open gates for each SVD subspace within every layer. The "dual-objective + subspace gating" automatically filters out subspaces that damage vision while retaining reasoning subspaces orthogonal to vision.

## Method

### Overall Architecture
The FRISM workflow consists of two stages. **Stage 1 (Offline Decomposition and Initialization)**: Define task vectors for LRM and VLM as $\tau_{\text{vlm}}=\theta_{\text{vlm}}-\theta_{\text{base}}$ and $\tau_{\text{lrm}}=\theta_{\text{lrm}}-\theta_{\text{base}}$. Perform SVD on $\tau_{\text{lrm}}^l$ for each linear layer to obtain $\mathbf{U}^{(l)}, \mathbf{S}^{(l)}, \mathbf{V}^{(l)\top}$ and freeze them. Introduce a zero-initialized learnable gate $\mathbf{g}^l\in\mathbb{R}^r$ for each layer. **Stage 2 (Online Injection and Training)**: Use the merged model $\theta_{\text{merged}}^l=\theta_{\text{vlm}}^l+\lambda_{\text{lrm}}\cdot\mathbf{U}^{(l)}(\sigma(\mathbf{g}^l)\odot\mathbf{S}^{(l)})\mathbf{V}^{(l)\top}$ for unlabeled self-distillation. Using the original VLM as the teacher, encourage the student to match teacher outputs (KL distance) on pure vision perception data (e.g., VizWiz VQA). Simultaneously, add a loss term to maximize the spectral magnitude of the injected subspaces. This training updates only the gate $\mathbf{g}^l$, which is extremely small in scale and converges quickly.

### Key Designs

1. **Subspace-Level Merging Paradigm (Eq. 7)**:
    - **Function**: Assigns an independent scaling coefficient to each of the $r$ subspaces within a layer, fundamentally escaping the coupling dilemma of "one $\lambda$ per layer."
    - **Mechanism**: After performing SVD on $\tau_{\text{lrm}}^l$ and freezing $\mathbf{U}, \mathbf{S}, \mathbf{V}$, only $\mathbf{g}^l\in\mathbb{R}^r$ is learned. After passing through a Sigmoid, $\sigma(\mathbf{g}^l)\in(0,1)^r$ is element-wise multiplied with original singular values $\mathbf{S}$ to form "effective singular values" $\mathbf{S}_{\text{eff}}=\sigma(\mathbf{g}^l)\odot\mathbf{S}$. The merged weights are $\theta_{\text{merged}}^l=\theta_{\text{vlm}}^l+\lambda_{\text{lrm}}\,\mathbf{U}^{(l)}\mathbf{S}_{\text{eff}}^{(l)}\mathbf{V}^{(l)\top}$.
    - **Design Motivation**: The low-rank structure of LRM task vectors aligns with empirical observations that "reasoning concentrates on few directions" (Cai 2025, Ping 2024, Sharma 2024). Keeping the basis fixed while modifying intensity preserves the semantic direction of reasoning while allowing fine-grained adjustment—a classic "frozen basis, learned spectrum" approach.

2. **Unlabeled Self-Distillation: Vision Preservation Objective (Eq. 8)**:
    - **Function**: Constrains the merged model to avoid output distribution shifts on pure vision tasks in the absence of VL reasoning labels, keeping vision degradation within acceptable bounds.
    - **Mechanism**: The teacher is the original VLM $\theta_{\text{vlm}}$ (frozen), and the student is the merged model $\theta_{\text{vlrm}}(\mathbf{g})$. Minimize KL on a calibration dataset $\mathcal{D}$ (e.g., VizWiz VQA): $\mathcal{L}_{\text{distill}}=\mathbb{E}_{x\sim\mathcal{D}}\mathrm{KL}\!\left(P(\cdot|x;\theta_{\text{vlm}})\,\|\,P(\cdot|x;\theta_{\text{vlrm}})\right)$.
    - **Design Motivation**: VL reasoning data is scarce and unevenly distributed; direct supervision is risky. Self-distillation using the original VLM as a reference transforms "vision preservation" into a data-cheap, clean-objective constraint, turning the merging problem into "finding the strongest reasoning injection within a KL radius."

3. **Spectral Magnitude Maximization for Reasoning Absorption + Total Objective (Eq. 9–10)**:
    - **Function**: Prevents gates from collapsing to a trivial "zero injection" solution by actively encouraging $\mathbf{S}_{\text{eff}}$ to be as large as possible, stacking as much LRM subspace into the VLM as possible.
    - **Mechanism**: Define $\mathcal{L}_{\text{inject}}=-\sum_l\|\mathbf{S}_{\text{eff}}^{(l)}\|^2=-\sum_l\|\sigma(\mathbf{g}^{(l)})\odot\mathbf{S}^{(l)}\|^2$; stronger injection results in lower loss. The combined objective is $\mathcal{L}=\mathcal{L}_{\text{distill}}+\alpha\mathcal{L}_{\text{inject}}$. A second-order expansion illustrates that under the Hessian $\mathbf{H}=\nabla^2\mathcal{L}_{\text{vis}}$ and the assumption of decoupled SVD subspaces, $\partial\mathcal{L}/\partial\lambda_i\approx(h_i-2\alpha\|B_i\|_F^2)\lambda_i$. If the vision curvature $h_i$ of a subspace exceeds the injection benefit $2\alpha\|B_i\|_F^2$, the gate suppresses it; otherwise, it opens.
    - **Design Motivation**: This combination acts as an "automatic filter": subspaces orthogonal to vision perception (low $h_i$) are permitted, while high-curvature subspaces damaging to vision are closed. This mechanism solves the trade-off automatically using data priors and spectral structure without reasoning supervision.

### Loss & Training
Only the gate $\mathbf{g}^l$ is trained, representing negligible parameters compared to the base model. The total loss $\mathcal{L}=\mathcal{L}_{\text{distill}}+\alpha\mathcal{L}_{\text{inject}}$, where $\alpha$ controls injection intensity. $\mathcal{L}_{\text{inject}}$ is normalized before training due to scale differences across models (Appendix H). Only the LLM portion of the VLM layers participates in merging; the vision tower and projector remain unchanged.

## Key Experimental Results

### Main Results: Multi-benchmark Average Scores for Qwen2.5-VL × LRM Merging (Tab. 1)

| Method | VL Reasoning Avg | VL Perception Avg |
|--------|------------------|-------------------|
| **3B Merging SmallThinker-3B** | | |
| Base | 33.2 | 79.7 |
| Task Arithmetic Best $\lambda$ | 33.0 | 79.8 |
| Ties-Merging | 31.6 | 77.0 |
| IP-Merging Best $T$ | 32.2 | 77.0 |
| **FRISM** | **35.0 (+1.8)** | 79.7 |
| **7B Merging DeepSeek-R1-Distill-Qwen-7B** | | |
| Base | 47.4 | 82.9 |
| Task Arithmetic Best $\lambda$ | 47.8 (collapsed at high $\lambda$) | 82.4 |
| Ties-Merging | 45.3 | 78.9 |
| IP-Merging Best $T$ | 47.7 | 82.3 |
| **FRISM** | **49.4 (+2.0)** | **83.0** |

### Subspace-level Diagnosis (Fig. 3)

| Experiment | Key Finding |
|------------|-------------|
| Iterative injection of different rank subspaces | Different ranks peak at different $\lambda$ values, proving subspace heterogeneity and the suboptimality of "layer-wise single $\lambda$." |
| Standard layer-wise merging | Shows a significant performance gap compared to subspace-level optimization; layer granularity cannot accommodate multiple optimal $\lambda$ values simultaneously. |

### Vision–Reasoning Trade-off (Fig. 2)
- In the 2D space of "VL Reasoning vs. VL Perception," Task Arithmetic and IP-Merging form a clear trade-off curve (improving one sacrifices the other).
- FRISM jumps to the upper-right corner of the curve, proving the gate successfully filters out subspaces that damage vision while contributing little to reasoning.

### Key Findings
- In 7B merging, Task Arithmetic experiences a "vision cliff" starting from $\lambda=0.15$ (POPE drops from 86.4 to 73.9). FRISM maintains vision performance comparable to the Base while gaining ~2pt in reasoning—direct evidence of the success of subspace-level refinement.
- Removing $\mathcal{L}_{\text{inject}}$ causes gates to shrink toward negative infinity (no injection), proving that "active amplification" is necessary as an observable proxy for reasoning in the absence of labels.
- The second-order expansion $\partial\mathcal{L}/\partial\lambda_i\approx(h_i-2\alpha\|B_i\|_F^2)\lambda_i$ provides interpretable filtering rules: high-curvature vision directions are suppressed, while low-curvature directions are enabled, corroborating the observations of subspace heterogeneity.

## Highlights & Insights
- The reframing that "layers are not atomic units of capability, SVD subspaces are" is very sharp. Once accepted, existing "single $\lambda$" merging methods become suboptimal special cases, making FRISM a general solution.
- Using a minimalist "frozen basis, learned spectrum" gating structure reduces training costs to nearly zero while remaining universal and easy to plug into any VLM, which is highly beneficial for smaller research teams.
- The combination of "vision preservation + spectral maximization" acts as an implicit subspace filter that distinguishes "vision-neutral" from "vision-destructive" directions without reasoning labels. This can be extended to other capabilities like safety alignment or coding.
- The analytical derivation of $\partial\mathcal{L}/\partial\lambda_i$ provides mechanism-level interpretability, which is more persuasive than empirical comparisons alone.

## Limitations & Future Work
- Vision preservation relies on "KL on VizWiz." The degree of protection for other tasks (grounding, OCR, video) depends on calibration data distribution; narrow calibration data might cause other vision capabilities to drift.
- "Approximate decoupling of SVD subspaces in vision loss" is a key analytical assumption, but the Hessian may not be perfectly diagonal in practice.
- SVD is currently performed independently per linear layer without cross-layer coordination. Layer-wise subspaces might have synergies or conflicts; future work could explore "layer-subspace joint sparse" merging.
- Only the LLM Reasoning $\to$ VLM direction was evaluated; the reverse (VLM $\to$ LRM) or multi-way merging has yet to be verified.

## Related Work & Insights
- **Vs. Task Arithmetic / Ties-Merging / DARE**: Traditional multi-task merging emphasizes reducing interference but uses layer-level scalar coefficients, failing to resolve "intra-layer capability aliasing." FRISM pushes the trade-off curve outward using gate vectors.
- **Vs. IP-Merging / FRANK**: Both still operate at the "layer" granularity for selective merging. FRISM extends this logic by reducing granularity to the subspace and adding learning capabilities.
- **Vs. LoRA / PiSSA**: PEFT learns low-rank deltas on top of a base. FRISM does the inverse: it decomposes an existing delta into SVD subspaces and decides which ones to include, similar to "subspace pruning" + "subspace weighting."
- **Vs. SVDiff / SVD Distillation**: While model compression often keeps singular values fixed and changes the basis, FRISM keeps the basis fixed and changes singular values. This "frozen basis, learned spectrum" approach can be applied to compression, alignment, and safety tasks.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Refining model merging to the SVD subspace level with an unlabeled self-distillation framework is both novel and self-consistent.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers 3B/7B/32B scales, multiple benchmarks, and various baselines, supplemented by subspace-level ablations.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation (Figs 2-3) is very clear, with theoretical analysis and experiments supporting each other.
- **Value**: ⭐⭐⭐⭐⭐ Provides a low-cost, plug-and-play, highly interpretable framework for capability injection, significantly advancing the reasoning-vision fusion field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Model Merging Scaling Laws in Large Language Models](model_merging_scaling_laws_in_large_language_models.md)
- [\[ICML 2026\] Geo-Expert: Fine-tuning 8B Models into Expert-Level Geological Reasoning LLMs using LoRA](geo-expert_towards_expert-level_geological_reasoning_via_parameter-efficient_fin.md)
- [\[ICML 2026\] Saliency-Aware Model Merging](saliency-aware_model_merging.md)
- [\[ICML 2026\] Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training](decouple_searching_from_training_scaling_data_mixing_via_model_merging_for_large.md)
- [\[ICML 2026\] When Shared Knowledge Hurts: Spectral Over-Accumulation in Model Merging](when_shared_knowledge_hurts_spectral_over-accumulation_in_model_merging.md)

</div>

<!-- RELATED:END -->
