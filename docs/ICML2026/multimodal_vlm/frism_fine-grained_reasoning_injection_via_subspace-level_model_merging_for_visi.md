---
title: >-
  [Paper Note] FRISM: Fine-Grained Reasoning Injection via Subspace-Level Model Merging for Vision–Language Models
description: >-
  [ICML 2026][Multimodal VLM][Model Merging] FRISM refines "VLM × LRM merging" from the layer level to the SVD subspace level: it uses the SVD subspaces of LRM task vectors as reasoning priors…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "Model Merging"
  - "SVD Subspace"
  - "Reasoning Injection"
  - "Vision Preservation"
  - "Unlabeled Self-Distillation"
date: 2026-05-08
content_hash: 7e783cfb38e334cd
---

# FRISM: Fine-Grained Reasoning Injection via Subspace-Level Model Merging for Vision–Language Models

**Conference**: ICML 2026  
**arXiv**: [2601.21187](https://arxiv.org/abs/2601.21187)  
**Code**: None  
**Area**: Multimodal VLM / Model Merging / Reasoning Injection  
**Keywords**: Model Merging, SVD Subspace, Reasoning Injection, Vision Preservation, Unlabeled Self-Distillation

## TL;DR
FRISM refines "VLM × LRM merging" from the layer level to the SVD subspace level: it uses the SVD subspaces of LRM task vectors as reasoning priors, then employs an unlabeled self-distillation (with learnable gating only, KL for vision preservation + spectral norm maximization for reasoning absorption) to find the optimal injection strength, thereby significantly improving VL reasoning performance without notable vision degradation.

## Background & Motivation
**Background**: VLMs (Qwen2.5-VL, LLaVA, InternVL, etc.) have strong general capabilities but clear reasoning weaknesses; LRMs (DeepSeek-R1, OpenAI-o1) excel at math/logic/programming tasks. There are two main approaches to transfer LRM reasoning to VLMs: (1) large-scale retraining via RL/SFT; (2) model merging. The latter has near-zero training cost and requires no labeled data, making it widely adopted (e.g., BR2V, FRANK, IP-Merging).

**Limitations of Prior Work**: Existing merging methods mostly operate at the "layer" level—each layer is merged using a single mixing coefficient like $\lambda_{\text{vlm}}\tau_{\text{vlm}}+\lambda_{\text{lrm}}\tau_{\text{lrm}}$. Experiments in Figure 2 show that whether using Task Arithmetic or IP-Merging, tuning a single coefficient always leads to a clear vision–reasoning trade-off: either vision drops or reasoning is weak.

**Key Challenge**: By performing SVD on DeepSeek-R1-Distill-Qwen-7B task vectors and injecting them rank by rank into Qwen2.5-VL, the authors find that "the optimal scaling coefficients for different rank subspaces vary greatly" (Figure 3): some subspaces peak at $\lambda=0.1$, others require higher values; a single layer-wise $\lambda$ inevitably entangles this heterogeneity, introducing both useful reasoning and harmful vision noise. In other words, **layers are not atomic units of capability—subspaces are**.

**Goal**: Refine merging granularity to the SVD subspace level, allowing the model to automatically determine which subspaces should be strongly injected and which should be suppressed, all without relying on any VL reasoning labels.

**Key Insight**: Directly use the SVD decomposition of LRM task vectors as "reasoning prior subspaces," freeze $\mathbf{U},\mathbf{S},\mathbf{V}$, and only learn a per-rank gating vector $\mathbf{g}^l$; then, through "unlabeled self-distillation + spectral norm maximization," let the gating automatically find the balance between "maximum injection + minimal vision loss."

**Core Idea**: At each layer, open gates for each SVD subspace—"dual objectives + subspace gating" automatically filter out subspaces that harm vision, retaining reasoning subspaces orthogonal to vision.

## Method

### Overall Architecture
FRISM consists of two steps. **Stage 1 (Offline Decomposition & Initialization)**: Define LRM and VLM task vectors as $\tau_{\text{vlm}}=\theta_{\text{vlm}}-\theta_{\text{base}}$, $\tau_{\text{lrm}}=\theta_{\text{lrm}}-\theta_{\text{base}}$; perform SVD on each linear layer's $\tau_{\text{lrm}}^l$ to obtain $\mathbf{U}^{(l)},\mathbf{S}^{(l)},\mathbf{V}^{(l)\top}$, which are frozen; introduce a zero-initialized learnable gating $\mathbf{g}^l\in\mathbb{R}^r$ for each layer. **Stage 2 (Online Injection & Training)**: The merged model is $\theta_{\text{merged}}^l=\theta_{\text{vlm}}^l+\lambda_{\text{lrm}}\cdot\mathbf{U}^{(l)}(\sigma(\mathbf{g}^l)\odot\mathbf{S}^{(l)})\mathbf{V}^{(l)\top}$, which undergoes unlabeled self-distillation: using the original VLM as teacher, the student is trained on pure vision perception data (e.g., VizWiz VQA) to match the teacher's output (KL divergence); an additional loss maximizes the spectral norm of the injected subspace. Only the gating $\mathbf{g}^l$ is updated, making the process lightweight and fast to converge.

### Key Designs

1. **Subspace-Level Merging Paradigm (Eq. 7)**:

    - **Function**: Each of the $r$ subspaces within a layer has an independent scaling coefficient, fundamentally breaking the coupling of "one $\lambda$ per layer."
    - **Mechanism**: After SVD on $\tau_{\text{lrm}}^l$, freeze $\mathbf{U},\mathbf{S},\mathbf{V}$, and only learn $\mathbf{g}^l\in\mathbb{R}^r$; after Sigmoid, $\sigma(\mathbf{g}^l)\in(0,1)^r$ is elementwise multiplied with the original singular values $\mathbf{S}$ to form the "effective singular values" $\mathbf{S}_{\text{eff}}=\sigma(\mathbf{g}^l)\odot\mathbf{S}$. The merged weights are $\theta_{\text{merged}}^l=\theta_{\text{vlm}}^l+\lambda_{\text{lrm}}\,\mathbf{U}^{(l)}\mathbf{S}_{\text{eff}}^{(l)}\mathbf{V}^{(l)\top}$.
    - **Design Motivation**: The low-rank structure of LRM task vectors aligns with the empirical observation that "reasoning is concentrated in a few directions" (Cai 2025, Ping 2024, Sharma 2024); keeping the basis fixed and only adjusting the strength preserves the semantic direction of reasoning while allowing fine-grained control over the intensity per subspace—a classic "frozen basis, learn spectrum" approach.

2. **Unlabeled Self-Distillation: Vision Preservation Objective (Eq. 8)**:

    - **Function**: Without any VL reasoning labels, constrain the merged model to avoid output distribution drift on pure vision tasks, keeping vision degradation within acceptable bounds.
    - **Mechanism**: The teacher is the original VLM $\theta_{\text{vlm}}$ (frozen), the student is the current merged model $\theta_{\text{vlrm}}(\mathbf{g})$; on the calibration dataset $\mathcal{D}$ (VizWiz VQA in the paper), minimize KL: $\mathcal{L}_{\text{distill}}=\mathbb{E}_{x\sim\mathcal{D}}\mathrm{KL}\!\left(P(\cdot|x;\theta_{\text{vlm}})\,\|\,P(\cdot|x;\theta_{\text{vlrm}})\right)$.
    - **Design Motivation**: VL reasoning data is scarce and unevenly distributed, making direct supervision risky; using self-distillation with the original VLM as a reference turns "vision preservation" into a cheap, clean constraint, reframing merging as "finding the strongest reasoning injection within a KL radius."

3. **Spectral Norm Maximization for Reasoning Absorption + Overall Objective (Eq. 9–10)**:

    - **Function**: Prevent the gating from collapsing to the trivial "inject nothing" solution, actively encouraging $\mathbf{S}_{\text{eff}}$ to be as large as possible, i.e., to inject as much of the LRM subspace as possible into the VLM.
    - **Mechanism**: Define $\mathcal{L}_{\text{inject}}=-\sum_l\|\mathbf{S}_{\text{eff}}^{(l)}\|^2=-\sum_l\|\sigma(\mathbf{g}^{(l)})\odot\mathbf{S}^{(l)}\|^2$; stronger injection yields lower loss. The total loss is $\mathcal{L}=\mathcal{L}_{\text{distill}}+\alpha\mathcal{L}_{\text{inject}}$. The paper further provides a second-order expansion: under the Hessian $\mathbf{H}=\nabla^2\mathcal{L}_{\text{vis}}$ and the assumption of "approximate decoupling of different SVD subspaces," $\partial\mathcal{L}/\partial\lambda_i\approx(h_i-2\alpha\|B_i\|_F^2)\lambda_i$; thus, if a subspace's vision curvature term $h_i$ exceeds the injection benefit $2\alpha\|B_i\|_F^2$, the gating will suppress it; otherwise, it will be allowed.
    - **Design Motivation**: This combination is equivalent to "automatic filtering": subspaces orthogonal to vision perception (low $h_i$) are allowed, while high-curvature, vision-damaging subspaces are closed. The entire mechanism requires no reasoning supervision, achieving automatic trade-off resolution via data priors and spectral structure.

### Loss & Training
Only the gating $\mathbf{g}^l$ is trained, with negligible parameter count compared to the original model; the total loss is $\mathcal{L}=\mathcal{L}_{\text{distill}}+\alpha\mathcal{L}_{\text{inject}}$, where $\alpha$ controls injection strength. $\mathcal{L}_{\text{inject}}$ varies greatly across model scales, so normalization is performed before training (Appendix H). Only the LLM part of each VLM layer participates in merging; the vision tower and projection layers remain unchanged.

## Key Experimental Results

### Main Results: Multi-Benchmark Average Scores for Qwen2.5-VL × LRM Merging (Tab. 1)

| Method | VL Reasoning Avg. | VL Perception Avg. |
|--------|-------------------|-------------------|
| **3B Merging SmallThinker-3B** | | |
| Base | 33.2 | 79.7 |
| Task Arithmetic Best $\lambda$ | 33.0 | 79.8 |
| Ties-Merging | 31.6 | 77.0 |
| IP-Merging Best $T$ | 32.2 | 77.0 |
| **FRISM** | **35.0 (+1.8)** | 79.7 |
| **7B Merging DeepSeek-R1-Distill-Qwen-7B** | | |
| Base | 47.4 | 82.9 |
| Task Arithmetic Best $\lambda$ | 47.8 (high $\lambda$ collapsed) | 82.4 |
| Ties-Merging | 45.3 | 78.9 |
| IP-Merging Best $T$ | 47.7 | 82.3 |
| **FRISM** | **49.4 (+2.0)** | **83.0** |

### Subspace-Level Diagnosis (Figure 3)

| Experiment | Key Observation | Description |
|------------|----------------|-------------|
| Injecting individual rank subspaces | Different ranks peak at different $\lambda$ | Demonstrates subspace heterogeneity; "single $\lambda$ per layer" is necessarily suboptimal |
| Standard layer-wise merging | Significant gap from subspace-level optimum | Layer granularity cannot accommodate multiple optimal $\lambda$ values simultaneously |

### Vision–Reasoning Trade-off (Figure 2)
- In the "VL reasoning benchmark + VL perception benchmark" 2D space, Task Arithmetic / IP-Merging form a clear trade-off curve (either reasoning increases at the expense of vision, or vision is preserved but reasoning is weak).
- FRISM jumps directly to the upper right of the curve, showing that gating successfully filters out subspaces that "harm vision but contribute little to reasoning."

### Key Findings
- For 7B merging, Task Arithmetic shows a "cliff-like drop in vision" (POPE from 86.4 → 73.9) at $\lambda=0.15$, while FRISM achieves a 2pt average reasoning gain with almost no vision metric loss compared to Base—direct evidence of the advantage of subspace-level refinement.
- Removing $\mathcal{L}_{\text{inject}}$ causes the gating to collapse to negative infinity (no injection), proving the necessity of this "active amplification" term—it serves as an observable proxy for reasoning in the absence of reasoning labels.
- The second-order expansion $\partial\mathcal{L}/\partial\lambda_i\approx(h_i-2\alpha\|B_i\|_F^2)\lambda_i$ provides an interpretable filtering rule: high-vision-curvature directions are suppressed, low-curvature directions are allowed, corroborating the subspace heterogeneity observed in Figure 3.

## Highlights & Insights
- The reframing that "layers are not atomic units of capability, SVD subspaces are" is particularly sharp: once this assumption is accepted, all existing "single $\lambda$" merging methods become suboptimal special cases, and FRISM naturally emerges as the general solution in this space.
- The minimalist "frozen basis, learn spectrum" gating structure reduces training cost to nearly zero and is plug-and-play for any VLM, making it especially friendly for small and medium teams.
- The "vision preservation + spectral norm maximization" dual-objective acts as an implicit subspace filter, distinguishing "vision-irrelevant" from "vision-damaging" directions without reasoning labels; this mechanism can be extended to inject other capabilities (e.g., safety alignment, coding ability) as long as the corresponding capability can be represented as a task vector on the base model.
- The analytic derivation of $\partial\mathcal{L}/\partial\lambda_i$ in the paper provides mechanism-level interpretability, which is more convincing than mere experimental comparison.

## Limitations & Future Work
- Vision preservation relies on "KL on VizWiz," and the degree of protection for other vision tasks (grounding, OCR, video) depends on the calibration data distribution; if calibration data is narrow, other vision capabilities may drift unnoticed.
- The key theoretical assumption is "approximate decoupling of different SVD subspaces in vision loss," but in practice the Hessian may not be strictly diagonal; deviation from this assumption reduces the interpretability of the gating.
- Currently, SVD is performed independently for each linear layer, without cross-layer joint consideration; there may be synergy/conflict between layer-wise subspaces, suggesting future work on "layer-subspace joint sparsity" merging forms.
- Only LLM reasoning → VLM direction is evaluated; the reverse (VLM vision → LRM) or multi-way merging (multiple domain models into one base) remains untested.

## Related Work & Insights
- **vs Task Arithmetic / Ties-Merging / DARE**: Traditional multi-task merging emphasizes interference reduction but uses single-layer scalar coefficients, unable to address "intra-layer capability entanglement"; FRISM's gating vector pushes the trade-off curve outward.
- **vs IP-Merging (layer-level similarity threshold) / FRANK (Taylor closed-form layer weights)**: Both still operate at the "layer" granularity for selective merging; FRISM lowers the granularity to subspace and adds learning capability, a natural extension of this line of thought.
- **vs LoRA / PiSSA and other SVD-based PEFT**: PEFT learns a set of low-rank deltas on top of the basis; FRISM does the opposite—decomposing the trained delta into SVD subspaces and deciding which subspaces to inject, akin to "subspace pruning" + "subspace weighting."
- **vs SVDiff / SVD distillation in model compression**: FRISM is the opposite of "fix singular values, change basis"; it "fixes the basis, changes singular values." This "frozen basis, learn spectrum" approach can be borrowed for compression, alignment, safety, and other tasks.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Refines model merging from layer to SVD subspace granularity and provides an unlabeled self-distillation framework; innovative and self-consistent.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 3B/7B/32B scales, multiple benchmarks, and baselines, with subspace-level ablation.
- Writing Quality: ⭐⭐⭐⭐ Motivation and derivation (Figures 2–3) are very clear, with theory and experiments mutually supporting; some derivations rely on the appendix.
- Value: ⭐⭐⭐⭐⭐ Offers a low-cost, plug-and-play, highly interpretable capability injection framework, significantly advancing reasoning-vision integration.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] FREE-Merging: Fourier Transform for Efficient Model Merging](../../ICCV2025/multimodal_vlm/free-merging_fourier_transform_for_efficient_model_merging.md)
- [\[CVPR 2026\] HandVQA: Diagnosing and Improving Fine-Grained Spatial Reasoning about Hands in Vision-Language Models](../../CVPR2026/multimodal_vlm/handvqa_diagnosing_and_improving_fine-grained_spatial_reasoning_about_hands_in_v.md)
- [\[CVPR 2026\] DC-Merge: Improving Model Merging with Directional Consistency](../../CVPR2026/multimodal_vlm/dc-merge_improving_model_merging_with_directional_consistency.md)
- [\[ICML 2026\] Vision-aligned Latent Reasoning for Multi-modal Large Language Model](vision-aligned_latent_reasoning_for_multi-modal_large_language_model.md)
- [\[NeurIPS 2025\] Unveiling Chain of Step Reasoning for Vision-Language Models with Fine-grained Rewards](../../NeurIPS2025/multimodal_vlm/unveiling_chain_of_step_reasoning_for_visionlanguage_models.md)

</div>

<!-- RELATED:END -->
