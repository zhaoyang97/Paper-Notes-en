---
title: >-
  [Paper Note] OmniAID: Decoupling Semantic and Artifacts for Universal AI-Generated Image Detection in the Wild
description: >-
  [ICML 2026][Image Generation][AIGI Detection] OmniAID proposes a decoupled MoE architecture consisting of "Semantic Experts + a Universal Artifact Expert." It learns two types of forgery clues—"content-related flaws" and…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "AIGI Detection"
  - "MoE"
  - "Semantic-Artifact Decoupling"
  - "SVD Residual Subspace"
  - "Mirage Dataset"
date: 2026-05-08
content_hash: 6f4fd909086c678c
---

# OmniAID: Decoupling Semantic and Artifacts for Universal AI-Generated Image Detection in the Wild

**Conference**: ICML 2026  
**arXiv**: [2511.08423](https://arxiv.org/abs/2511.08423)  
**Code**: https://github.com/yunncheng/OmniAID  
**Area**: AI Safety / AIGC Detection / Multi-Expert Models  
**Keywords**: AIGI Detection, MoE, Semantic-Artifact Decoupling, SVD Residual Subspace, Mirage Dataset

## TL;DR
OmniAID proposes a decoupled MoE architecture consisting of "Semantic Experts + a Universal Artifact Expert." It learns two types of forgery clues—"content-related flaws" and "content-independent artifacts"—within the low-rank residual subspaces of CLIP-ViT attention weights derived via SVD. Combined with a modernized dataset, Mirage, it achieves average detection accuracies of 95.9%, 91.4%, and 88.4% across GenImage, Chameleon, and Mirage-Test benchmarks.

## Background & Motivation
**Background**: Current AI-generated image (AIGI) detection follows two paths. The artifact-specific approach uses handcrafted features like frequency domain filtering or upsampling fingerprints to capture generator-specific low-level traces. The VFM-based approach, which currently dominates, leverages the strong semantic priors of Vision Foundation Models (e.g., CLIP, DINOv2) through LoRA, SVD residual fine-tuning (Effort), or category prototype injection (C2P-CLIP) to achieve cross-generator generalization.

**Limitations of Prior Work**: The authors identify two primary bottlenecks. First, **Single Entangled Representation**: State-of-the-art (SOTA) methods compress "content-related semantic flaws" (e.g., malformed faces, non-physical structures) and "content-independent universal artifacts" (e.g., frequency fingerprints, VAE reconstruction traces) into the same feature space. Consequently, detectors trained on one domain (e.g., Animals) suffer significant performance drops when tested on others (e.g., Scenes), as verified in Fig 2a/b. Second, **Outdated Benchmarks**: Datasets like GenImage primarily stem from GANs and early Stable Diffusion versions; models trained on these fail nearly completely on modern in-the-wild sets like Chameleon (Fake accuracy drops to near 0).

**Key Challenge**: The pre-training objective of VFMs is general semantic understanding, not forgery detection. Simultaneously, mixed-data training couples semantic flaws and low-level artifacts, forcing the model to compromise between over-fitting training-domain semantic patterns or generator-specific artifacts during cross-domain evaluation.

**Goal**: To **explicitly decouple** two types of evidence: (i) flaws between different semantic domains should be independent, and (ii) content-dependent semantic flaws should be independent of content-independent universal artifacts. The authors also aim to provide a new benchmark reflecting the generator distribution of 2025.

**Key Insight**: The authors observe that "content-independent artifacts" exist in all images and should be handled by an **always-active** expert. Conversely, "content-dependent flaws" are highly domain-dependent and should be learned by **routable domain experts**. Furthermore, training experts in low-rank residual subspaces derived by SVD maintains CLIP's principal component priors while allowing different experts to be assigned to different directions via orthogonality constraints, naturally supporting decoupling.

**Core Idea**: Utilize a "Routable Semantic Experts + Fixed Universal Artifact Expert" hybrid MoE to separate "what is generated" from "how it is generated" into distinct subspaces.

## Method

### Overall Architecture
The input is an RGB image, and the backbone is a frozen CLIP-ViT-L/14@336px. For each attention layer weight $\mathbf{W}\in\mathbb{R}^{d_{out}\times d_{in}}$, SVD is performed: $\mathbf{W}=\mathbf{W}_M+\mathbf{W}_R$, where $\mathbf{W}_M=\mathbf{U}_{:d-r}\mathbf{\Sigma}_{:d-r}\mathbf{V}_{:d-r}^{T}$ is the frozen "main subspace" preserving CLIP's knowledge, and $\mathbf{W}_R=\mathbf{U}_{d-r:}\mathbf{\Sigma}_{d-r:}\mathbf{V}_{d-r:}^{T}$ is the "residual subspace" for experts. Two types of experts are placed in the residue: $N_S$ semantic experts $\mathcal{E}_S=\{e_1,\dots,e_{N_S}\}$ (e.g., Human, Animal, Object, Scene, Anime) and one universal artifact expert $\mathcal{E}_U$. Features from an independent frozen CLIP encoder are fed into a lightweight MLP router $\mathcal{R}$ to select the top-$k_S$ semantic experts. The final layer weight is $\mathbf{W}_F=\mathbf{W}_M+\mathbf{W}_{R,U}+\sum_{i\in S}g_i\cdot\mathbf{W}_{R,i}$, where $g_i=\mathrm{Softmax}(\mathbf{z}_\mathbf{x})_i$. Notably, the universal artifact expert $\mathbf{W}_{R,U}$ does not compete in routing and is **activated in every forward pass**. Training occurs in two stages: individual expert specialization followed by router training with frozen experts.

### Key Designs

1.  **Hybrid Orthogonal MoE in Residual Subspaces**:
    *   **Function**: Incorporate multiple types of forgery evidence into low-rank residuals without disrupting CLIP's principal priors.
    *   **Mechanism**: All experts occupy $\mathbf{W}_R$, spanned by the smallest singular components. Semantic experts are routable, and the universal artifact expert is fixed and persistent. Semantic experts use domain-specific data (e.g., all Human images) to learn domain-specific "malformations," while the universal artifact expert uses semantically aligned real/reconstructed pairs (COCO with multiple VAEs like SDv1.x–SD3.5, TAESD, TAESDXL) to learn "low-level traces left by any VAE."
    *   **Design Motivation**: To solve "single entangled representation." The fixed expert ensures artifact signals are always captured, while the routable experts prevent different semantic domains from being homogenized. Orthogonal residuals ensure that updates for experts do not collapse into each other or the main subspace.

2.  **Two-stage Decoupled Training + Cross-expert Orthogonal Constraints**:
    *   **Function**: Ensure each expert learns complementary, non-overlapping forgery evidence.
    *   **Mechanism**: In Stage 1, only one expert $e_a$ is activated at a time while others are frozen, with the objective $\mathcal{L}_{\text{Stage1}}=\mathcal{L}_{\text{cls}}+\lambda_1\mathcal{L}_{\text{orth}}$. The orthogonal loss $\mathcal{L}_{\text{orth}}=\sum_{j\in\mathcal{I}_{\text{prev}}}(\|\mathbf{U}_i^T\mathbf{U}_j\|_F^2+\|\mathbf{V}_i^T\mathbf{V}_j\|_F^2)$ constrains the new expert to be orthogonal to both the main subspace and **all previously trained experts**. The classification head is reset after each expert's training to avoid memory pollution. In Stage 2, all experts are frozen to train the router and a new classification head with $\mathcal{L}_{\text{Stage2}}=\mathcal{L}_{\text{cls}}+\lambda_2\mathcal{L}_{\text{gating}}+\lambda_3\mathcal{L}_{\text{balance}}$, where $\mathcal{L}_{\text{gating}}$ supervises the router with true domain labels $y_e$ for a sharp distribution, and $\mathcal{L}_{\text{balance}}$ encourages diversity using a Switch Transformer-style load balancing loss.
    *   **Design Motivation**: Compared to Effort, this method pushes the orthogonal boundary to "all previous semantic experts," providing a hard constraint for semantic decoupling. The two-stage separation prevents the router from collapsing to the fastest-learning expert during joint training.

3.  **Modernized Mirage Dataset + Anchored Synthesis Pipeline**:
    *   **Function**: Provide a 2025-scale training and testing foundation reflecting in-the-wild distributions.
    *   **Mechanism**: Mirage-Train contains 933K real and 1674K fake images across five categories; fakes are generated using SOTA T2I models like SD3.5, Flux.1, and commercial APIs. The synthesis pipeline utilizes "real-image-anchored prompting": every real image is captioned by an LMM, and this description is fed to T2I generators to **force semantic alignment** between real and fake pairs, preventing the model from relying on content differences as a shortcut. Mirage-Test includes 22K real and 28K fake images from held-out SOTA generators fine-tuned for realism.
    *   **Design Motivation**: Older benchmarks (GenImage uses 2022 models) fail to reflect 2025 generator capabilities. Anchored prompting provides the "semantically aligned pairs" necessary to force the model to learn true artifacts rather than semantic shortcuts.

### Loss & Training
- **Stage 1**: $\mathcal{L}_{\text{cls}}$ + $\lambda_1\mathcal{L}_{\text{orth}}$, training experts sequentially by unfreezing their specific $\mathbf{U}_{d-r:}, \mathbf{\Sigma}_{d-r:}, \mathbf{V}_{d-r:}$ and the classification head.
- **Stage 2**: $\mathcal{L}_{\text{cls}}$ + $\lambda_2\mathcal{L}_{\text{gating}}$ + $\lambda_3\mathcal{L}_{\text{balance}}$, with all experts frozen.
- **Config**: AdamW, $lr=2\times 10^{-4}$, batch size 32. 1 epoch per stage on 4× H200. GenImage-SDv1.4 training takes 3 hours; Mirage takes 18 hours.

## Key Experimental Results

### Main Results

| Dataset | Metric | OmniAID | OmniAID-Mirage | Prev. SOTA | Gain |
| :--- | :--- | :--- | :--- | :--- | :--- |
| GenImage (Avg. 8 subsets) | Acc % | 95.9 | 97.2 | Effort 91.1 | +4.8 / +6.1 |
| Chameleon (in-the-wild) | Acc % | ~77 | **91.4** | GenImage-baselines ~50% (collapse) | > 2x |
| Mirage-Test (Avg. 5 cat.) | Acc % | 51.1 | **88.4** | Effort 43.0, DRCT 42.0 | +45.4 |
| Mirage-Test (Avg. 5 cat.) | AP % | 53.4 | **96.8** | Effort 46.8 | +50.0 |

*Note: Standard OmniAID is trained only on GenImage-SDv1.4 for fair comparison; OmniAID-Mirage uses the new Mirage-Train. The BigGAN subset highlights the value of decoupling: OmniAID 98.7% vs Effort 77.6%.*

### Ablation Study

| Config ($e_0$=H/A, $e_1$=O/S, $e_U$=Artifact) | GenImage | Chameleon | Mirage-Test | Description |
| :--- | :--- | :--- | :--- | :--- |
| $e_0$ only | 84.4 | 58.9 | 39.6 | Single semantic expert |
| $e_1$ only | 85.2 | 59.0 | 36.3 | Single semantic expert |
| $e_U$ only | 83.3 | 60.9 | 45.1 | Single artifact expert (beats semantic on OOD) |
| $e_0+e_1$ (No Artifact Expert) | 92.2 | 66.1 | 44.5 | Without $e_U$, Chameleon drops 11.3 |
| $e_0+e_U$ | 91.9 | 68.1 | 47.4 | Without $e_1$ |
| $e_1+e_U$ | 93.5 | 70.8 | 49.0 | Without $e_0$ |
| **Full ($e_0+e_1+e_U$)** | **95.9** | **77.4** | **51.1** | Synergy of all three |

### Key Findings
- **Universal Artifact Expert is Crucial for Generalization**: Removing $e_U$ causes an 11.3% drop on Chameleon, far exceeding the drop from removing any single semantic expert ($\leq$ 6.5%). Semantic experts are prone to over-fitting in-domain flaws, while $e_U$ learns more transferable low-level traces.
- **Strong Subject Semantics Can Hinder Performance**: Removing the Object/Scene expert resulted in a larger performance drop than removing the Human/Animal expert. The authors speculate that salient subjects like humans lead models toward semantic over-fitting, contributing less to generalization than more diverse classes.
- **Router Interpretability**: Single Human images give a 0.94 weight to the Human expert. Mixed images (e.g., "Animal with Human") are automatically distributed (e.g., Animal 0.69 + Human 0.31). Unseen categories (e.g., medical images) are routed reasonably (Object 0.57, Human 0.37) while keeping the artifact expert active, achieving 92% Acc on 400 medical samples.
- **Data + Architecture Must Evolve Together**: Mirage-Train improves all methods, but only OmniAID avoids regression on old benchmarks, indicating that success on old benchmarks does not simply translate to in-the-wild scenarios.

## Highlights & Insights
- The design of **"content-independent artifacts = fixed, non-routable expert"** is clever. Since universal artifacts should appear in every image, allowing them to participate in top-k competition would risk them being displaced by semantic experts. Fixing their activation acknowledges the fundamental difference between the two types of evidence at a structural level.
- **Cross-expert Orthogonal Constraints** extend the "expert vs main subspace" orthogonality from Effort to an "expert vs all previous experts" constraint. This is a simple yet powerful engineering trick to prevent MoE experts from learning redundant features.
- **Anchored Prompting** uses data engineering to compensate for the model's lack of "semantic invariance." This is a useful paradigm for training any discriminative model that needs to exclude content shortcuts.
- While not explicitly an "open-set" method, the combination of a universal artifact expert and soft router allocation provides a natural mechanism for handling unseen categories.

## Limitations & Future Work
- The authors acknowledge limitations primarily regarding psychological impact (ethical); the main text lacks in-depth failure case analysis.
- Observed limitations: (i) $N_S$ semantic experts depend on a fixed taxonomy, requiring redesign for truly novel semantic classes (e.g., emerging art styles); (ii) Reliance on CLIP-ViT-L/14@336px without analyzing the impact of different VFM scales; (iii) The router uses an independent CLIP encoder, adding inference latency not detailed in the main text.
- Future Work: Implementing "soft-gating" for continuous expert activation; using contrastive learning to further separate expert features; and introducing frequency/DCT branches into the universal artifact expert.

## Related Work & Insights
- **vs Effort (Yan et al., 2025b)**: Both use SVD residual subspaces for PEFT, but Effort uses a single adapter. Ours extends "one low-rank adaptation" into a "low-rank MoE" by splitting the subspace into $N_S+1$ orthogonal experts.
- **vs DRCT / AlignedForensics**: These rely on semantically aligned real/reconstructed pairs to force artifact learning. Ours adopts this data construction but limits its use to the "universal artifact expert," while maintaining separate semantic experts.
- **vs C2P-CLIP (Tan et al., 2025)**: C2P reinforces category prototype matching. Ours instead splits "semantic evidence" by domain and adds a fallback artifact channel.
- **vs AIDE (Yan et al., 2025a)**: AIDE implements "semantic + DCT frequency" dual paths. Ours follows a similar dual-path logic but integrates both as MoE experts within a unified architecture.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Modeling two levels of decoupling ("artifact vs semantic" and "inter-semantic") into an MoE with a fixed artifact expert is a distinct architectural stance in AIGI detection.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive coverage across three benchmarks, two training sets, full ablation, data vs architecture analysis, router visualization, and unseen category probing.
- **Writing Quality**: ⭐⭐⭐⭐ Pipeline diagrams are clear and notation is consistent, though some theoretical analysis of orthogonality is relegated to the Appendix.
- **Value**: ⭐⭐⭐⭐⭐ Provides both a SOTA detector and a modernized benchmark for 2025-era generators, significantly pushing the field toward practical deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] DGS-Net: Distillation-Guided Gradient Surgery for CLIP Fine-Tuning in AI-Generated Image Detection](dgs-net_distillation-guided_gradient_surgery_for_clip_fine-tuning_in_ai-generate.md)
- [\[AAAI 2026\] Beyond Semantic Features: Pixel-Level Mapping for Generalized AI-Generated Image Detection](../../AAAI2026/image_generation/beyond_semantic_features_pixel-level_mapping_for_generalized_ai-generated_image_.md)
- [\[AAAI 2026\] Aggregating Diverse Cue Experts for AI-Generated Image Detection](../../AAAI2026/image_generation/aggregating_diverse_cue_experts_for_ai-generated_image_detec.md)
- [\[ICML 2026\] Semantic Granularity Navigation in Image Editing](semantic_granularity_navigation_in_image_editing.md)
- [\[NeurIPS 2025\] Physics-Driven Spatiotemporal Modeling for AI-Generated Video Detection](../../NeurIPS2025/image_generation/physics-driven_spatiotemporal_modeling_for_ai-generated_video_detection.md)

</div>

<!-- RELATED:END -->
