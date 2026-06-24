---
title: >-
  [Paper Note] H2OFlow: Grounding Human-Object Affordances with 3D Generative Models and Dense Diffused Flows
description: >-
  [ICLR 2026][3D Vision][3D Affordance] H2OFlow utilizes 3D generative models to create synthetic HOI data and models point-wise displacement distributions from human to target poses via "dense diffused flow" on point clouds. **With zero manual annotation**, it simultaneously learns contact, orientation, and spatial occupancy affordances, generalizing effectively to real-world point clouds.
tags:
  - "ICLR 2026"
  - "3D Vision"
  - "3D Affordance"
  - "Human-Object Interaction"
  - "Dense Diffused Flow"
  - "Diffusion Transformer"
  - "Point Cloud"
  - "Synthetic Data"
date: 2026-05-08
content_hash: e75f889ca52acf7f
---

# H2OFlow: Grounding Human-Object Affordances with 3D Generative Models and Dense Diffused Flows

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=QhqJ1DCp1X](https://openreview.net/forum?id=QhqJ1DCp1X)  
**Code**: To be confirmed  
**Area**: 3D Vision / Human-Object Interaction (HOI) / 3D Affordance  
**Keywords**: 3D Affordance, Human-Object Interaction, Dense Diffused Flow, Diffusion Transformer, Point Cloud, Synthetic Data  

## TL;DR
H2OFlow utilizes 3D generative models to create synthetic HOI data and models point-wise displacement distributions from human to target poses via "dense diffused flow" on point clouds. **With zero manual annotation**, it simultaneously learns contact, orientation, and spatial occupancy affordances, generalizing effectively to real-world point clouds.

## Background & Motivation
- **Background**: Human-object affordance is a core capability for robotics, vision, and embodied intelligence—enabling agents to understand "how an object can be used." However, most existing 3D HOI affordance methods only learn **contact**, relying on dense supervision from RGB images, point clouds, or human model contact regions.
- **Limitations of Prior Work**: First, contact annotation is **extremely costly and difficult to generalize** to new objects. Second, focusing solely on contact is too narrow—human-object interactions also involve **orientation** (e.g., preference for facing a TV or a specific wrist angle when holding a hammer) and **spatial occupancy** (e.g., standing in front of a microwave rather than behind it), which are non-contact geometric patterns.
- **Key Challenge**: COMA (Kim et al. 2024) recently proposed "comprehensive affordance" to probabilistically model contact and non-contact relations. However, it relies on **2D→3D lifting** from synthetic RGB images, which requires complex masking, is prone to failure, and necessitates **watertight mesh** algorithms. Consequently, it is nearly unusable for noisy real-world point clouds and lacks generalization.
- **Goal**: Simultaneous learning of contact, orientation, and spatial affordances under the premise of **zero manual annotation** and **point-cloud-only input** (without watertight meshes), ensuring generalization to incomplete point clouds from real-world captures.
- **Core Idea**: **[Generative Data Synthesis + Flow as Intermediate Representation]** A pretrained 3D generative model directly generates 3D HOI samples (bypassing error-prone 2D→3D lifting). **Dense diffused flow**—learning a point-wise displacement distribution from T-pose humans to target interaction poses on point clouds—serves as a generalizable intermediate representation. Finally, three types of affordances are **analytically** derived from the sampled flows and point clouds.

## Method

### Overall Architecture
H2OFlow consists of three stages: ① Using the pretrained 3D generative model CHOIS to generate diverse HOI mesh sequences from text as training data; ② Sampling point clouds from human/object meshes and training a DiT diffusion model to learn the distribution $p_\theta(F\mid O)$ of point-wise displacement flow $F$ conditioned on the object point cloud; ③ During inference, multiple flows are sampled for unseen object point clouds to reconstruct human configurations, followed by an **analytical** derivation of contact, orientation, and spatial affordance scores from the human-object point pair distributions. The key lies in encoding "how humans move" into a dense flow distribution, modeling interaction multimodality while eliminating dependence on watertight meshes.

```mermaid
flowchart LR
    A[Text Prompt] --> B[Pre-trained 3D Generative Model CHOIS<br/>Generate HOI Mesh Sequences]
    B --> C[Sample Point Clouds<br/>Human H / Object O]
    C --> D[Train DiT Diffusion Model<br/>Learn Dense Diffused Flow p_θ&#40;F|O&#41;]
    E[Unseen Object Point Cloud O] --> F[Sample Multiple Flows F~p_θ&#40;F|O&#41;<br/>H=H0+F Reconstruct Target Human]
    D --> F
    F --> G[Point-Pair Distribution P_ij]
    G --> H1[Contact Affordance C]
    G --> H2[Orientation Affordance R]
    G --> H3[Spatial Affordance S]
```

### Key Designs

**1. Dense Diffused Flow Representation: Encoding Human Interaction as Point-wise Displacement** The core is to predict the **flow field** $F=\{f_i\}$ rather than direct human poses. Given a fixed canonical T-pose (0-pose) SMPL human point cloud $H_0$, $N_H$ points are sampled from target interaction meshes to ensure one-to-one correspondence. Flow is defined as point-wise displacement $f_i := h_i - h_{0,i}$, i.e., $F := H - H_0$, with reconstruction via $H = H_0 + F$. This "flow" representation applies to both rigid and deformable bodies and naturally resides on point clouds, which is why H2OFlow generalizes to unseen objects—the model learns how **local geometric cues drive human point displacement** rather than memorizing global mesh templates.

**2. Modeling Multimodal Flow Distribution with DiT** Since HOI is highly multimodal (e.g., left/right hand or different regions contactable), a deterministic prediction is insufficient. The authors learn a **conditional distribution** $p_\theta(F\mid O)$. A standard diffusion process is used: forward noise addition $F_t = \sqrt{\bar\alpha_t}F_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$ and reverse denoising $p_\theta(F_{t-1}\mid F_t)$, supervised by the hybrid loss from Nichol & Dhariwal (noise loss + $\Sigma_\theta$-based cumulative KL loss). The backbone is a **Diffusion Transformer (DiT)**: point-wise features are extracted for noisy flow $F_t$, human $H$, and object $O$ using shared-weight MLPs. Flow and human features are concatenated as $f_{FH}$ for input, conditioned on object features $f_O$. Each DiT block performs **self-attention** on $f_{FH}$ (internal human flow coordination) followed by **cross-attention** between $f_{FH}$ and $f_O$ (capturing global human-object interaction). The cross-attention weights $w_{ij}$ are later reused for affordance scoring.

**3. Analytical Derivation of Three Affordances** During inference, flows are sampled for unseen objects to obtain a set of target humans. Defining the conditional distribution of human point $h_i$ relative to object point $o_j$ as $P_{ij}:=p(h_i\mid o_j)$, the affordances are expectations over this distribution:
- **Contact**: Weighted by human-object point distance and cross-attention, $c_{ij} = \mathbb{E}_{h_i\sim P_{ij}}\!\left[w_{ij}\cdot \exp(-\|d_{ij}\|)/\tau\right]$, where $d_{ij}=h_i-o_j$; closer points yield higher scores.
- **Orientation**: Unlike COMA, which calculates noise-sensitive surface normals, this method uses **flow vectors as direction proxies**. Taking the cross product of displacement and flow $x_{ij} = (d_{ij}\times f_i)/\|d_{ij}\times f_i\|$, the unit sphere is discretized into $n_b$ bins for probability estimation $p_{x,ij}(n)$ via Gaussian kernels. **Negative Shannon entropy** $H_{ij}=\mathbb{E}[\log p_{x,ij}(n)]$ measures orientation consistency. Finally, $R_{ij}=\mathbb{E}_{h_i\sim P_{ij}}[w_{ij}\cdot H_{ij}/\tau]$—low entropy (concentrated orientation, e.g., hands) yields high affordance, while high entropy (scattered, e.g., feet) yields low.
- **Spatial**: A voxel grid $G\in\mathbb{R}^{H\times W\times L}$ is established around the object. An indicator function $\delta_{ij}$ marks whether a voxel is occupied by $h_i$. $S_{ij}=\mathbb{E}_{h_i\sim P_{ij}}[\delta_{ij}]$ represents the expected occupancy, resulting in a probability map of human occupancy around the object. All three are computed only on sampled points, allowing for GPU parallelism without high-quality meshes.

## Key Experimental Results

### Main Results (OMOMO Dataset, Comparison with COMA etc.)

| Method | SIM-H↑ | SIM-O↑ | MAE-H↓ | MAE-O↓ | Precision@K↑ | MSE↓ |
|------|--------|--------|--------|--------|--------------|------|
| COMA | 41.3% | 56.9% | 0.22 | 0.14 | 42.9% | 0.14 |
| COMA-Recon (Point cloud to mesh input) | 20.7% | 31.8% | 0.62 | 0.51 | 9.1% | 0.66 |
| H2OSMPL (Direct SMPL param prediction) | 57.3% | 68.0% | 0.21 | 0.15 | 53.6% | 0.14 |
| H2Flow-NoAttn (No cross-attention) | 67.3% | 76.4% | 0.15 | 0.09 | 69.2% | 0.12 |
| **H2OFlow** | **72.3%** | **81.0%** | **0.11** | **0.07** | **75.6%** | **0.12** |

H2OFlow significantly outperforms all baselines. Notably, **COMA's performance collapses when inputs are meshes reconstructed from point clouds** (COMA-Recon), confirming its dependence on clean watertight meshes and incompatibility with real-world point clouds.

### Ablation Study (Downstream HOI Inference Task)

| Variant | mAP@1↑ | mAP@5↑ | mAP@10↑ | Top-5 Acc↑ | Collision↓ | Leakage↓ |
|------|--------|--------|---------|------------|-----------|----------|
| C (Contact only) | 0.52 | 0.60 | 0.66 | 0.41 | 0.31 | 0.27 |
| C+O (+Orientation) | 0.57 | 0.63 | 0.69 | 0.55 | 0.26 | 0.22 |
| C+S (+Spatial) | 0.56 | 0.62 | 0.68 | 0.51 | 0.21 | 0.20 |
| **C+O+S (Full)** | **0.63** | **0.69** | **0.76** | **0.64** | **0.18** | **0.17** |
| Shuffled (R/S control) | 0.54 | 0.61 | 0.66 | 0.44 | 0.29 | 0.26 |

Orientation and spatial affordances provide distinct gains, with the combination yielding the best results. The **Shuffled control (retaining contact but shuffling orientation/spatial)** negates the gains, proving that improvements originate from structured orientation/spatial information rather than increased feature capacity.

### Key Findings
- **Learning Flow > Learning SMPL Parameters**: The H2OSMPL variant is markedly weaker than H2OFlow, indicating that point-wise flow representations are superior for generalization.
- **Cross-attention as an Error-Correction Signal**: Removing attention results in poorer symmetry for contact/orientation affordances. Under sparse sampling, attention weights utilize learned human-object geometric correlations to compensate, producing more symmetrical and reasonable affordance maps.
- **Robustness to Real Point Clouds**: Even on real point clouds captured via iPhone depth cameras with RealityKit and FPS downsampling, the model outputs multimodal and semantically sound affordances (e.g., different bag components, orientation around the head). This is attributed to random perturbations and occlusions applied to object point clouds during training. COMA degrades to oversimplified unimodal maps in these scenarios.
- **Improved Efficiency**: Since operations occur on sparse point clouds, H2OFlow significantly excels in memory and runtime compared to COMA.

## Highlights & Insights
- **"Generating Data" instead of "Labeling Data"**: Using text-driven 3D generative models to synthesize HOI bypasses expensive and non-generalizable manual contact annotations, while avoiding COMA's 2D→3D lifting failure modes.
- **Flow as a Unified Intermediate Representation**: By unifying "how humans interact" into point-wise displacement distributions, all three affordances are **analytically** derived from the same flow distribution, ensuring framework simplicity and consistency.
- **Using Flow Vectors Instead of Normals for Orientation**: The cross product $d_{ij}\times f_i$ couples "relative displacement direction" with "motion direction" into a stable directional vector, avoiding the instability and cost of calculating normals on noisy meshes.
- **Entropy as a Consistency Metric**: The use of entropy is intuitive—hands (consistent orientation) yield low entropy and high scores, while feet (disordered) yield high entropy and low scores.

## Limitations & Future Work
- **Dependence on Pretrained Model Quality**: Training data is purely generated by CHOIS; the coverage, object categories, and interaction diversity of the generative model set the upper bound for affordance learning. Rare objects/interactions may be limited.
- **Sensitivity to 0-pose Placement**: Human 0-pose placement relative to the object requires prior settings (Appendix B); inaccuracies here may affect flow semantics.
- **Affordance Fusion**: Downstream fusion uses normalized linear weighting $\phi_{ij}=\lambda_c\hat c_{ij}+\lambda_o\hat R_{ij}+\lambda_s\hat S_{ij}$, which requires manual tuning and lacks an adaptive mechanism.
- Evaluations are primarily on indoor household objects (OMOMO/BEHAVE); larger-scale scenes, multi-person, or dynamic interactions remain unverified.

## Related Work & Insights
- **COMA (Kim et al. 2024)**: The most direct baseline. It first proposed comprehensive affordance but relies on 2D→3D lifting, watertight meshes, and surface normals. H2OFlow replaces both the representation (flow vs. mesh normals) and the data source (generation vs. lifting).
- **3D Flow (Eisner et al. 2022 / Xu et al. 2024 / Cai et al. 2024)**: Flow has proven effective in policy learning and object understanding. This work **transfers flow to the human body** and models its multimodality via diffusion.
- **3D HOI Generation (CHOIS / Diller & Dai / Peng et al.)**: Reusing CHOIS as a data engine exemplifies the paradigm of "using generative models as synthetic data sources," which is valuable for tasks where 3D interaction labels are scarce.
- Insight: When annotations are expensive and existing representations (like watertight meshes) limit deployment, the paradigm of "Synthetic Generation + Learning Generalizable Intermediate Representation (Flow) + Analytical Derivation" is highly reproducible.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Dense diffused flow as an intermediate HOI affordance representation and using cross-product entropy instead of normals is a novel combination that solves COMA's real-world generalization issues.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Includes quantitative results on OMOMO/BEHAVE, shuffled ablation, real iPhone point clouds, and memory/speed comparisons. The evidence chain is complete, though multi-person/large scenes are missing.
- **Writing Quality**: ⭐⭐⭐⭐ Clear motivation, well-defined formulas for affordances, and rich illustrations. Some details are appropriately deferred to the appendix.
- **Value**: ⭐⭐⭐⭐ Zero-annotation learning of multiple 3D affordances compatible with real point clouds holds high value for robotics and embodied interaction deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] QueryMe: Query-Driven Open-Vocabulary 3D Object Affordances Grounding from Multimodal Evidence](../../CVPR2026/3d_vision/queryme_query-driven_open-vocabulary_3d_object_affordances_grounding_from_multim.md)
- [\[ICLR 2026\] Generative Human Geometry Distribution](generative_human_geometry_distribution.md)
- [\[CVPR 2026\] Affostruction: 3D Affordance Grounding with Generative Reconstruction](../../CVPR2026/3d_vision/affostruction_3d_affordance_grounding_with_generative_reconstruction.md)
- [\[ICLR 2026\] SpatialHand: Generative Object Manipulation from 3D Perspective](spatialhand_generative_object_manipulation_from_3d_prespective.md)
- [\[CVPR 2026\] MonoVLM: Monocular 3D Visual Grounding with Vision Language Models](../../CVPR2026/3d_vision/monovlm_monocular_3d_visual_grounding_with_vision_language_models.md)

</div>

<!-- RELATED:END -->
