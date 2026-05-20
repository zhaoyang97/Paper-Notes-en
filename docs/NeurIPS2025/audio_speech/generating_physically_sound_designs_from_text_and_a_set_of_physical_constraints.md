---
title: >-
  [Paper Note] Generating Physically Sound Designs from Text and a Set of Physical Constraints
description: >-
  [NeurIPS 2025][Audio & Speech][topology optimization] This paper proposes TIDES, a framework that combines the visual guidance of pretrained text-image models (CLIP) with a differentiable finite-element physics simulator…
tags:
  - "NeurIPS 2025"
  - "Audio & Speech"
  - "topology optimization"
  - "CLIP"
  - "differentiable physics"
  - "text-informed design"
  - "structural optimization"
  - "3D printing"
date: 2026-05-08
content_hash: 1cea06edb2c98de4
---

# Generating Physically Sound Designs from Text and a Set of Physical Constraints

**Conference**: NeurIPS 2025
**arXiv**: [2602.02213](https://arxiv.org/abs/2602.02213)  
**Authors**: Gregory Barber, Todd C. Henry, Mulugeta A. Haile (DEVCOM Army Research Laboratory)
**Code**: None  
**Area**: Generative Design / Topology Optimization / Text-Guided Design
**Keywords**: topology optimization, CLIP, differentiable physics, text-informed design, structural optimization, 3D printing

## TL;DR

This paper proposes TIDES, a framework that combines the visual guidance of pretrained text-image models (CLIP) with a differentiable finite-element physics simulator. By jointly optimizing a visual similarity loss and a structural compliance loss, TIDES generates load-bearing structural designs that satisfy both engineering performance requirements and text-specified visual characteristics, starting from text descriptions and physical constraints. The method is validated through 3D-printed three-point bending experiments.

## Background & Motivation

Generative design has been advancing rapidly and independently in two domains: text-to-image generation (physics-agnostic) and engineering structural optimization (physics-constrained). Each domain exhibits significant limitations:

**Limitations of text-to-image generation**: Models such as DALL-E and Stable Diffusion can generate "an avocado-shaped armchair," but provide no guarantee that the chair can bear load, is made of real materials, or is manufacturable. These models lack any understanding of physical constraints such as structural deformation, material weight, or manufacturability.

**Limitations of structural optimization**: Topology optimization can design load-bearing columns, beams, and bridges, but it is extremely difficult to express complex design intent—for example, specifying the use of arch, hexagonal, or triangular support strategies, or requiring a design to possess certain aesthetic characteristics. Traditional methods rely on random initialization to produce design diversity and lack intelligent exploration of the design space.

The paper's core insight is that both text-to-image generation and structural optimization operate on pixel/element grids, and both performance metrics are differentiable. The two approaches can therefore share the same design space and be jointly optimized. TIDES is built on this natural coincidence, "grafting" physical constraints onto a pretrained text-to-image model without requiring the training of a new text-to-design model from scratch.

## Method

### Overall Architecture of TIDES

TIDES (Text Informed DESign) consists of three core components:

1. **Design Encoding**: Parameterizes the design space, determining whether material is placed at each location.
2. **Physics Simulator**: A differentiable finite-element solver that evaluates the structural performance of a design.
3. **Visual Judge**: A pretrained CLIP model that evaluates how well a design matches a text description visually.

The total loss function is:

$$\mathcal{L} = c + \beta_1 m - \beta_2 v$$

where $c$ is compliance (measuring structural deformation under load; lower values indicate greater stiffness), $m$ is material cost (MAE between target and current density), $v$ is the CLIP visual similarity score, and $\beta_1$ and $\beta_2$ are balancing weights.

### Design Encoding and the Hill Function

The design space is discretized into a finite-element mesh, with each element assigned a continuous density parameter $d \in [0,1]$. Elements are initialized to all ones (solid block), and a Gaussian blur filter is applied to avoid checkerboard artifacts.

One of the key technical innovations is the introduction of a **Hill-function-style sigmoid**:

$$d = \frac{1}{1 + \frac{\alpha}{x^n + 0.1}}$$

where $\alpha = 0.8$ and $n = 20$. This function pushes density values toward 0 or 1 (corresponding to the absence or presence of material), addressing the problem that CLIP tends to generate grayscale gradients (artistic shading) rather than binary distributions. Ablation studies show that removing the Hill function results in a large number of intermediate gray values, which are detrimental to both structural optimization and manufacturability.

### Modified SIMP Method

The SIMP (Solid Isotropic Material with Penalization) method is adopted, with the stiffness coefficient of each element defined as:

$$E_e(d_e) = E_{min} + d_e^p (E_0 - E_{min})$$

where $p=3$ is the penalization factor that drives densities toward 0 or 1. Structural performance is evaluated via compliance (elastic strain energy):

$$c(d) = U^T K U = \sum_{e=1}^{n} E_e(d_e) u_e^T k_0 u_e$$

where $KU = F$, $K$ is the global stiffness matrix, and $F$ is the applied force vector. Lower compliance indicates greater structural stiffness.

### Visual–Physical Domain Mapping

CLIP is trained on 224×224 three-channel RGB images, whereas structural optimization operates on single-channel density maps of arbitrary size. TIDES bridges this gap as follows:

- **Size adaptation**: Bilinear interpolation rescales the design space to 224×224.
- **Channel adaptation**: The single-channel density map is replicated three times to produce a grayscale image.
- **Data augmentation**: Following CLIPDraw's image augmentation scheme—random cropping, random perspective transforms, and resizing—a batch of images is generated and fed into the CLIP encoder. Sampling different regions stochastically enables TIDES to produce different designs across repeated runs even under identical prompts and problem settings.

Since CLIP's training data includes binary images such as outlines, silhouettes, and line drawings, appending descriptors such as "dark black outline" to text prompts helps guide CLIP toward binary designs.

### Compliance Masking to Eliminate Floating Material

The introduction of the visual loss can produce aesthetically appealing but structurally irrelevant "floating" features (e.g., cloud-like elements). TIDES addresses this with a **compliance-based masking method**:

- Per-element compliance is obtained from the physics simulator.
- Thresholding: $\text{mask} = \log(\text{compliance}) \geq -20$
- The mask is multiplied with the design encoding, ensuring CLIP only observes pixels that directly contribute to structural support.

Compared to the connected-component labeling (CCL) algorithm used in concurrent work, compliance masking incurs zero additional computational overhead (since compliance is already computed during structural optimization) and introduces no extra computational bottleneck as design resolution increases. Ablation studies confirm that masked designs are free of floating material, whereas removing the mask produces floating structures.

## Key Experimental Results

### Multi-Resolution Design Generation

On a tower structural optimization problem, resolution is progressively increased from 32×32 to 512×512:

- **"Avocado-shaped armchair"**: At 64×64, only a simple outline is captured; at 128×128, textured cutouts appear; at 512×512, the dimpled skin texture of an avocado is reproduced. The design consists of two parts: a rounded upper form and a supporting lower structure.
- **"Art Deco hotel"**: At 64×64, a rectangle with square windows appears; at 128×128, a triangular form resembling New York's Flatiron Building emerges; at 512×512, the tower features a detailed canopy at its top.

The compliance of all TIDES designs is several orders of magnitude lower than vision-only results and on the same order of magnitude as physics-only results, demonstrating that the designs maintain physical performance while incorporating text-specified features.

### Text-Driven Navigation of the Design Space

On suspension bridge and ring design problems, 30 trials are conducted for each of three prompts—"large triangle," "large hexagon," and "large arch":

- **Hexagon prompt**: Honeycomb-like hexagonal grid supports are generated.
- **Triangle prompt**: A large central triangle surrounded by smaller triangles is produced.
- **Arch prompt**: A large central arch with surrounding smaller arches appears.

Designs guided by different text prompts are competitive in physical performance, and in some cases even overlap with physics-only results. This suggests that the visual loss can help escape local optima in certain scenarios—adding visual constraints actually yields better structural performance in some cases.

### Robustness Across Diverse Design Problems

Across roof, staircase, cantilever beam, dam, and multi-story building configurations with various force/support setups and text prompts:

- Vision-only designs exhibit recognizable features but lack physical integrity.
- Physics-only designs resist deformation but employ simple support strategies.
- TIDES designs combine both: compliance on the same order of magnitude as physics-only results, together with complex text-specified support features.

### 3D-Printed Three-Point Bending Experiments

TIDES-generated beam designs (672×96 resolution, 7:1 inch aspect ratio) were fabricated using a Markforged X7 printer with Onyx carbon-fiber nylon material; three replicates per design were tested under three-point bending:

- **Physics-only design**: compliance 1.0e-3 N/mm (simulation 280.0 c)
- **"Honeycomb/multi-hexagon"**: compliance 1.13e-3 N/mm (simulation 299.55 c)
- **"Large arch"**: compliance 1.19e-3 N/mm (simulation 285.71 c)

All three designs performed within the same order of magnitude in both experiment and simulation, confirming that all designs successfully resisted deformation under the applied load and validating the physical realizability of TIDES-generated designs.

### Fine-Tuning from Existing Designs

TIDES also supports initialization from images generated by Stable Diffusion, which are converted to single-channel density maps and used as the initial design encoding before joint optimization improves physical performance:

- The Eiffel Tower design retains its iconic form while TIDES reinforces the support trusses.
- The St. Louis Arch design preserves the arch shape while adding horizontal support members.
- For more complex initial conditions such as the Space Needle and robotic forms, TIDES removes floating material, reconnects broken structures, and redistributes material.

## Highlights & Insights

- **No model retraining required**: The central finding is that imposing physical constraints on a pretrained text-to-image model suffices for generating structurally sound designs, without requiring a new model to be trained from scratch on physics data. This substantially lowers the barrier to implementation.
- **Text as a design space explorer**: Traditional topology optimization relies on random initialization to drive diversity; TIDES uses text prompts to purposefully navigate the design space, generating diverse functional designs with different support strategies (arch/hexagon/triangle) that are comparably competitive in performance.
- **Zero-overhead compliance masking**: Leveraging per-element compliance already computed by the physics simulator as a mask eliminates floating material at zero additional computational cost—a more efficient and physically meaningful approach than CCL methods.
- **Hill function bridging the binary/continuous domain gap**: A sigmoid-style transform elegantly resolves the contradiction between CLIP's preference for grayscale gradients and structural optimization's requirement for binary distributions, in a concise and effective manner.
- **Closed-loop experimental validation**: The complete validation chain from simulation to 3D printing to physical testing—rather than remaining at the simulation level—substantially strengthens the paper's credibility.

## Limitations & Future Work

- **Limited to 2D structures**: Current designs are confined to 2D planes; 3D designs are realized only via 2D extrusion. Future work should extend the framework to simulators supporting 3D meshes.
- **Limitations of CLIP's visual understanding**: CLIP's comprehension of binary images is limited; it relies on appended descriptor words (e.g., "dark black outline") for guidance, and the transmission of complex textual concepts remains unstable.
- **Hyperparameter tuning**: $\beta_1$ and $\beta_2$ require manual adjustment for different design problems to balance visual, physical, and material cost objectives.
- **Dependence on differentiable physics**: The framework has a strong dependency on differentiable simulators; although genetic algorithms are mentioned as an adaptation for non-differentiable problems, this is not experimentally validated.
- **Manufacturing precision constraints**: The minimum feature size of 3D printing (200-micron path width) limits the fidelity of fine features, and stress concentrations at fine structural elements may degrade real-world performance.
- **Manufacturability constraints not addressed**: Optimization objectives consider only stiffness and material cost; design-for-manufacturing constraints such as minimum feature size and overhang angle are not incorporated.

## Related Work & Insights

- **Classical topology optimization**: The SIMP method (Bendsøe & Kikuchi, 1988) is the dominant tool for structural optimization; this paper builds upon it by adding a visual loss to enable text-guided co-design.
- **CLIP-guided generation**: CLIPDraw (Frans et al., 2021), Deep Daze, and similar works use CLIP to generate artistic images without physical constraints; this paper is the first to combine CLIP guidance with differentiable physics.
- **Physics-constrained 3D generation**: Atlas3D (Chen et al., 2024) uses physical losses to improve the stability of 3D objects generated by diffusion models; DSO (Li et al., 2025) fine-tunes diffusion models. This paper addresses a different problem—the performance of load-bearing structures under multi-directional forces—and does not require a diffusion model as a starting point.
- **Concurrent work comparison**: Zhong et al. (2023) also investigate CLIP-guided topology optimization, but use RGB channels, which can cause overfitting (windows exist only in color space rather than as real structural features). TIDES's binary designs avoid this problem.

## Rating ⭐

| Dimension | Rating |
|-----------|--------|
| Novelty | ⭐⭐⭐⭐ |
| Theoretical Depth | ⭐⭐⭐ |
| Practicality | ⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Overall Recommendation | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Seeing Sound, Hearing Sight: Uncovering Modality Bias and Conflict of AI Models in Sound Localization](seeing_sound_hearing_sight_uncovering_modality_bias_and_conflict_of_ai_models_in.md)
- [\[NeurIPS 2025\] Sound Logical Explanations for Mean Aggregation Graph Neural Networks](sound_logical_explanations_for_mean_aggregation_graph_neural_networks.md)
- [\[NeurIPS 2025\] AudSemThinker: Enhancing Audio-Language Models through Reasoning over Semantics of Sound](audsemthinker_enhancing_audio-language_models_through_reasoning_over_semantics_o.md)
- [\[NeurIPS 2025\] Node-Based Editing for Multimodal Generation of Text, Audio, Image, and Video](node-based_editing_for_multimodal_generation_of_text_audio_image_and_video.md)
- [\[ICLR 2026\] AutoFigure: Generating and Refining Publication-Ready Scientific Illustrations](../../ICLR2026/audio_speech/autofigure_generating_and_refining_publication-ready_scientific_illustrations.md)

</div>

<!-- RELATED:END -->
