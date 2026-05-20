---
title: >-
  [Paper Note] TaxaDiffusion: Progressively Trained Diffusion Model for Fine-Grained Species Generation
description: >-
  [ICCV 2025][Image Generation][Taxonomic guidance] TaxaDiffusion leverages the hierarchical structure of biological taxonomy (Kingdom→Phylum→Class→Order→Family→Genus→Species) to progressively train a diffusion model…
tags:
  - "ICCV 2025"
  - "Image Generation"
  - "Taxonomic guidance"
  - "progressive training"
  - "fine-grained species generation"
  - "knowledge transfer"
  - "few-shot generation"
date: 2026-05-08
content_hash: c6a5713c5b95b5c4
---

# TaxaDiffusion: Progressively Trained Diffusion Model for Fine-Grained Species Generation

**Conference**: ICCV 2025
**arXiv**: [2506.01923](https://arxiv.org/abs/2506.01923)  
**Code**: [Project Page](https://amink8.github.io/TaxaDiffusion/)  
**Area**: Fine-Grained Image Generation / Diffusion Models
**Keywords**: Taxonomic guidance, progressive training, fine-grained species generation, knowledge transfer, few-shot generation

## TL;DR
TaxaDiffusion leverages the hierarchical structure of biological taxonomy (Kingdom→Phylum→Class→Order→Family→Genus→Species) to progressively train a diffusion model, gradually refining from high-level shared characteristics to species-level subtle distinctions. This approach achieves high-precision fine-grained animal image generation, reducing FID to 31.87 on the FishNet dataset (vs. 43.91 for LoRA), improving BioCLIP alignment score by 37%, and remaining effective for rare species with very few samples (even as few as 1).

## Background & Motivation

### Core Problem

While diffusion models have achieved remarkable success in generating images of general concepts, they face significant challenges in generating **fine-grained animal species** images: given a species name, existing models often fail to produce morphologically accurate images that faithfully reflect species identity.

### Two Domain-Specific Challenges

**High dynamism**: Animals exhibit extremely high degrees of freedom, displaying various poses and movements, leading to large intra-class variation.

**Data scarcity**: With millions of species on Earth, collecting sufficient samples per species to capture both intra-class variation and inter-class discriminability is practically infeasible.

### Key Insight

Despite the enormous number of species, they are **not independent categories**. Over 500 million years of evolution, a small number of ancestral species branched into millions. Species closer in the taxonomic tree share more visual characteristics. Species within the same Genus or Family often differ only in **subtle variations of shape, color, or pattern**.

### Intuitive Analogy

Millions of species did not appear all at once but evolved through progressive branching. The goal of this paper is to **teach the diffusion model to mimic this evolutionary process**—not learning all species simultaneously, but learning progressively.

### Limitations of Prior Work

- **Zero-shot generation** (Vanilla SD): Lacks fine-grained species knowledge, resulting in morphologically inaccurate images.
- **LoRA fine-tuning**: Provides all taxonomic level information at once, making it difficult for the model to learn hierarchical relationships effectively, especially for rare species.
- **Full fine-tuning**: Offers limited improvement at high computational cost.
- **FineDiffusion**: Uses only two taxonomic levels (superclass/subclass), insufficiently exploiting the hierarchy.

## Method

### Overall Architecture

TaxaDiffusion comprises three core components:

1. **Efficient domain adaptation**: Adapting the pretrained SD to the biological data domain using LoRA.
2. **Taxonomy-guided progressive training**: Incrementally training conditional encoding modules level by level (Kingdom→Species).
3. **Taxonomy-guided inference**: Using higher-level taxonomic signals in place of unconditional estimates for CFG.

### Key Design 1: Progressive Training

**Domain adaptation stage**: LoRA modules are added to SD v1.5 (Q/K/V/O projections in self-attention and cross-attention), and first trained at the Kingdom level for domain adaptation:

$$\mathbf{Q} = \mathcal{W}^Q \mathbf{z} + \alpha \cdot \mathbf{A}\mathbf{B}^T \mathbf{z}$$

**Conditional encoding modules**: A frozen CLIP text encoder converts taxonomic names into latent representations, which are then refined by a **trainable module containing two Transformer layers**. Each taxonomic level has an independent conditional encoding module.

**Progressive training procedure**:

1. **Stage 1 (Kingdom/Phylum/Class)**: Trains the corresponding conditional modules to learn the coarsest shared visual features (e.g., basic animal morphology).
2. **Stage 2 (Order)**: Freezes the previous stage's module parameters and trains the Order-level module to learn shared body shape and posture features at the level of, e.g., Carnivora.
3. **Stage 3 (Family)**: Continues downward to learn features of, e.g., Chaetodontidae (butterflyfishes), such as ocular black stripes and dorsal fin morphology.
4. **Stage 4 (Genus)**: Learns genus-level distinctions (e.g., vertical stripes vs. caudal black spots).
5. **Stage 5 (Species)**: Learns fine-grained species-level distinctions.

**Key mechanisms**:
- Each level is trained for 250K iterations.
- After training one level, its parameters are **frozen** before proceeding to the next.
- Conditional embeddings across levels are aggregated by **summation** and fed into the diffusion model.
- Only the current level's module remains trainable, ensuring stability of previously acquired knowledge.

### Key Design 2: Taxonomy-Guided Inference (TaxaGuide)

Standard CFG uses an unconditional estimate for guidance:

$$\tilde{\epsilon}_\theta(\mathbf{x}_t, t, \mathbf{c}) = (1+w) \times \epsilon_\theta(\mathbf{x}_t, t, \mathbf{c}) - w \times \epsilon_\theta(\mathbf{x}_t, t)$$

TaxaDiffusion replaces the unconditional estimate with a **Kingdom-level conditional estimate**:

$$\tilde{\epsilon}_\theta(\mathbf{x}_t, t, \mathbf{c}^{(i)}) = (1+w) \times \epsilon_\theta(\mathbf{x}_t, t, \mathbf{c}^{(i)}) - w \times \epsilon_\theta(\mathbf{x}_t, t, \mathbf{c}^{(0)})$$

**Intuition**: Using a high-level taxonomic condition as a "weak version" of self-guidance (analogous to Autoguidance) focuses the guidance direction on the **difference** between shared and species-specific features, rather than generating from scratch.

### Loss Function

Standard diffusion training loss:

$$\mathcal{L}(\theta) = \mathbb{E}_{\mathbf{x}_0, \epsilon, t}\left[\|\epsilon - \epsilon_\theta(\mathbf{x}_t, t, \mathbf{c})\|^2\right]$$

Each level is trained independently using the corresponding level's condition $\mathbf{c}^{(i)}$.

## Experiments

### FishNet Main Results

17,357 fish species across 5 taxonomic levels; the challenge lies in inter-species differences that are subtle yet biologically meaningful:

| Method | FID ↓ (Species) | LPIPS ↓ | BioCLIP ↑ (Species) |
|--------|-----------------|---------|---------------------|
| SD (Zero-shot) | 61.93 | 0.7737 | 3.35 |
| SD + LoRA | 43.91 | 0.7574 | 7.61 |
| SD + Full | 39.41 | 0.7574 | 8.31 |
| **TaxaDiffusion** | **31.87** | **0.7319** | **10.43** |

Key findings:
- TaxaDiffusion substantially outperforms all baselines across all metrics and all taxonomic levels.
- Compared to LoRA fine-tuning, FID decreases by ~27% (43.91→31.87) and BioCLIP improves by 37% (7.61→10.43).
- Clear advantages over full fine-tuning demonstrate that progressive training outperforms joint training across all levels simultaneously.

### iNaturalist Experiments

10,000 species spanning plants and animals:

| Method | FID ↓ (Species) | LPIPS ↓ | BioCLIP ↑ |
|--------|-----------------|---------|-----------|
| SD (Zero-shot) | 73.13 | 0.8134 | 6.1 |
| **TaxaDiffusion** | **46.39** | **0.7475** | **10.41** |

FID reduced by ~37% (73.13→46.39), validating the method's generalization to mixed-species datasets.

### Comparison with State of the Art

Comparison with FineDiffusion (using DiT-XL/2 architecture with only two taxonomic levels):

| Method | FID ↓ | LPIPS ↓ | BioCLIP ↑ |
|--------|-------|---------|-----------|
| FineDiffusion (DiT-XL/2) | 74.80 | 0.7613 | 6.46 |
| **TaxaDiffusion (U-Net)** | **43.71** | **0.7170** | **8.15** |

Even with the weaker U-Net architecture, TaxaDiffusion substantially surpasses DiT-based FineDiffusion (FID reduced by 42%).

### Ablation Study

Comparison of training strategies:

| Strategy | FID ↓ | LPIPS ↓ | BioCLIP ↑ |
|----------|-------|---------|-----------|
| All (all levels jointly) | 32.43 | 0.7487 | 9.32 |
| Random (random level selection) | 29.53 | 0.7429 | 9.85 |
| **Progressive** | **31.87** | **0.7319** | **10.43** |

Comparison of inference guidance strategies:

| Guidance | FID ↓ | LPIPS ↓ | BioCLIP ↑ |
|----------|-------|---------|-----------|
| Vanilla CFG | 47.61 | 0.7313 | 9.45 |
| **TaxaGuide** | **32.42** | **0.7092** | **11.53** |

Key findings:
- The Random strategy achieves a slightly lower FID (29.53) but inferior BioCLIP alignment compared to the progressive strategy (9.85 vs. 10.43), indicating that progressive training leads to generated images more faithful to species identity.
- TaxaGuide substantially improves both FID (47.61→32.42) and BioCLIP (9.45→11.53) over vanilla CFG.

### Trait Discovery

Images generated by TaxaDiffusion at different taxonomic levels exhibit a progressive emergence of biological traits:
- **Family: Chaetodontidae (butterflyfishes)**: Ocular black stripes, elongated dorsal fin.
- **Genus: Amphichaetodon**: Black vertical stripes.
- **Genus: Chaetodon**: Large black spot near the caudal fin.

The transition from Family to Genus preserves shared features (ocular stripes) while acquiring genus-specific characteristics (stripe patterns), demonstrating the hierarchical learning effect of progressive training.

## Highlights & Insights

1. **Domain knowledge-driven algorithm design**: Integrating the accumulated scientific framework of biological taxonomy into diffusion model training serves as an exemplar of interdisciplinary methodology.
2. **Progressive training as a simulation of evolution**: The coarse-to-fine training procedure mirrors the branching process of species evolution, enabling the model to first learn shared traits before learning distinctions.
3. **Knowledge transfer enables few-shot learning**: For rare species with only 1–5 samples, the model only needs to learn the subtle differences from congeneric species in the final stage, with all prior stage knowledge transferring from phylogenetically related species.
4. **Elegant design of TaxaGuide**: Replacing the unconditional estimate with a high-level taxonomic condition focuses the guidance direction on the transition from shared to species-specific features.
5. **Trait discovery application**: Generated images can assist biologists in discovering and visualizing evolutionary traits at different taxonomic levels.

## Limitations & Future Work

1. The current use of U-Net (SD v1.5) without leveraging more modern Transformer architectures (e.g., DiT) may limit the performance ceiling.
2. Training 250K iterations per taxonomic level results in long total training time across seven levels.
3. Validation is limited to animal and plant datasets; generalization to other fine-grained domains (e.g., minerals, architectural styles) remains unexplored.
4. Taxonomy does not fully correspond to evolutionary relationships (e.g., genetic similarity), and visual similarity is not always linearly correlated with taxonomic distance.

## Related Work & Insights

- **Fine-grained generation**: sketch-based control, attribute manipulation, personalized generation → Ours (taxonomy-conditioned progressive training)
- **Progressive hierarchical generation**: VQ-VAE multi-scale autoencoders, HI-Diff (hierarchical diffusion deblurring) → Ours (taxonomy-level progressive diffusion)
- **Diffusion model fine-tuning**: LoRA, DreamBooth, Textual Inversion → Ours (LoRA domain adaptation + progressive conditional modules)

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Introducing taxonomic hierarchy into the diffusion training paradigm; biology-inspired design is highly original.
- Technical Depth: ⭐⭐⭐⭐ — Simple yet effective design; TaxaGuide inference guidance is theoretically grounded.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Three datasets + SOTA comparison + comprehensive ablations + trait discovery case studies.
- Value: ⭐⭐⭐⭐ — Direct applicability to biodiversity research (species identification, trait visualization).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] CharaConsist: Fine-Grained Consistent Character Generation](characonsist_fine-grained_consistent_character_generation.md)
- [\[ICCV 2025\] ShortFT: Diffusion Model Alignment via Shortcut-based Fine-Tuning](shortft_diffusion_model_alignment_via_shortcut-based_fine-tuning.md)
- [\[ICCV 2025\] Multimodal Latent Diffusion Model for Complex Sewing Pattern Generation](multimodal_latent_diffusion_model_for_complex_sewing_pattern_generation.md)
- [\[ICCV 2025\] FlowEdit: Inversion-Free Text-Based Editing Using Pre-Trained Flow Models](flowedit_inversion-free_text-based_editing_using_pre-trained_flow_models.md)
- [\[ICCV 2025\] HypDAE: Hyperbolic Diffusion Autoencoders for Hierarchical Few-shot Image Generation](hypdae_hyperbolic_diffusion_autoencoders_for_hierarchical_few-shot_image_generat.md)

</div>

<!-- RELATED:END -->
