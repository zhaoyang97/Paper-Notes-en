---
title: >-
  [Paper Note] ArtFormer: Controllable Generation of Diverse 3D Articulated Objects
description: >-
  [CVPR 2025][Text Generation][Articulated Objects] This work proposes the ArtFormer framework, which generates high-quality, diverse, and kinematically accurate 3D articulated objects from text/image descriptions via tree structure parameterization and a conditional diffusion shape prior, significantly outperforming existing methods in generation quality and diversity.
tags:
  - "CVPR 2025"
  - "Text Generation"
  - "Articulated Objects"
  - "Tree Structure Parameterization"
  - "Shape Prior"
  - "Controllable Generation"
  - "Text/Image Guidance"
date: 2026-05-08
content_hash: 2e71c1b91d87c122
---

# ArtFormer: Controllable Generation of Diverse 3D Articulated Objects

**Conference**: CVPR 2025  
**arXiv**: [2412.07237](https://arxiv.org/abs/2412.07237)  
**Code**: [https://github.com/ShuYuMo2003/ArtFormer](https://github.com/ShuYuMo2003/ArtFormer)  
**Area**: 3D Generation / Articulated Object Modeling  
**Keywords**: Articulated Objects, Tree Structure Parameterization, Shape Prior, Controllable Generation, Text/Image Guidance

## TL;DR
This work proposes the ArtFormer framework, which generates high-quality, diverse, and kinematically accurate 3D articulated objects from text/image descriptions via tree structure parameterization and a conditional diffusion shape prior, significantly outperforming existing methods in generation quality and diversity.

## Background & Motivation

### Background

**Background**: Research on generating articulated objects (multiple rigid bodies connected by joints) remains limited. NAP is restricted by pre-defined graph structures, while CAGE/SINGAPO rely on retrieval-based methods, which limits diversity.

**Key Challenge**: Quality vs. flexibility (fixed structures restrict diversity) and geometric quality vs. kinematic accuracy (conflicts between the two constraints).

**Key Insight**: Articulated objects are essentially tree-structured $\rightarrow$ Tree Position Encoding captures hierarchical relationships + Shape Prior ensures geometric quality + Gumbel-Softmax expands diversity.

### Proposed Approach

**Goal**:  
### Overall Architecture
**Phase 1**: Shape Prior Pre-training (using VAE + SDF + conditional diffusion model to learn geometric priors)  
**Phase 2**: Articulation Transformer (using Tree Position Encoding + iterative decoding to generate nodes layer by layer)  

### Key Designs

1. **Tree Structure Parameterization**: Each node stores geometric attributes (bbox $b_i \in \mathbb{R}^6$, latent code $z_i \in \mathbb{R}^{768}$).


## Method

### Overall Architecture
**Phase 1**: Shape Prior Pre-training (using VAE + SDF + conditional diffusion model to learn geometric priors)  
**Phase 2**: Articulation Transformer (using Tree Position Encoding + iterative decoding to generate nodes layer by layer)  

### Key Designs

1. **Tree Structure Parameterization**: Each node stores geometric attributes (bbox $b_i \in \mathbb{R}^6$, latent code $z_i \in \mathbb{R}^{768}$) + kinematic attributes (joint axis $j_i \in \mathbb{R}^6$, motion range $l_i \in \mathbb{R}^4$), with a total dimension of $D=785$.

2. **Shape Prior (Gumbel-Softmax Sampling)**: Decomposes the geometric code into 4 components, expanding the latent space from $4N$ to $N^4$ via 4 independent codebooks. This significantly increases diversity without increasing computational costs.

3. **Tree Position Encoding (TPE)**: Encodes the root-to-node path using a bidirectional GRU for absolute position, and concatenates path node encodings for relative position. This supports arbitrary tree structures.

4. **Iterative Decoding**: Predicts child nodes for existing nodes round by round until all outputs are termination tokens, effectively capturing inter-part dependencies.

### Loss & Training
$L_{trans} = \beta_o L_o + \beta_P L_P + L_a$ (termination classification + codebook KL + attribute MSE)

## Key Experimental Results

### Main Results

| Method | POR↓ | MMD↓ | COV↑ | DS↑ |
|------|------|------|------|-----|
| NAP-128 | 0.805 | 0.3085 | 0.7021 | 0.13 |
| CAGE | 0.251 | 0.6064 | 0.5319 | 0.07 |
| **Ours** | **0.709** | **0.5213** | **0.5266** | **0.67** |

### Ablation Study

| Configuration | POR↓ | MMD↓ | COV↑ |
|------|------|------|------|
| Full Model | 0.709 | 0.5213 | 0.5266 |
| w/o TPE | 1.170% | 0.5000 | 0.5053 |
| w/o Shape Prior | 2.502% | 0.4574 | 0.7606 |

### Key Findings
- The diversity metric (DS=0.67) is significantly higher than existing methods (CAGE is only 0.07).
- After removing TPE, POR increases by 65%, demonstrating the necessity of positional information.
- All metrics severely degrade when the Shape Prior is removed.

## Highlights & Insights
- Tree structure parameterization elegantly simplifies the representation of articulated objects.
- Gumbel-Softmax expands the diversity space ($4N \rightarrow N^4$) without increasing computational overhead.
- Tree Position Encoding (TPE) is a key innovation, as clearly demonstrated by the ablation study.

## Limitations & Future Work
- Training was conducted on only 6 object categories, with limited sample sizes for some categories.
- Quantitative conditional control (e.g., rotation angles) is challenging.
- Multi-category training using SDF exhibits a drop in generalization performance.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Integration of Tree Position Encoding, Shape Prior, and Gumbel-Softmax
- Experimental Thoroughness: ⭐⭐⭐⭐ Detailed ablation study; lacks cross-dataset validation
- Writing Quality: ⭐⭐⭐⭐⭐ Clear figures and tables, logically rigorous
- Value: ⭐⭐⭐⭐ Strong potential for applications in robotics and digital twins

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Unveiling the Potential of Diffusion Large Language Model in Controllable Generation](../../ICLR2026/nlp_generation/unveiling_the_potential_of_diffusion_large_language_model_in_controllable_genera.md)
- [\[ACL 2026\] Difficulty-Controllable Cloze Question Distractor Generation](../../ACL2026/nlp_generation/difficulty-controllable_cloze_question_distractor_generation.md)
- [\[ACL 2025\] Odysseus Navigates the Sirens' Song: Dynamic Focus Decoding for Factual and Diverse Open-Ended Text Generation](../../ACL2025/nlp_generation/odysseus_dynamic_focus_decoding.md)
- [\[ICLR 2026\] Diverse Text Decoding via Iterative Reweighting](../../ICLR2026/nlp_generation/diverse_text_decoding_via_iterative_reweighting.md)
- [\[ACL 2025\] Principled Content Selection to Generate Diverse and Personalized Multi-Document Summaries](../../ACL2025/nlp_generation/dpp_diverse_multidoc_summary.md)

</div>

<!-- RELATED:END -->
