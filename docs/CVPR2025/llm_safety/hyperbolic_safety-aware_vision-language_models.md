---
title: >-
  [Paper Note] Hyperbolic Safety-Aware Vision-Language Models
description: >-
  [CVPR 2025][LLM Safety][Hyperbolic space] HySAC proposes constructing safety-aware vision-language models in hyperbolic space. By mapping safe and unsafe content to different regions of hyperbolic space via entailment cones (safe content near the origin, unsafe content far from the origin), the model is equipped with safe content classification and dynamic redirection capabilities, significantly outperforming existing unlearning methods in retrieval safety and NSFW detection.
tags:
  - "CVPR 2025"
  - "LLM Safety"
  - "Hyperbolic space"
  - "content safety"
  - "CLIP"
  - "entailment learning"
  - "NSFW detection"
date: 2026-05-08
content_hash: 7f9345c17932768c
---

# Hyperbolic Safety-Aware Vision-Language Models

**Conference**: CVPR 2025  
**arXiv**: [2503.12127](https://arxiv.org/abs/2503.12127)  
**Code**: [https://github.com/aimagelab/HySAC](https://github.com/aimagelab/HySAC)  
**Area**: Multimodal VLM  
**Keywords**: Hyperbolic space, content safety, CLIP, entailment learning, NSFW detection

## TL;DR
HySAC proposes constructing safety-aware vision-language models in hyperbolic space. By mapping safe and unsafe content to different regions of hyperbolic space via entailment cones (safe content near the origin, unsafe content far from the origin), the model is equipped with safe content classification and dynamic redirection capabilities, significantly outperforming existing unlearning methods in retrieval safety and NSFW detection.

## Background & Motivation
- **Background**: VLMs like CLIP are trained on large-scale web data, inevitably containing unsafe content such as violence and pornography. Existing methods (e.g., Safe-CLIP) mainly eliminate the model's knowledge of unsafe concepts through "unlearning".
- **Limitations of Prior Work**: Although unlearning methods effectively reduce unsafe outputs, they limit the model's ability to distinguish between safe and unsafe content—the model loses discriminative power. This is unfavorable for scenarios such as content moderation and user-defined filtering.
- **Key Challenge**: Unlearning vs. perception—forgetting NSFW knowledge means the model cannot detect unsafe content, nor can it provide safe alternatives.
- **Goal**: Transition from the "unlearning" paradigm to the "perception" paradigm, allowing the model to simultaneously understand what is safe and what is unsafe.
- **Key Insight**: Utilizing the natural hierarchical representation capability of hyperbolic space to organize safe and unsafe content into an entailment hierarchy.
- **Core Idea**: Safe content represents general concepts (closer to the origin), while unsafe content represents their concretization (further from the origin), modeling this asymmetric relationship via entailment cones.

## Method

### Overall Architecture
HySAC is based on the CLIP architecture, projecting the outputs of visual and text encoders to the Lorentz hyperbolic space via exponential mapping. Fine-tuning is performed using a dataset of safe/unsafe image-text quadruplets $(I_k, T_k, I_k^\star, T_k^\star)$. The training objective consists of two components: hyperbolic safety contrastive learning (aligning image-text pairs) and hyperbolic safety entailment learning (establishing a safety hierarchy). During inference, a safety traversal mechanism is utilized to move query embeddings within the hyperbolic space, achieving dynamic redirection and retrieval between safe and unsafe content.

### Key Designs
1. **Hyperbolic Space Embedding and Entailment Hierarchy**:
    - **Function**: To establish an explicit hierarchical structure of safe content in the embedding space.
    - **Mechanism**: The Lorentz model is used to implement the hyperbolic space. A four-level entailment inequality chain is defined: $g_T(T_k) \ll g_I(I_k) \ll g_T(T_k^\star) \ll g_I(I_k^\star)$, meaning safe text is closest to the origin, followed by safe images, then unsafe text, and unsafe images are the furthest. The Euclidean encoder outputs are projected onto the hyperboloid using learnable projection scalars $\alpha_{img}$, $\alpha_{txt}$, and the exponential map.
    - **Design Motivation**: Hyperbolic space is naturally suited for representing hierarchical structures (allowing low-distortion embedding of tree-like structures). The entailment relationship simultaneously preserves hierarchies across both the intra-modal (text-image) and safety (safe-unsafe) dimensions.

2. **Joint Hyperbolic Safety Contrastive + Entailment Loss**:
    - **Function**: To align image-text pairs in hyperbolic space while segregating safe and unsafe regions.
    - **Mechanism**: The contrastive loss $L_{\text{hSC}}$ is calculated based on negative Lorentz distances for safe pairs $(I,T)$, unsafe pairs $(I^\star,T^\star)$, and cross-safety pairs, respectively. The entailment loss $L_{\text{hSE}}$ utilizes the entailment cone's aperture $\omega(\mathbf{q}) = \sin^{-1}(\frac{2K}{\sqrt{\kappa}\|\tilde{\mathbf{q}}\|})$ and external angle $\phi$ to constrain the image to lie within the corresponding text's cone, and the unsafe text to lie within the safe image's cone. The overall loss is $L = L_{\text{hSC}} + L_{\text{hSE}}$.
    - **Design Motivation**: Contrastive learning alone cannot establish a safety hierarchy (entailment relations are invalid in Euclidean space), making the entailment loss key to realizing safety awareness. Joint training ensures both retrieval performance and safety awareness are optimized simultaneously.

3. **Safety Traversals**:
    - **Function**: To dynamically adjust the position of query embeddings between safe and unsafe regions during inference.
    - **Mechanism**: Calculates the average distance $\mu_X$ from each content class $X \in \{T, I, T^\star, I^\star\}$ to the root feature $\mathbf{r}$, and defines the boundary $\tau_X = \mu_X + \tanh(\frac{\mu_X - \alpha}{\kappa}) + 1$. The query is shifted along the direction vector $\mathbf{v}_{\text{dir}} = \mathbf{q} - \mathbf{r}$ to the target boundary: $\mathbf{q}^* = \mathbf{r} + \tau_X \cdot \frac{\mathbf{v}_{\text{dir}}}{\|\mathbf{v}_{\text{dir}}\|_L}$.
    - **Design Motivation**: Distance-based region separation allows simple directional movement to switch between safe and unsafe content, providing flexible content moderation control for users.

### Loss & Training
- Uses the AdamW optimizer with weight decay=0.2, batch size=256, trained for 20 epochs.
- The visual and text encoders are fine-tuned using LoRA (r=16) to reduce the number of parameters.
- Key hyperparameters: temperature $\tau=0.07$, projection scalar initialization $\alpha_{img}=\alpha_{txt}=1/\sqrt{512}$, and curvature $c=1.0$ (learnable).
- All scalars are learned in log-space. Mixed-precision training is used (exponential map and loss are calculated in FP32 to ensure numerical stability).

## Key Experimental Results

### Main Results — Safe Retrieval (ViSU Test Set)

| Model | Safe T→I R@1 | Safe I→T R@1 | Unsafe→Safe T→I R@1 | Unsafe→Safe I→T R@1 |
|------|-------------|-------------|-------------------|-------------------|
| CLIP | 36.8 | 39.8 | 2.0 | 4.6 |
| Safe-CLIP | 45.9 | 45.3 | 8.0 | 19.1 |
| MERU⋆ | 50.0 | 51.2 | 2.3 | 5.7 |
| **HySAC** | **49.8** | **48.2** | **30.5** | **42.1** |

### Ablation Study

| Configuration | Safe T→I R@1 | Unsafe→Safe T→I R@1 | Unsafe→Safe I→T R@1 | Description |
|------|-------------|-------------------|------|------|
| w/o Ent (contrastive only) | 52.3 | 4.1 | 5.5 | Without entailment loss, safety redirection fails |
| w/o S-Ent (no safety entailment) | 51.0 | 1.4 | 7.4 | Without safety hierarchy, redirection is virtually ineffective |
| HySAC (Full) | 49.8 | 30.5 | 42.1 | Safety redirection is significantly improved |

### Key Findings
- HySAC far outperforms Safe-CLIP in unsafe$\rightarrow$safe retrieval (30.5 vs 8.0 R@1) while maintaining competitiveness in purely safe retrieval.
- For unsafe content retrieval, HySAC also achieves the best performance (R@1: 81.4 vs CLIP 73.1), proving that the perception paradigm is more comprehensive than the unlearning paradigm.
- Embedding distance distribution visualization clearly shows a four-layer separated hierarchical structure (safe text $\rightarrow$ safe image $\rightarrow$ unsafe text $\rightarrow$ unsafe image).
- On real-world NSFW datasets (NudeNet, NSFW URLs, SMID), HySAC achieves a 96.2% safe retrieval rate.
- HySAC can also serve as an NSFW classifier, reaching competitive performance on NudeNet and Mixed NSFW, although not originally designed for classification.

## Highlights & Insights
- The shift in paradigm from "perception over unlearning" is the core contribution: retaining the model's knowledge of unsafe content while granting it the ability to distinguish is more flexible and controllable than simple erasure.
- Entailment cones in hyperbolic space are naturally suited for modeling the asymmetric inclusion relationship of "safe $\rightarrow$ unsafe", a hierarchy that cannot be effectively represented in Euclidean space.
- The safety traversal mechanism elegantly achieves flexible control at inference time; a single model can simultaneously serve three purposes: safe retrieval, content moderation, and NSFW classification.
- The design of the four-level inequality chain $g_T(T) \ll g_I(I) \ll g_T(T^\star) \ll g_I(I^\star)$ cleanly and comprehensively covers the hierarchies of both modalities and safety dimensions.
- Embedding space visualization (Figure 2) intuitively illustrates the clear separation of the four content types, validating the efficacy of the method.
- The LoRA fine-tuning strategy controls computational overhead, facilitating rapid adaptation of existing CLIP weights.

## Limitations & Future Work
- Flexibility of threshold settings in safety traversal: $\tau_X$ is sensitive to hyperparameters, and estimating $\mu_X$ requires a certain number of prior samples.
- Some parameters, such as $\alpha=0.8$, require empirical tuning and may need recalibration under different dataset distributions.
- Validation is limited to CLIP (ViT-L/14) level retrieval models; generalizability to generative VLMs (e.g., diffusion models) or larger models remains to be explored.
- The safe/unsafe pairs in the ViSU dataset are structurally constructed, which may differ from the distribution of real-world unsafe content.
- Safety traversals may introduce semantic shifts—while redirected retrievals are safe, relevance may decrease (evaluating this aspect is hindered by the lack of safe-alternative datasets).
- Current safety classification is based on 20 fixed NSFW categories; emerging harmful content types (such as deepfakes or AI-generated content) are not yet integrated.
- The requirement of FP32 precision makes training computationally more expensive than standard CLIP fine-tuning.

## Related Work & Insights
- **vs. Safe-CLIP**: Safe-CLIP erases unsafe knowledge through unlearning, whereas HySAC retains the knowledge but establishes hierarchical separation. This makes HySAC more flexible and controllable, outperforming Safe-CLIP across all retrieval metrics.
- **vs. MERU**: MERU models modality hierarchies (text $\rightarrow$ image) in hyperbolic space without considering safety; HySAC expands this to a dual-hierarchy of safety + modality.
- **vs. HyCoCLIP**: HyCoCLIP enhances visual understanding of hyperbolic CLIP using object-level compositions; HySAC introduces safety entailment as an entirely new dimension.
- **vs. Schramowsky et al.**: Their method uses negative guidance of NSFW concepts, which constitutes an inference-time intervention; HySAC achieves fundamental safety awareness by restructuring the embedding space during training.
- **vs. NudeNet/Q16**: These are dedicated classifiers. HySAC achieves comparable performance on NSFW classification while additionally supporting retrieval and redirection.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First to apply hyperbolic entailment cones to VLM safety; the paradigm shift from unlearning to perception is pioneering.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive evaluation on ViSU and convincing validation on real NSFW datasets, though validation on generative models is lacking.
- **Writing Quality**: ⭐⭐⭐⭐ Rigorous mathematical formulation, but the dense equations increase the reading threshold.
- **Value**: ⭐⭐⭐⭐ Provides a novel pathway for VLM safety, although the effectiveness of safety traversal in practical deployment requires further validation.

---

> This note is generated based on a full reading of the paper, covering Preliminaries, Method, Experiments, and Analysis.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Zero-Shot Robustness of Vision Language Models Via Confidence-Aware Weighting](../../NeurIPS2025/llm_safety/zero-shot_robustness_of_vision_language_models_via_confidence-aware_weighting.md)
- [\[NeurIPS 2025\] Geo-Sign: Hyperbolic Contrastive Regularisation for Geometrically Aware Sign Language Translation](../../NeurIPS2025/llm_safety/geo-sign_hyperbolic_contrastive_regularisation_for_geometrically_aware_sign_lang.md)
- [\[NeurIPS 2025\] Approximate Domain Unlearning for Vision-Language Models](../../NeurIPS2025/llm_safety/approximate_domain_unlearning_for_visionlanguage_models.md)
- [\[CVPR 2025\] CleanSight: Test-Time Attention Purification for Backdoored Large Vision Language Models](test-time_attention_purification_for_backdoored_large_vision_language_models.md)
- [\[CVPR 2025\] ForensicZip: More Tokens are Better but Not Necessary in Forensic Vision-Language Models](forensiczip_more_tokens_are_better_but_not_necessary_in_forensic_vision-language.md)

</div>

<!-- RELATED:END -->
