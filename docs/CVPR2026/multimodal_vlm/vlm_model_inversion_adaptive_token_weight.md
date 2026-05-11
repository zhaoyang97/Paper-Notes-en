---
title: >-
  [Paper Note] Do Vision-Language Models Leak What They Learn? Adaptive Token-Weighted Model Inversion Attacks
description: >-
  [CVPR 2026][Multimodal VLM][Model inversion attack] This paper presents the first systematic study of model inversion (MI) attacks against VLMs, proposing a suite of inversion strategies tailored to token generation (TMI/TMI-C/SMI) and an adaptive attention-weighted method SMI-AW that dynamically weights token gradient contributions based on visual attention intensity. Evaluated across 4 VLMs and 3 datasets, SMI-AW achieves up to 61.21% human-evaluated attack accuracy, revealing severe training data privacy leakage risks in VLMs.
tags:
  - CVPR 2026
  - Multimodal VLM
  - Model inversion attack
  - VLM privacy leakage
  - adaptive token weighting
  - visual attention guidance
  - training data reconstruction
date: 2026-05-08
content_hash: 00c8f345ece398ad
---

# Do Vision-Language Models Leak What They Learn? Adaptive Token-Weighted Model Inversion Attacks

**Conference**: CVPR 2026
**arXiv**: [2508.04097](https://arxiv.org/abs/2508.04097)
**Code**: [https://ngoc-nguyen-0.github.io/SMI_AW/](https://ngoc-nguyen-0.github.io/SMI_AW/)
**Area**: Multimodal VLM / AI Security
**Keywords**: Model inversion attack, VLM privacy leakage, adaptive token weighting, visual attention guidance, training data reconstruction

## TL;DR
This paper presents the first systematic study of model inversion (MI) attacks against VLMs, proposing a suite of inversion strategies tailored to token generation (TMI/TMI-C/SMI) and an adaptive attention-weighted method SMI-AW that dynamically weights token gradient contributions based on visual attention intensity. Evaluated across 4 VLMs and 3 datasets, SMI-AW achieves up to 61.21% human-evaluated attack accuracy, revealing severe training data privacy leakage risks in VLMs.

## Background & Motivation
Model inversion (MI) attacks aim to reconstruct private training data from trained models and have been extensively studied for unimodal DNNs, especially face recognition. However, VLMs present unique characteristics that prevent direct application of conventional MI methods:

1. VLM outputs are token sequences rather than class labels, requiring new inversion objective functions.
2. VLMs comprise multiple modules (visual encoder, projection layer, language model), with the visual encoder typically frozen—private information is primarily embedded in the language model and projection layer parameters.
3. Different output tokens vary in their dependence on visual input—some tokens are strongly visually grounded, while others are driven purely by linguistic context.

As VLMs are increasingly deployed in sensitive domains such as healthcare and finance, understanding their privacy risks is of urgent importance.

## Core Problem
Are VLMs as vulnerable to model inversion attacks as unimodal DNNs? How can effective MI attacks be designed to exploit the token generation characteristics of VLMs?

## Method

### Overall Architecture
White-box attack setting: the adversary has full access to the VLM's architecture, parameters, and attention maps. Given a text prompt $t$ (e.g., "Who is the person in the image?") and a target answer $y$ (e.g., a person's name), the method optimizes a latent code $w$ in the latent space of a pretrained StyleGAN2 such that $x = G(w)$ causes the VLM to output $y$.

### Key Designs

1. **Token-based MI (TMI)**: Iterates token-by-token—computes the inversion loss for each token $y_i$ in the sequence and updates the latent variable $w$ independently. All $m$ tokens are traversed once per round. Limitation: per-token gradients are noisy, and gradients from weakly visually grounded tokens may mislead optimization.

2. **Convergent Token-based MI (TMI-C)**: Applies $K$ updates per token until convergence before proceeding to the next. Limitation: convergence directions are unstable, resulting in the lowest match rates (<30%).

3. **Sequence-based MI (SMI)**: Aggregates losses across all tokens into a unified objective $\mathcal{L} = \frac{1}{m}\sum_{i=1}^m \mathcal{L}_{inv}(M(t, G(w), y_{<i}), y_i)$, updating $w$ with a global gradient at each step. Achieves match rates >95%, substantially outperforming TMI.

4. **SMI-AW (core contribution)**: Observes that different tokens exhibit different levels of visual attention—tokens with strong visual grounding (e.g., descriptive parts of a name) have high cross-attention values and carry richer visual information in their gradients, whereas language-driven tokens (e.g., articles) have weak attention and less informative gradients. SMI-AW dynamically computes per-token weights $\beta_i = \alpha_i / \sum_j \alpha_j$ from cross-attention values $\alpha_i$ and aggregates the loss as $\mathcal{L} = \sum_{i=1}^m \beta_i \mathcal{L}_{inv}$. Crucially, the weights are updated dynamically at each inversion step, since token dependence on visual input evolves as the reconstructed image progressively approximates the target.

### Loss & Training
- Three inversion losses are evaluated: cross-entropy $\mathcal{L}_{CE}$, maximum margin loss $\mathcal{L}_{MML}$, and logit maximization $\mathcal{L}_{LOM}$ (best performing).
- $\mathcal{L}_{LOM}$ directly maximizes the target token's logit with regularization to prevent unbounded growth.
- Inversion steps $N = 70$, update rate $\lambda = 0.05$.
- Initial candidate selection: 2,000 latent codes $w$ are sampled and the top-16 low-loss candidates are selected; final selection: 8 optimal samples chosen after 10 random augmentations.

## Key Experimental Results

### FaceScrub Dataset (LLaVA-v1.6-7B)

| Method | AttAcc_M ↑ | AttAcc_D Top1 ↑ | AttAcc_D Top5 ↑ | δ_face ↓ |
|--------|-----------|-----------------|-----------------|---------|
| TMI | 42.20% | 18.03% | 40.25% | 0.8901 |
| TMI-C | 16.08% | 3.85% | 11.64% | 1.1825 |
| SMI | 57.83% | 33.50% | 61.56% | 0.7473 |
| **SMI-AW** | **61.01%** | **37.62%** | **66.16%** | **0.7265** |

### Cross-Dataset Results (LLaVA-v1.6-7B + SMI-AW)

| Dataset | AttAcc_M ↑ | AttAcc_D Top1 ↑ |
|---------|-----------|-----------------|
| FaceScrub | 61.01% | 37.62% |
| CelebA | 67.05% | 45.25% |
| StanfordDogs | 78.13% | 55.83% |

### Cross-Model Results (FaceScrub + SMI-AW)

| VLM | AttAcc_M ↑ | δ_eval ↓ |
|-----|-----------|---------|
| LLaVA-v1.6-7B | 61.01% | 134.94 |
| InternVL2.5-8B | 55.05% | 139.18 |
| MiniGPT-v2 | 47.92% | 161.25 |
| Qwen2.5-VL-7B | 32.03% | 150.46 |

### Human Evaluation

| VLM | Dataset | AccAcc_H ↑ |
|-----|---------|-----------|
| LLaVA-v1.6-7B | CelebA | **61.21%** |
| LLaVA-v1.6-7B | FaceScrub | 56.93% |
| MiniGPT-v2 | FaceScrub | 57.22% |

### Ablation Study
- **Sequence vs. token**: Sequence-based methods achieve target match rates >95%, whereas token-based methods reach only 60–79% (TMI-C <30%), confirming that global gradient signals are more stable.
- **Adaptive vs. uniform weighting**: SMI-AW consistently outperforms SMI across all metrics, validating the effectiveness of visual attention-guided weighting.
- **Loss functions**: $\mathcal{L}_{LOM}$ performs best, followed by $\mathcal{L}_{CE}$; $\mathcal{L}_{MML}$ performs worst.
- **Prompt robustness**: Varying input prompts has minimal impact on attack performance (AttAcc_M ranges 59–61%).
- **Attack on public models**: Celebrity facial images are successfully reconstructed from publicly released LLaVA-v1.6-7B and MiniGPT-v2.

## Highlights & Insights
- **Pioneering problem formulation**: This is the first systematic study of model inversion attacks against VLMs, addressing an important gap in multimodal privacy security research.
- **Key insight**: Different output tokens exhibit varying degrees of visual grounding, and this grounding evolves dynamically across inversion steps—a property unique to VLMs that has no counterpart in unimodal MI.
- **Elegant method design**: Cross-attention maps serve as a proxy for gradient informativeness, converting VLM-internal mechanisms into an attack advantage.
- **Practical validation**: Successful reconstruction of celebrity faces from publicly deployed VLMs demonstrates that the privacy risk is realistic rather than merely theoretical.
- **Large-scale human evaluation**: Results are validated by 4,240–8,000 crowdsourced participants, lending credibility to the evaluation.

## Limitations & Future Work
- **White-box assumption**: In realistic scenarios, adversaries may not have full access to model parameters and attention maps.
- **Domain coverage**: Experiments are limited to face and dog breed datasets; generalization to natural scenes or medical images remains unexplored.
- **Frozen visual encoder assumption**: Attack effectiveness may differ if the visual encoder is also fine-tuned.
- **Defenses not explored**: The paper focuses exclusively on attacks and does not propose concrete defense mechanisms.
- **Weaker performance on Qwen2.5-VL** (only 32%), likely due to architectural differences, warrants further analysis.

## Related Work & Insights
- **vs. conventional MI (GMI/PPA/KEDMI)**: Traditional methods perform inversion against class labels of classification models; this paper extends MI to token sequence generation in VLMs, requiring fundamentally new optimization strategies.
- **vs. MI under contrastive learning**: Prior work primarily investigates alignment leakage in contrastive models such as CLIP; this paper targets the generative language modeling stage of VLMs, representing a different attack surface.
- **vs. federated learning privacy attacks**: Gradient inversion attacks in FL rely on intercepting gradients during training; this paper operates on already-trained models without requiring access to training-time gradients.

**Implications and connections:**
- **VLM privacy defense**: The attack surface identified in this paper suggests the need for privacy-preserving measures during VLM training—such as differential privacy, regularization, or decoy-signal mechanisms analogous to Trap-MID.
- **Connection to RED (Rationale-Enhanced Decoding)**: Both papers exploit the varying degrees of visual grounding among VLM output tokens, but in opposite directions—RED leverages this property to enhance reasoning, while SMI-AW leverages it to strengthen attacks.
- **Multimodal security**: As VLMs are increasingly adopted in high-stakes medical applications (e.g., radiology report generation), the real-world risks of such attacks cannot be overlooked.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First extension of MI attacks to VLMs; the problem is significant and the method design is well-motivated.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 4 VLMs, 3 datasets, 5 evaluation metrics (including large-scale human evaluation), and validation on publicly released models.
- Writing Quality: ⭐⭐⭐⭐ Logic is clear and problem formulation is precise; supplementary material is overly extensive and could be condensed.
- Value: ⭐⭐⭐⭐⭐ Carries significant implications for privacy security in VLM deployment and opens a new research direction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Where MLLMs Attend and What They Rely On: Explaining Autoregressive Token Generation](where_mllms_attend_and_what_they_rely_on_explaining_autoregressive_token_generat.md)
- [\[CVPR 2026\] HulluEdit: Single-Pass Evidence-Consistent Subspace Editing for Mitigating Hallucinations in Large Vision-Language Models](hulluedit_single-pass_evidence-consistent_subspace_editing_for_mitigating_halluc.md)
- [\[CVPR 2026\] Aligning What Vision-Language Models See and Perceive with Adaptive Information Flow](aif_adaptive_information_flow_vlm.md)
- [\[CVPR 2026\] What Do Visual Tokens Really Encode? Uncovering Sparsity and Redundancy in Multimodal Large Language Models](what_do_visual_tokens_really_encode_uncovering_sparsity_and_redundancy_in_multim.md)
- [\[CVPR 2026\] Adaptive Vision-Language Model Routing for Computer Use Agents](adaptive_vision-language_model_routing_for_computer_use_agents.md)

</div>

<!-- RELATED:END -->
