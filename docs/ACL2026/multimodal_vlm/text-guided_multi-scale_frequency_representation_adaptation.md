---
title: >-
  [Paper Note] Text-Guided Multi-Scale Frequency Representation Adaptation
description: >-
  [ACL2026][Multimodal VLM][Frequency domain adaptation] This paper proposes FreqAdapter: visual and text embeddings from CLIP/LLaVA are first transformed into the DCT frequency domain. Visual frequency representations are…
tags:
  - "ACL2026"
  - "Multimodal VLM"
  - "Frequency domain adaptation"
  - "DCT"
  - "Multi-scale features"
  - "Text-guided"
  - "CLIP/LLaVA"
date: 2026-05-08
content_hash: 1da565f43a217005
---

# Text-Guided Multi-Scale Frequency Representation Adaptation

**Conference**: ACL2026  
**arXiv**: [2605.08181](https://arxiv.org/abs/2605.08181)  
**Code**: https://github.com/Kelvin-ywc/FreqAdapter  
**Area**: Multimodal VLM / Parameter-Efficient Fine-Tuning  
**Keywords**: Frequency domain adaptation, DCT, Multi-scale features, Text-guided, CLIP/LLaVA

## TL;DR
This paper proposes FreqAdapter: visual and text embeddings from CLIP/LLaVA are first transformed into the DCT frequency domain. Visual frequency representations are then fine-tuned using text-guided multi-scale global adaptation and cross-modal modulation. With approximately 0.11% additional parameters, it consistently outperforms common prompt/adapter methods in image-text retrieval and VQA.

## Background & Motivation
**Background**: Multimodal foundation models like CLIP and LLaVA possess strong vision-language representation capabilities but still require adaptation for new data distributions or downstream tasks. To reduce training costs, the community commonly uses prompt tuning, adapter tuning, LoRA, or visual prompting methods, updating only a few parameters to improve performance on tasks like image-text retrieval and VQA.

**Limitations of Prior Work**: Most parameter-efficient fine-tuning (PEFT) methods perform uniform adjustments directly in the spatial or feature domains. This presents two issues: first, spatial/patch representations contain significant redundant information, making limited parameters prone to fitting noise or local distributions; second, many methods treat all tokens and feature channels equally, failing to explicitly leverage the multi-scale structure of visual signals or fully incorporate textual semantics into visual adaptation.

**Key Challenge**: Multimodal tasks require capturing both fine-grained details and global semantics simultaneously, yet PEFT must avoid making large changes to the backbone. If the adaptation module is too weak, it only provides shallow linear corrections; if it is too strong or involves excessive cross-modal interaction, it may interfere with the pre-learned unimodal representations of the original model.

**Goal**: The authors aim to identify an adaptation space that is both lightweight and stable, allowing visual features to be selectively adjusted based on frequency and scale under textual conditions, thereby reducing redundancy, enhancing cross-modal alignment, and maintaining low parameters and FLOPs.

**Key Insight**: The paper uses DCT to analyze the information distribution of visual embeddings, finding that semantic information is more concentrated in low-frequency components: retaining 198/768 low-frequency components achieves a reconstruction cosine similarity of 0.5, while retaining 495 and 626 components exceeds 0.8 and 0.9, respectively. This observation supports compact adaptation in the frequency domain.

**Core Idea**: Both visual and text embeddings are mapped to the frequency domain simultaneously. Text-generated modulation parameters are applied across different spatial scales, and the adjusted visual frequency representations are restored to the spatial domain via IDCT, serving as plug-and-play multimodal adaptation features.

## Method
The basic workflow of FreqAdapter is as follows: CLIP encodes images and text to obtain visual and text embeddings; DCT is applied to both to obtain frequency domain representations; multi-scale aggregation is performed on visual features in the frequency domain; each scale passes through MGFA and MCFA; outputs from all scales are upsampled and fused via averaging; finally, IDCT restores the features to the spatial domain, which are then fed into the subsequent CLIP transformer or LLaVA projector.

### Overall Architecture
Given an image-text pair, CLIP generates visual embeddings $E_v\in\mathbb{R}^{S_v\times D_v}$ and text embeddings $E_t\in\mathbb{R}^{S_t\times D_t}$. FreqAdapter first computes $X_v=DCT(E_v)$ and $X_t=DCT(E_t)$ to obtain compact, frequency-controlled representations. The visual token sequence is reshaped into an $H\times W\times D_v$ grid and downsampled at multiple scales; each scale output is calibrated by a Multi-scale Global Frequency Adapter (MGFA) and then injected with text guidance via a Multi-scale Cross-modal Frequency Adapter (MCFA). The outputs are restored to the original resolution through repeat-interleave. The average of all scale outputs yields $\tilde{X}_v$, which is converted back to adapted visual features $\tilde{E}_v$ via IDCT.

In CLIP retrieval tasks, the adapted visual embeddings enter the final transformer layer, and the `[CLS]` visual feature is used for contrastive learning with the text feature. In LLaVA, FreqAdapter can be inserted directly between the CLIP vision encoder and the LLaVA multimodal projector, delivering more text-relevant visual features to the LLM.

### Key Designs
1. **Frequency Domain over Spatial Domain Adaptation**:
    - Function: Reduces noise in redundant representations, allowing a small number of parameters to focus on adjusting information-dense frequency components.
    - Mechanism: DCT is an orthogonal transform that converts embeddings from spatial/channel forms into frequency coefficients. Empirical findings show low-frequency components retain significant semantics, so frequency domain adaptation is equivalent to more controllable correction of different frequency bands without redundant updates to all patch representations.
    - Design Motivation: Spatial domain adapters are prone to overfitting local noise within limited training steps. The frequency domain better separates low-frequency structures, high-frequency details, and noise, allowing for smoother parameter updates and convergence within a single epoch.

2. **Multi-Scale Adaptation Strategy**:
    - Function: Allows the adaptation module to perceive local details and coarse-grained global structures simultaneously.
    - Mechanism: The visual frequency grid is downsampled by scale $n$ to obtain $X_{v,n}$; each scale has its own MGFA and MCFA, outputting $\tilde{X}_{v,n}=G_n+wC_n$; interleave-repeat restores the original size, and the $N$ scales are averaged. Appendix results show $N=3$ is optimal; excessive receptive fields lose local information.
    - Design Motivation: Image-text matching requires identifying local objects and understanding the overall scene. A single scale either only modifies details or over-smoothes; multi-scale allows selecting different receptive fields based on the semantic emphasis in captions.

3. **Complementary Frequency Modulation of MGFA + MCFA**:
    - Function: MGFA provides stable global calibration of visual frequencies, while MCFA performs fine-grained cross-modal alignment under textual conditions.
    - Mechanism: MGFA is a lightweight bottleneck with two projection layers and ReLU, performing global transformation $G_n=f(X_{v,n})$ on visual frequency features at each scale. MCFA predicts modulation parameters $\gamma,\beta$ from text frequency representations $X_t$ to perform $C_n=\gamma\odot X_{v,n}+\beta$ on visual features. A weight $w$ controls the cross-modal injection intensity.
    - Design Motivation: Relying solely on visual global calibration lacks textual conditioning, while relying solely on text modulation may excessively interfere with visual representations. Adding both modules balances stability and semantic alignment.

### Loss & Training
FreqAdapter is trained using the CLIP contrastive loss on image-text pairs with the backbone mostly frozen, optimizing only the adaptation modules. Retrieval experiments are trained on the COCO 2017 train set for 1 epoch with a batch size of 128 and AdamW learning rate of 0.001; the multimodal weight $w=0.01$ for retrieval and $w=1.0$ for VQA. All CLIP experiments were completed on a single A100-40G. The paper notes that smaller cross-modal weights are generally better for retrieval tasks to prevent excessive text information from interfering with modal-specific feature extraction.

## Key Experimental Results

### Main Results
Image-text retrieval experiments were evaluated on COCO 2017 validation and Flickr30K validation/test, measured by R@1/R@5/R@10 for image-to-text and text-to-image. FreqAdapter consistently outperforms CoOp, MaPLe, CLIP-Adapter, MMA, and LoR-VP across CLIP-B/16, CLIP-L/14, and CLIP-L/14-336 backbones.

| Backbone | Method | COCO I2T R@1 | COCO T2I R@1 | Flickr30K I2T R@1 | Flickr30K T2I R@1 | Description |
|----------|------|--------------|--------------|-------------------|-------------------|------|
| CLIP-B/16 | Original CLIP | 51.82 | 32.65 | 85.30 | 62.28 | Unadapted baseline |
| CLIP-B/16 | CLIP-Adapter | 56.30 | 41.60 | 83.90 | 71.26 | Improvement on COCO, slight drop on Flickr I2T |
| CLIP-B/16 | FreqAdapter | 57.96 | 43.30 | 86.80 | 73.42 | Balanced retrieval improvement |
| CLIP-L/14 | CLIP-Adapter | 60.38 | 43.18 | 87.30 | 75.76 | Strong adapter baseline |
| CLIP-L/14 | FreqAdapter | 61.02 | 44.18 | 87.50 | 75.72 | Stronger on COCO, comparable on Flickr |
| CLIP-L/14-336 | CLIP-Adapter | 60.42 | 44.62 | 90.00 | 77.28 | High-resolution baseline |
| CLIP-L/14-336 | FreqAdapter | 61.42 | 45.23 | 90.90 | 77.60 | Overall best, COCO T2I R@1 improved to 45.23 |

VQA experiments integrated FreqAdapter trained on CLIP into LLaVA 1.5. Results show FreqAdapter is not just a retrieval adapter but also improves image understanding in VQA, particularly significant for 13B LLaVA on MM-Vet.

| Base Model | Method | MM-Vet | LLaVA-Bench | Interpretation |
|------------|------|--------|-------------|------|
| LLaVA 1.5-7B | w/o prompt | 30.9 | 64.3 | Direct response baseline |
| LLaVA 1.5-7B | CLIP-Adapter | 27.1 | 61.8 | Generalization drops after distribution specialization |
| LLaVA 1.5-7B | FreqAdapter | 31.8 | 64.8 | Exceeds 7B baseline on both metrics |
| LLaVA 1.5-13B | w/o prompt | 32.8 | 71.9 | Original 13B baseline |
| LLaVA 1.5-13B | CLIP-Adapter | 32.9 | 64.9 | Significant drop on LLaVA-Bench |
| LLaVA 1.5-13B | FreqAdapter | 37.4 | 72.4 | Significant gain on MM-Vet; exceeds original LLaVA-Bench |
| LLaVA 1.5-13B | API (LLaVA) | 36.6 | 74.8 | FreqAdapter exceeds API(LLaVA) on MM-Vet, but LLaVA-Bench remains lower |

### Ablation Study
Module ablation proves both MGFA and MCFA are effective, with MCFA's text-guided modulation contributing more; the combination is optimal. Evaluation used CLIP-L/14-336 on MSCOCO retrieval.

| MGFA | MCFA | I2T R@1 | I2T R@5 | I2T R@10 | T2I R@1 | T2I R@5 | T2I R@10 | Conclusion |
|------|------|---------|---------|----------|---------|---------|----------|------|
| - | - | 57.34 | 80.38 | 87.64 | 36.08 | 60.70 | 70.66 | Original CLIP-L/14-336 |
| - | ✓ | 58.16 | 81.90 | 88.94 | 42.81 | 67.86 | 77.43 | Text-guided modulation brings large T2I gain |
| ✓ | - | 58.70 | 82.28 | 89.54 | 43.47 | 68.66 | 78.04 | Global frequency calibration is also effective |
| ✓ | ✓ | 61.42 | 83.64 | 90.10 | 45.23 | 70.92 | 80.02 | Modules are complementary; overall best |

Multi-scale ablation shows $N=3$ is most suitable. For $N=1$ (single scale), I2T R@1 is 60.18; $N=2$ improves to 61.32; $N=3$ reaches 61.42 and T2I R@1 45.23; $N=4$ drops, indicating large aggregation windows lose fine details.

| Scales N | I2T R@1 | I2T R@5 | I2T R@10 | T2I R@1 | T2I R@5 | T2I R@10 | Description |
|----------|---------|---------|----------|---------|---------|----------|------|
| 1 | 60.18 | 81.78 | 87.50 | 44.08 | 69.80 | 77.28 | Single scale; insufficient detail/global balance |
| 2 | 61.32 | 82.76 | 88.42 | 44.81 | 70.20 | 79.47 | Multi-scale begins steady improvement |
| 3 | 61.42 | 83.64 | 90.10 | 45.23 | 70.92 | 80.02 | Best configuration |
| 4 | 60.36 | 82.58 | 88.12 | 45.06 | 70.70 | 79.36 | Excessive receptive field leads to info loss |

In terms of computation, FreqAdapter's parameters and FLOPs are very lightweight. It has significantly fewer parameters than MaPLe, and its GFLOPs are nearly identical to CLIP-Adapter and MMA.

| Method | Params | Param % | GFLOPs | Description |
|------|--------|----------|--------|------|
| CLIP | - | - | 362.5 | Backbone |
| CoOp | 16.4k | 0.003% | 370.8 | Fewest params but higher computation |
| MaPLe | 798.7k | 0.19% | 362.9 | Most parameters |
| CLIP-Adapter | 524.3k | 0.12% | 362.5 | Common adapter baseline|
| MMA | 118.7k | 0.03% | 362.7 | More lightweight |
| FreqAdapter | 476.4k | 0.11% | 362.6 | Fewer params than CLIP-Adapter; negligible extra overhead |

### Key Findings
- Frequency domain adaptation is more stable than spatial domain adaptation. Comparison with a "SpatialAdapter" shows spatial methods suffer from overfitting and loss rebound in the late first epoch, while FreqAdapter remains smoother on COCO and Flickr30K.
- MCFA is the key to performance jumps, especially for text-to-image retrieval; MGFA further stabilizes adjustments across frequency bands.
- Moderate cross-modal interaction is essential. In retrieval, a smaller $w=0.01$ is better; excessive text injection interferes with modal-specific features.
- FreqAdapter can migrate to LLaVA, though gains do not lead all API baselines, indicating that frequency adapters for complex generative VQA are still limited by the backbone and projector.

## Highlights & Insights
- This paper provides an interesting "adaptation space" perspective for PEFT. While others focus on structural differences (prompt vs. adapter vs. LoRA), this emphasizes switching to the frequency domain to update small modules after reducing redundancy.
- The combination of DCT and multi-scale is natural. The frequency domain handles information decomposition by band, while multi-scale aggregates information by spatial receptive field, corresponding to "frequency" and "region" dimensions in visual semantics.
- MCFA's text-conditioned modulation is highly reusable. For any CLIP-like VLM, text frequency features can generate $\gamma, \beta$ to tune visual frequency representations given visual/text tokens.
- Frequency and spatial domain adaptations are not mutually exclusive. Appendix discussions on fusing FreqAdapter with CLIP-Adapter suggest the two are complementary, leaving room for future hybrid PEFT.

## Limitations & Future Work
- Theoretical explanation remains empirical. While information concentration and training curves demonstrate frequency stability, there is no rigorous proof of which frequencies map to specific semantics or why DCT is optimal for specific tasks.
- Experiments are centered on CLIP-based architectures and LLaVA 1.5; model scale is limited. Performance on larger VLMs, different vision encoders, or end-to-end multimodal LLMs requires re-validation.
- The method relies on text guidance; in scenarios with poor-quality text prompts or only images, MCFA modulation might introduce incorrect semantic bias.
- Many gains are 1-2 percentage points; while stable, they are not overwhelming. Future work requires larger benchmarks, more seeds, and statistical significance analysis.
- DCT is a fixed basis and may not be optimal for all learned representations. Future work could explore learnable frequency bases, wavelets, multi-resolution spectral decomposition, or dynamic band selection.

## Related Work & Insights
- **vs CLIP-Adapter**: CLIP-Adapter adds adapters to spatial/feature outputs. Ours moves embeddings to the frequency domain for global calibration and text modulation, yielding more stable generalization.
- **vs CoOp / MaPLe**: Prompt tuning modifies input prompts or cross-modal prompts, whereas FreqAdapter modifies visual frequency representations, suitable for tasks requiring fine-grained visual feature adjustment.
- **vs LoRA / LoR-VP**: LoRA-type methods change model weights or visual prompts through low-rank parameters; FreqAdapter uses an external frequency module for plug-and-play while keeping the backbone frozen.
- **vs Frequency Vision Methods**: SpectFormer, VFPT, and SFMFusion show frequency domains improve visual representation. Ours extends frequency processing to image-text cross-modal adaptation for CLIP/LLaVA.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Frequency domain concepts are established in vision, but the application to text-guided multi-scale VLM adapters is innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers retrieval, VQA, ablations, scale, computation, and spatial/frequency comparisons, though model variety is limited.
- Writing Quality: ⭐⭐⭐⭐☆ Clear methodology and comprehensive tables; theoretical depth could be improved.
- Value: ⭐⭐⭐⭐☆ Insightful for PEFT and VLM adaptation, especially for low-cost enhancement of CLIP-like model perception and retrieval.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] FlashCache: Frequency-Domain-Guided Outlier-KV-Aware Multimodal KV Cache Compression](../../CVPR2026/multimodal_vlm/flashcache_frequency_kv_cache_compression.md)
- [\[ACL 2026\] TRACE: Unleashing Spatial Reasoning in Multimodal Large Language Models via Textual Representation Guided Reasoning](unleashing_spatial_reasoning_in_multimodal_large_language_models_via_textual_rep.md)
- [\[ACL 2026\] TEMA: Anchor the Image, Follow the Text for Multi-Modification Composed Image Retrieval](tema_anchor_the_image_follow_the_text_for_multi-modification_composed_image_retr.md)
- [\[ACL 2026\] AFMRL: Attribute-Enhanced Fine-Grained Multi-Modal Representation Learning in E-commerce](afmrl_attribute-enhanced_fine-grained_multi-modal_representation_learning_in_e-c.md)
- [\[CVPR 2026\] Decoupling Stability and Plasticity for Multi-Modal Test-Time Adaptation](../../CVPR2026/multimodal_vlm/decoupling_stability_and_plasticity_for_multi-modal_test-time_adaptation.md)

</div>

<!-- RELATED:END -->
