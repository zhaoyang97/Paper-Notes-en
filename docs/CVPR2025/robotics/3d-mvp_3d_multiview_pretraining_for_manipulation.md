---
title: >-
  [Paper Note] 3D-MVP: 3D Multiview Pretraining for Robotic Manipulation
description: >-
  [CVPR 2025][Robotics][Masked Autoencoder] This paper proposes 3D-MVP, which extends Masked Autoencoder pretraining from 2D to a 3D multiview setting. By pretraining the multiview Transformer encoder of RVT on 200K 3D objects from Objaverse, downstream fine-tuning improves the average success rate on RLBench from 62.9% to 67.5% and significantly enhances robustness against environmental variations (such as texture, size, and lighting) on COLOSSEUM.
tags:
  - "CVPR 2025"
  - "Robotics"
  - "Masked Autoencoder"
  - "Multiview Pretraining"
  - "Robotic Manipulation"
  - "RVT"
  - "Objaverse"
date: 2026-05-08
content_hash: ab7da4a3a4cb8c64
---

# 3D-MVP: 3D Multiview Pretraining for Robotic Manipulation

**Conference**: CVPR 2025  
**arXiv**: [2406.18158](https://arxiv.org/abs/2406.18158)  
**Code**: [https://jasonqsy.github.io/3DMVP](https://jasonqsy.github.io/3DMVP)  
**Area**: Robotics / Self-Supervised Learning / 3D Vision Pretraining  
**Keywords**: Masked Autoencoder, Multiview Pretraining, Robotic Manipulation, RVT, Objaverse  

## TL;DR
This paper proposes 3D-MVP, which extends Masked Autoencoder pretraining from 2D to a 3D multiview setting. By pretraining the multiview Transformer encoder of RVT on 200K 3D objects from Objaverse, downstream fine-tuning improves the average success rate on RLBench from 62.9% to 67.5% and significantly enhances robustness against environmental variations (such as texture, size, and lighting) on COLOSSEUM.

## Background & Motivation
While visual pretraining (e.g., MAE on Ego4D) has proven effective for robotics, existing approaches only pretrain on 2D images. However, state-of-the-art manipulation methods (e.g., RVT, PerAct, Act3D) construct explicit 3D representations to make decisions. Features from 2D pretraining cannot be directly transferred to these 3D architectures because they lack standard 2D vision encoders. On the other hand, although large-scale 3D datasets (such as Objaverse with over 800K objects) are available, they lack robotic annotations. Utilizing these large-scale 3D datasets to pretrain 3D manipulation policies remains an unexplored gap.

## Core Problem
How to extend MAE pretraining from 2D to 3D so that it is compatible with robotic manipulation methods based on multiview 3D representations (e.g., RVT), thereby leveraging large-scale 3D object datasets to improve the performance and generalization of manipulation policies?

## Method

### Overall Architecture
A two-stage pipeline:
1. **Pretraining**: Split the multiview Transformer of RVT into a vision encoder $\mathcal{E}$ and an action decoder $\mathcal{D}$. Render RGBD images from 5 orthogonal virtual views of 3D objects from Objaverse, randomly mask 75% of the tokens, and train the encoder to reconstruct the original multiview images.
2. **Fine-tuning**: Discard the MAE decoder, connect the pretrained encoder $\mathcal{E}$ with the action decoder $\mathcal{D}$, and perform end-to-end fine-tuning on manipulation demonstration data.

### Key Designs
1. **Multiview MAE Pretraining**: The core idea is to force the encoder to reconstruct masked patches through cross-view information fusion, thereby compelling the model to understand 3D spatial relationships. Five orthogonal virtual cameras (top, left, right, front, back) generate 10-channel virtual images (RGB + Depth + world coordinates + camera coordinates), which are tokenized into $5N$ patches and fed together into the Transformer.

2. **Encoder-Decoder Split**: The original RVT is end-to-end, taking language instructions as input and outputting actions. To adapt to pretraining on unannotated 3D datasets, RVT is split into a vision-only encoder (8-layer Transformer) and a lightweight MAE decoder (2-layer). Language or action annotations are not required during pretraining.

3. **Leveraging Large-Scale 3D Datasets**: 200K high-quality 3D models are sampled from Objaverse and directly rendered into the orthogonal virtual view format used by RVT. The pretraining data is completely independent of downstream tasks, focusing purely on learning visual understanding of 3D objects.

4. **RGB-only Masking Strategy**: Instead of masking all channels, only the RGB channels are masked (preserving Depth and coordinate channels). Experiments show that masking all channels makes the pretraining task overly difficult, which actually degrades downstream performance (similar to the case where the mask ratio is >80% in MAE).

### Loss & Training
**Pretraining**: Pixel-level L2 reconstruction loss:
$$\mathcal{L}_{recon} = \frac{1}{5WH}\sum_{i=1}^{5}\sum_{p}\|I_i(p) - \tilde{I}_i(p)\|^2$$
- 8×V100, 15 epochs, AdamW (lr=1e-4, wd=0.01), batch size 3, mask ratio 0.75

**Fine-tuning**: Standard RVT training setup:
- 8×V100, 15 epochs, Lamb optimizer (lr=1e-4), batch size 3, 2000 warmup steps

## Key Experimental Results

### RLBench (Average Success Rate on 18 Tasks)

| Method | Average Success Rate |
|------|-----------|
| Image-BC (CNN) | 1.3% |
| PerAct | 49.4% |
| RVT (Scratch) | 62.9% |
| **3D-MVP** | **67.5%** (+4.6%) |

Significant improvements are observed in several tasks: Insert Peg 11.2 -> 20.0, Put in Cupboard 49.6 -> 60.0, Screw Bulb 48.0 -> 60.0.

### More RLBench Single-Task Comparisons

| Task | 3D-MVP | RVT Baseline | Difference |
|------|--------|---------|------|
| Stack Blocks | 28.8 | 24.8 | +4.0 |
| Place Wine | 68.8 | 62.4 | +6.4 |
| Open Drawer | 76.8 | 72.0 | +4.8 |
| Slide Block | 81.6 | 74.0 | +7.6 |
| Meat off Grill | 93.6 | 96.4 | -2.8 |

Note that 3D-MVP might show slight performance drops in simple tasks that are already highly saturated (e.g., Meat off Grill, Close Jar). The improvements are primarily concentrated on tasks of moderate difficulty.

### COLOSSEUM (Generalization Robustness)
3D-MVP outperforms RVT and 2D pretraining methods (MVP, R3M) under most environmental perturbations. It shows significant improvements especially under changes in receptacle texture/size, manipulated object size, lighting color, and tabletop color. Under the more challenging "Receptacle Color" and "Size Distraction" perturbations, 3D-MVP improves over the RVT baseline by 15% and 12%, respectively. This indicates that 3D pretraining not only enhances performance but, more importantly, strengthens robustness to visual distractors. In contrast, 2D pretraining methods underperform even the from-scratch RVT on COLOSSEUM, suggesting that 2D features cannot be effectively transferred to 3D manipulation architectures. The failure of 2D methods likely stems from the fact that RVT's multiview Transformer projects inputs into virtual orthogonal views for 3D feature fusion, whereas features from 2D pretraining lack cross-view consistency.

### Ablation Study
- **Architecture vs. Pretraining**: Directly fine-tuning the split architecture without pretraining yields 62.9% (same as RVT), proving that the performance gain comes from pretraining rather than architectural changes.
- **Objaverse (200K) vs. Objaverse (18K)**: 67.6% vs. 65.3%, showing that larger data scale yields better results.
- **Objaverse vs. 3D-FRONT**: 67.6% vs. 63.6%, demonstrating that object-level datasets perform better than room-level ones, where diversity and scale are key.
- **Pretraining on RLBench**: Achieves 67.5% but generalizes poorly on COLOSSEUM—pretraining on the target domain leads to overfitting.
- **RGB Masking vs. All-channel Masking**: 67.6% vs. 64.4%, showing that masking only RGB channels performs better.

## Highlights & Insights
- **First to extend MAE pretraining to a 3D multiview setting for robotic manipulation**—bridging the gap between 2D pretraining and 3D manipulation methods.
- **Plug-and-play design**: The encoder-decoder split decouples pretraining from downstream tasks, allowing the pretrained encoder to directly connect with RVT's action decoder.
- **Extensive ablation studies**: Systematically investigates factors such as dataset selection, scale, and masking strategies, providing useful guidelines for future research.
- **Generalization validation**: Evaluation on the COLOSSEUM benchmark demonstrates that 3D pretraining not only boosts performance but also enhances robustness to environmental variations.

## Limitations & Future Work
- Fixed 5 orthogonal views make the method unable to handle occlusions and arbitrary views, restricting its applicability to non-orthogonal camera configurations.
- Assumes quasi-static dynamics, failing to handle dynamic interactions between the robot and the environment.
- Still requires a small amount of annotated demonstration data for fine-tuning, and zero-shot transfer to new tasks is not achieved.
- Only evaluated in simulation (RLBench/COLOSSEUM) with no real-robot experiments, leaving the sim-to-real gap unmeasured.
- Pretraining was only conducted on a subset of Objaverse (200K); scaling up to larger datasets (e.g., Objaverse-XL with over 10M objects) might yield further improvements.
- Pretraining for 15 epochs may be insufficient; whether longer pretraining continues to improve performance remains unexplored.

## Related Work & Insights
- **MVP/R3M** (2D pretraining): Performs significantly worse than 3D methods on COLOSSEUM. 2D pretrained features are incompatible with 3D manipulation strategies like RVT.
- **RVT**: The baseline architecture for 3D-MVP. RVT trained from scratch achieves 62.9%, while 3D-MVP pretraining boosts it to 67.5% (+4.6%), with improvements mainly in medium-difficulty tasks.
- **PerAct**: Uses voxels + Perceiver; 3D-MVP outperforms it in almost all tasks.
- **GNFactor**: Relies on pretrained VLMs to inject semantics, whereas 3D-MVP directly learns features from 3D objects.

### Insights & Connections
- The paradigm of 3D pretraining followed by downstream fine-tuning can be extended to other 3D robotic tasks such as navigation and grasp planning.
- Demonstrates the value of large-scale 3D object datasets (Objaverse) for the robotics community, extending beyond just 3D generation.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Extending MAE from 2D to 3D multiview is a natural yet effective idea; the core contribution is systematic validation rather than methodological novelty.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Involves two benchmarks and extensive ablations, but lacks real-robot experiments.
- **Writing Quality**: ⭐⭐⭐⭐ Clear, systematic, and well-organized ablation studies.
- **Value**: ⭐⭐⭐⭐ Points the direction for 3D robotic vision pretraining, though the lack of real-world validation reduces immediate practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Lift3D Foundation Policy: Lifting 2D Large-Scale Pretrained Models for Robust 3D Robotic Manipulation](lift3d_policy_lifting_2d_foundation_models_for_robust_3d_robotic_manipulation.md)
- [\[CVPR 2026\] DiffuView: Multi-View Diffusion Pretraining for 3D-Aware Robotic Manipulation](../../CVPR2026/robotics/diffuview_multi-view_diffusion_pretraining_for_3d_aware_robotic_manipulation.md)
- [\[CVPR 2025\] RoboGround: Robotic Manipulation with Grounded Vision-Language Priors](roboground_robotic_manipulation_with_grounded_vision-language_priors.md)
- [\[CVPR 2025\] Mitigating the Human-Robot Domain Discrepancy in Visual Pre-training for Robotic Manipulation](mitigating_the_human-robot_domain_discrepancy_in_visual_pre-training_for_robotic.md)
- [\[CVPR 2025\] g3D-LF: Generalizable 3D-Language Feature Fields for Embodied Tasks](g3d-lf_generalizable_3d-language_feature_fields_for_embodied_tasks.md)

</div>

<!-- RELATED:END -->
