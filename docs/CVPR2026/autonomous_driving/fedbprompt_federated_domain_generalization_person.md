---
title: >-
  [Paper Note] FedBPrompt: Federated Domain Generalization Person Re-Identification via Body Distribution Aware Visual Prompts
description: >-
  [CVPR 2026][Autonomous Driving][Federated Learning] FedBPrompt introduces learnable visual prompts partitioned into body part alignment prompts (with constrained local attention to handle viewpoint misalignment) and holistic full-body prompts (to suppress background interference), coupled with a prompt-only federated fine-tuning strategy that transmits only prompt parameters (~0.46M vs. ~86M for the full model), achieving consistent improvements on FedDG-ReID benchmarks.
tags:
  - CVPR 2026
  - Autonomous Driving
  - Federated Learning
  - Person Re-Identification
  - Visual Prompts
  - Domain Generalization
  - Communication Efficiency
  - Body Part Alignment
date: 2026-05-08
content_hash: f0869e61c515b7f6
---

# FedBPrompt: Federated Domain Generalization Person Re-Identification via Body Distribution Aware Visual Prompts

**Conference**: CVPR 2026
**arXiv**: [2603.12912](https://arxiv.org/abs/2603.12912)
**Code**: [GitHub](https://github.com/leavlong/FedBPrompt)
**Area**: Autonomous Driving
**Keywords**: Federated Learning, Person Re-Identification, Visual Prompts, Domain Generalization, Communication Efficiency, Body Part Alignment

## TL;DR
FedBPrompt introduces learnable visual prompts partitioned into body part alignment prompts (with constrained local attention to handle viewpoint misalignment) and holistic full-body prompts (to suppress background interference), coupled with a prompt-only federated fine-tuning strategy that transmits only prompt parameters (~0.46M vs. ~86M for the full model), achieving consistent improvements on FedDG-ReID benchmarks.

## Background & Motivation
Federated domain generalization person re-identification (FedDG-ReID) requires multiple clients to collaboratively train a global model that generalizes to unseen target domains without sharing raw data. While ViT has become the dominant backbone for ReID, its global attention mechanism faces two challenges that are amplified by client heterogeneity in FedDG-ReID settings: (1) **Background-induced attention distraction**—clients deployed in different environments exhibit highly heterogeneous background distributions, causing ViT to be misled by dominant yet irrelevant background cues, resulting in false matches across identities; (2) **Viewpoint-induced body misalignment**—significant camera angle variation across clients leads to severe spatial misalignment of body parts for the same pedestrian across viewpoints, sharply degrading feature similarity. Existing FedDG-ReID methods largely focus on data augmentation-level diversity and do not directly address these two model-level challenges.

## Core Problem
How to simultaneously address ViT attention distraction by background and cross-client viewpoint misalignment in a federated distributed setting, while maintaining extremely low communication overhead?

## Method

### Overall Architecture
FedBPrompt comprises two core components: (1) a Body distribution-Aware visual Prompt Mechanism (BAPM), which guides ViT attention to focus on pedestrians and align body parts during full-parameter training; and (2) a Prompt-based Fine-Tuning Strategy (PFTS), which freezes the backbone and updates only prompt parameters, enabling communication-efficient federated training.

### Key Designs
1. **BAPM – Prompt Partitioning**: $m$ learnable prompt tokens are injected into each Transformer layer (with $m=50$ in experiments) and divided into two functional groups: (a) **Body part alignment prompts** (15 tokens), equally split into $P_{\text{upper}}$/$P_{\text{mid}}$/$P_{\text{lower}}$ (5 each), corresponding to the upper, middle, and lower body, respectively; (b) **Holistic full-body prompts** (35 tokens) $P_{\text{full}}$, capturing the overall pedestrian appearance.
2. **BAPM – Constrained Attention Mask**: Functional separation is enforced via a structured attention mask $M$. Body part prompts can only interact with image patch tokens in their corresponding spatial region (e.g., $P_{\text{upper}}$ attends only to patches in the upper half of the image), with $M_{ij} = -\infty$ for mismatched prompt–patch pairs. Critically, all prompts are free to interact with each other ($M_{ij} = 0$), enabling part prompts to model structural relationships and full-body prompts to integrate part-level signals into a global context. Spatial regions are defined with overlapping patch index sets (the middle region covers $1/4$ to $3/4$ of the image height), ensuring smooth transitions across body parts.
3. **PFTS – Communication-Efficient Fine-Tuning**: A pre-trained ReID model without prompts is first distributed to all clients. Each client then freezes the backbone, inserts randomly initialized prompt parameters, and trains only the prompts. In each round, only the prompt parameters (~0.46M) are uploaded, aggregated via FedAvg, and redistributed, reducing communication volume to approximately 1% of the full model size.

### Loss & Training
- Standard ReID loss $\mathcal{L}_{\text{ReID}}$ (cross-entropy + triplet loss) is applied to the global model $g(x; \Theta_b, \Theta_p)$
- Under PFTS, the optimization objective is $\min_{\Theta_p} \sum \mathcal{L}_{\text{ReID}}$, with backbone parameters $\Theta_b$ frozen
- Federated aggregation uses weighted averaging: $\Theta_p^{t+1} = \sum \frac{|D_k|}{\sum |D_j|} \cdot \Theta_{p,k}^{t+1}$
- Backbone is ViT-B/16, built upon the SSCU framework

## Key Experimental Results

**Protocol-1 (Leave-One-Out: three source domains for federated training, one target domain for evaluation)**:

| Method | MS+C2+C3→M | MS+C2+M→C3 | C2+C3+M→MS | Avg. mAP/R-1 |
|--------|------------|------------|------------|--------------|
| SSCU (baseline) | 46.3/69.6 | 33.7/33.4 | 20.0/43.7 | 33.3/48.9 |
| SSCU+PFTS | 48.9/72.4 | 35.5/35.8 | 21.3/46.0 | 35.2/51.4 |
| SSCU+BAPM | **49.1/73.4** | **37.4/38.4** | **23.4/49.5** | **36.6/53.8** |
| FedProx+BAPM | 47.3/70.9 | 33.7/33.7 | 17.7/41.2 | 32.9/48.6 |
| DACS+BAPM | 49.7/74.3 | 34.6/34.8 | 21.9/48.5 | 35.4/52.5 |

- BAPM yields average gains of +3.3% mAP / +4.9% R-1 over SSCU; gains over the weaker FedProx baseline reach +13.9% mAP / +13.3% R-1
- PFTS transmits only ~1% of parameters and achieves significant improvements within a few aggregation rounds

### Ablation Study
- **Full-body prompts only vs. part prompts only vs. full BAPM**: The full combination (BAPM) consistently performs best. Using SSCU as the baseline: full-body prompts yield mAP=48.4, part prompts yield 47.7, and BAPM achieves 49.1. However, on the C2+C3+M→MS task, part prompts (22.7) outperform full-body prompts (22.9) and the baseline (20.0), indicating that part alignment is particularly critical in scenarios with large viewpoint variation.
- **Attention visualization**: The baseline ViT disperses attention across backgrounds; under BAPM, $P_{\text{upper}}$/$P_{\text{mid}}$/$P_{\text{lower}}$ precisely localize their corresponding body regions, while $P_{\text{full}}$ covers the full-body silhouette.
- **Insertion AUC**: BAPM > VPs > SSCU baseline (0.7559 > 0.7103 > 0.6160, using class token), confirming superior attention focus quality.
- **t-SNE feature space**: BAPM produces more compact intra-domain clusters and clearer inter-domain separation.

## Highlights & Insights
- Prompt partitioning combined with constrained attention masks is an elegant way to inject spatial priors into ViT—without modifying the backbone architecture—while enforcing part-level feature alignment.
- The design choice of allowing all prompts to freely interact with each other is crucial: it enables part prompts to model structural relationships rather than operating in isolation, and allows full-body prompts to integrate part-level signals.
- Freezing the backbone and transmitting only prompts (~0.46M vs. ~86M) achieves a drastic reduction in federated communication cost, with measurable gains appearing within the first few aggregation rounds.
- The conceptual leap from VPT to BAPM lies in **structured prompt design**—rather than simply prepending prompt tokens, each prompt is assigned a functional role with spatially constrained attention.

## Limitations & Future Work
- The body region partitioning is coarse (three equal segments: upper/middle/lower) and relies on the assumption that pedestrian images are already cropped and roughly aligned.
- The fixed spatial partitioning may fail under extreme occlusion (e.g., only the lower body visible) or unusual poses (e.g., crouching, bending).
- The prompt count of 50 is fixed; adaptive prompt sizing or dynamic pruning is not explored.
- Protocol-1 uses only 4 ReID datasets; validation on larger-scale benchmarks (e.g., LaST, PRCC) or cross-modal scenarios is absent.
- Comparisons with other parameter-efficient methods such as LoRA and adapters are missing.

## Related Work & Insights
- vs. **VPT (Jia et al., ECCV 2022)**: VPT prepends learnable tokens to the ViT input sequence without spatial constraints; BAPM introduces functional partitioning and enforces spatial correspondence via attention masks, offering stronger inductive bias.
- vs. **PromptFL (Guo et al., TMC 2023)**: PromptFL leverages prompts for communication-efficient federated learning but focuses on the text domain; FedBPrompt is the first to introduce structured visual prompts into federated ReID.
- vs. **SSCU (MM 2025)**: The current FedDG-ReID state of the art; FedBPrompt achieves an average gain of +3.3% mAP via BAPM, while PFTS reduces communication overhead by 99%.
- vs. **DACS (AAAI 2024)**: DACS improves generalization through style augmentation but does not directly address attention distraction or body misalignment; BAPM and DACS are orthogonal and complementary, with further gains when combined.

## Related Papers

- [\[CVPR 2026\] Open-Vocabulary Domain Generalization in Urban-Scene Segmentation](open-vocabulary_domain_generalization_in_urban-scene_segmentation.md)
- [\[CVPR 2026\] F3DGS: Federated 3D Gaussian Splatting for Decentralized Multi-Agent World Modeling](f3dgs_federated_3d_gaussian_splatting_for_decentralized_multi-agent_world_modeli.md)
- [\[CVPR 2026\] CCF: Complementary Collaborative Fusion for Domain Generalized Multi-Modal 3D Object Detection](ccf_complementary_collaborative_fusion_for_domain_generalized_multi-modal_3d_obj.md)
- [\[AAAI 2026\] Hierarchical Prompt Learning for Image- and Text-Based Person Re-Identification](../../AAAI2026/autonomous_driving/hierarchical_prompt_learning_for_image-_and_text-based_person_re-identification.md)
- [\[CVPR 2026\] Neural Distribution Prior for LiDAR Out-of-Distribution Detection](neural_distribution_prior_for_lidar_ood_detection.md)

## Rating
- **Novelty**: 7/10 — Prompt partitioning with constrained attention masks is a novel and conceptually clean contribution to federated ReID.
- **Experimental Thoroughness**: 8/10 — Two protocols, multiple baselines, complete ablations, and rich visualizations; large-scale and cross-modal validation is lacking.
- **Writing Quality**: 7/10 — Clear structure with well-presented equations, algorithms, and visualizations.
- **Value**: 7/10 — Plug-and-play improvements across multiple FedDG-ReID methods with minimal communication cost; strong practical value.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Open-Vocabulary Domain Generalization in Urban-Scene Segmentation](open-vocabulary_domain_generalization_in_urban-scene_segmentation.md)
- [\[CVPR 2026\] F3DGS: Federated 3D Gaussian Splatting for Decentralized Multi-Agent World Modeling](f3dgs_federated_3d_gaussian_splatting_for_decentralized_multi-agent_world_modeli.md)
- [\[CVPR 2026\] CCF: Complementary Collaborative Fusion for Domain Generalized Multi-Modal 3D Object Detection](ccf_complementary_collaborative_fusion_for_domain_generalized_multi-modal_3d_obj.md)
- [\[CVPR 2026\] CoLC: Communication-Efficient Collaborative Perception with LiDAR Completion](colc_communication-efficient_collaborative_perception_with_lidar_completion.md)
- [\[AAAI 2026\] When Person Re-Identification Meets Event Camera: A Benchmark Dataset and An Attribute-guided Re-Identification Framework](../../AAAI2026/autonomous_driving/when_person_re-identification_meets_event_camera_a_benchmark_dataset_and_an_attr.md)

<!-- RELATED:END -->
