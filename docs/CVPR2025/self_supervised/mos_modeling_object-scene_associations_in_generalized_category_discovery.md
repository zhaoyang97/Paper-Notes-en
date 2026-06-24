---
title: >-
  [Paper Note] MOS: Modeling Object-Scene Associations in Generalized Category Discovery
description: >-
  [CVPR 2025][Self-Supervised Learning][Generalized Category Discovery] Challenges the traditional view in GCD that "scene information is noise," revealing that scenes are misunderstood as noise due to the "ambiguity challenge" (conflicts in base/novel relations between objects and scenes). It proposes the MOS framework, which effectively utilizes scene information through a dual-branch network and an MLP scene-aware module, achieving an average improvement of $4\%$ on fine-gra…
tags:
  - "CVPR 2025"
  - "Self-Supervised Learning"
  - "Generalized Category Discovery"
  - "scene information"
  - "object-scene association"
  - "dual-branch network"
  - "fine-grained classification"
date: 2026-05-08
content_hash: 47c25a79e423844f
---

# MOS: Modeling Object-Scene Associations in Generalized Category Discovery

**Conference**: CVPR 2025  
**arXiv**: [2503.12035](https://arxiv.org/abs/2503.12035)  
**Code**: [GitHub](https://github.com/JethroPeng/MOS)  
**Area**: Others (Generalized Category Discovery)  
**Keywords**: Generalized Category Discovery, scene information, object-scene association, dual-branch network, fine-grained classification

## TL;DR

Challenges the traditional view in GCD that "scene information is noise," revealing that scenes are misunderstood as noise due to the "ambiguity challenge" (conflicts in base/novel relations between objects and scenes). It proposes the MOS framework, which effectively utilizes scene information through a dual-branch network and an MLP scene-aware module, achieving an average improvement of $4\%$ on fine-grained GCD.

## Background & Motivation

Generalized Category Discovery (GCD) is a semi-supervised classification task: given a labeled dataset (containing base classes) and an unlabeled dataset (containing base and novel classes), the goal is to classify the unlabeled data. The core challenge of GCD lies in utilizing knowledge of known classes to discover unknown novel classes.

**Limitations of Prior Work**: Existing GCD methods generally assume that scene information (such as forest, ocean, sky, and other backgrounds) is noise that interferes with model generalization, and thus minimize its impact using various techniques. Empirical evidence supporting this view is that GCD performance usually improves after removing the scene background.

**Ours**: The authors conduct a fine-grained analysis of scene information on the CUB bird dataset and find that the performance improvement only occurs in the subset where **the object-scene base/novel relationship conflicts**:
- Novel birds appearing in known scenes $\rightarrow$ easily misclassified as known classes (misguided by scene features)
- Known birds appearing in novel scenes $\rightarrow$ easily mistaken for novel classes (interfered by scene features)

In contrast, in the relationship-consistent subset (e.g., novel bird + novel scene), removing scenes actually **slightly decreases** performance. This indicates that scene information is inherently a valuable prior, and the issue lies in the "ambiguity challenge" rather than the scene background itself.

## Method

### Overall Architecture

MOS adopts a dual-branch design: (1) **Original Image Branch**: processes the complete image containing both scene and object; (2) **Object Image Branch**: processes the foreground object extracted via saliency segmentation (with the background filled with mean pixels). The two branches share a backbone (ViT-B/16) and a scene-aware module but compute losses separately. Only the output of the object branch is used during inference.

### Key Designs

**1. Object-Scene Decoupling**

A zero-shot saliency segmentation model, IS-Net, is used to separate the foreground object from the scene. It generates a saliency mask $M$, keeping the foreground pixels and replacing the background pixels with the image mean $\mu$. Mean filling instead of zero filling is used to minimize the domain shift caused by missing scenes.

**2. Scene-Aware Module (SA Module)**

The core is a simple MLP interaction design. Taking object/image features $v_i$ ($i \in \{o, x\}$) and scene features $v_s$ (practically replaced by original image features) as input, they are first concatenated, then L2-normalized, and processed through an MLP: $IM_i = \text{MLP}(\text{normalize}(v_i \oplus v_s))$.

Key design details:
- **Scene features are replaced by original image features**: Because scenes are often located at the edges of the image, they are difficult to extract independently by DINO; given the object feature, the conditional entropy of the original image feature equals that of the scene feature.
- **The teacher network outputs scene features** (detached, no gradient propagation): This addresses the issue of the feature space changing too rapidly when $v_s = v_x$ and the over-optimization of $v_x$.
- To ensure a fair comparison, one MLP layer is removed from the DINO header to compensate for the newly added module.

**3. Dual-Branch Joint Training**

Each branch independently computes four types of losses: supervised contrastive learning loss + classification cross-entropy loss for labeled data; unsupervised contrastive learning loss + teacher-student cross-entropy loss for unlabeled data. The total loss is the weighted sum of the losses from the two branches $L = \lambda_1 \cdot L_{\text{origin}} + \lambda_2 \cdot L_{\text{object}}$, where $\lambda_1 = \lambda_2 = 1$ is used in practice.

### Loss & Training

Inheriting the loss design of SimGCD, each branch uses: $L_{\text{branch}} = (1-\lambda)(L_{\text{un}}^{\text{nce}} + L_{\text{un}}^{\text{cls}}) + \lambda(L_{\text{sup}}^{\text{nce}} + L_{\text{sup}}^{\text{cls}})$, where $\lambda = 0.35$ balances labeled and unlabeled data. $L_{\text{un}}^{\text{cls}}$ uses an EMA teacher network to generate pseudo-labels. The temperature parameters are $\tau_u = 0.07$, $\tau_c = 1.0$, $\tau_t = 0.07$, $\tau_s = 0.1$.

## Key Experimental Results

### Main Results: SSB Fine-Grained Dataset (Table 2)

| Method | CUB All↑ | Stanford Cars All↑ | FGVC All↑ | Average All↑ |
|------|:---:|:---:|:---:|:---:|
| SimGCD | 60.3 | 53.8 | 54.2 | 56.1 |
| SPTNet | 65.8 | 59.0 | 59.3 | 61.4 |
| InfoSieve | 69.4 | 55.7 | 56.3 | 60.5 |
| LeGCD | 63.8 | 57.3 | 55.0 | 58.7 |
| **MOS** | **69.6** | **64.6** | **61.1** | **65.1** |

MOS achieves an average accuracy of $65.1\%$ across the three SSB datasets, outperforming SimGCD by $9\%$ and the strongest baseline SPTNet by $3.7\%$. The performance gain is particularly significant on Stanford Cars ($+5.6\%$), indicating that scene information is also effective for car classification.

### Ablation Study (Table 4, CUB Dataset)

| Configuration | All↑ | Base↑ | Novel↑ |
|------|:---:|:---:|:---:|
| Original SimGCD | 61.50 | 65.70 | 59.40 |
| + Object Branch | 63.06 | 64.71 | 62.23 |
| + MOS Dual-Branch | 67.86 | 70.78 | 66.40 |
| + SA Module | **69.57** | **72.31** | **68.20** |

- Introducing the object image branch yields a $1.5\%$ improvement, demonstrating the basic value of training without scenes.
- Dual-branch joint training significantly improves performance by $4.8\%$ (with the Novel class increasing from $62.2\%$ to $66.4\%$), showing that the model learns scene information through contrastive differences.
- The SA Module contributes an additional $1.7\%$ (with the Novel class increasing from $66.4\%$ to $68.2\%$).
- **Weight sharing has the greatest impact on the Novel class** (Fig. 5 right), showing that effectively extracting scene features is crucial for inferring novel classes.

### Key Findings

- Accuracy on Oxford-IIIT Pet reaches $93.2\%$, outperforming SimGCD by $4.4\%$.
- The L1 deviation continuously increases during training (Fig. 6), demonstrating that the network indeed learns to differentiate between global features and object features.
- IS-Net segmentation does not affect the overall efficacy even with failed cases (Fig. 7), showing that the framework is robust to segmentation quality.
- Performance remains stable (within $\pm 1\%$) when parameters $\lambda_1, \lambda_2$ vary around 1, indicating that the method is insensitive to hyperparameters.

## Highlights & Insights

1. **Courage and Evidence to Challenge Traditional Assumptions**: Through a meticulously designed "four-subset" experiment, it clearly reveals that the root cause of scene information being misunderstood as noise is the ambiguity challenge, rather than the scenes themselves being valueless.
2. **The "Prior" Value of Scene Information**: In resource-constrained GCD settings, scenes can provide powerful class inference clues (e.g., "a bird on the water is more likely to be a waterfowl").
3. **Efficacy of a Minimalist MLP**: The scene-aware module is merely an MLP. Compared to the baseline, it incurs almost no additional parameter overhead but yields significant improvements, proving that the key lies in the design rather than complexity.

## Limitations & Future Work

- Scene annotations for the CUB dataset require extra manual work; the paper specifically annotated 24 scene categories for this, which may not be easily reproducible for analysis on other datasets.
- The quality of the IS-Net saliency segmentation directly affects the object-scene decoupling effect. Although experiments show robustness to segmentation failures, systematic segmentation errors could degrade performance.
- Replacing scene features with original image features is an approximation, which may have limited effectiveness when the scene occupies only a small proportion of the image.
- The value of scene information is only validated on fine-grained datasets; on general classification datasets like CIFAR-100, scene background might be less helpful.

## Related Work & Insights

- **SimGCD**: The baseline for MOS, contributing the parametric classifier and the contrastive learning framework; the modules added by MOS on top of it are extremely lightweight.
- **DCCL/PromptCAL**: Improve GCD from the perspectives of concept discovery and prompt learning, respectively, but neither utilizes scene information.
- **Insights**: The idea that "a signal considered as noise might actually be a misused signal" can be extended to other tasks; scene priors can be explored in combination with open-vocabulary detection.

## Rating

⭐⭐⭐⭐ — The core finding (that the ambiguity challenge causes scene information to be misunderstood as noise) is highly insightful, and the method is simple yet effective, achieving SOTA on fine-grained GCD. The argumentation chain is clear, forming a complete closed loop from observation $\rightarrow$ analysis $\rightarrow$ design $\rightarrow$ verification.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Hyperbolic Category Discovery](hyperbolic_category_discovery.md)
- [\[NeurIPS 2025\] Consistent Supervised-Unsupervised Alignment for Generalized Category Discovery](../../NeurIPS2025/self_supervised/consistent_supervised-unsupervised_alignment_for_generalized_category_discovery.md)
- [\[ICCV 2025\] A Hidden Stumbling Block in Generalized Category Discovery: Distracted Attention](../../ICCV2025/self_supervised/a_hidden_stumbling_block_in_generalized_category_discovery_d.md)
- [\[NeurIPS 2025\] SEAL: Semantic-Aware Hierarchical Learning for Generalized Category Discovery](../../NeurIPS2025/self_supervised/seal_semantic-aware_hierarchical_learning_for_generalized_category_discovery.md)
- [\[ICLR 2026\] Bures-Isotropy Alignment: Manifold Learning of Generalized Category Discovery](../../ICLR2026/self_supervised/bures-isotropy_alignment_manifold_learning_of_generalized_category_discovery.md)

</div>

<!-- RELATED:END -->
