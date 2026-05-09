---
title: >-
  [Paper Note] DETree: DEtecting Human-AI Collaborative Texts via Tree-Structured Hierarchical Representation Learning
description: >-
  [NEURIPS2025][Object Detection][AI text detection] This paper proposes DETree, a framework that constructs a Hierarchical Affinity Tree (HAT) to model the hierarchical relationships among diverse human-AI collaborative text generation processes, and designs a Tree-Structured Contrastive Loss (TSCL) to align the representation space. DETree achieves significant advantages in mixed-text detection and OOD generalization scenarios.
tags:
  - NEURIPS2025
  - Object Detection
  - AI text detection
  - human-AI collaborative text
  - hierarchical representation learning
  - contrastive learning
  - out-of-distribution generalization
date: 2026-05-08
content_hash: d5918878c2f0ecec
---

# DETree: DEtecting Human-AI Collaborative Texts via Tree-Structured Hierarchical Representation Learning

**Conference**: NEURIPS2025
**arXiv**: [2510.17489](https://arxiv.org/abs/2510.17489)
**Code**: [heyongxin233/DETree](https://github.com/heyongxin233/DETree)
**Area**: Object Detection
**Keywords**: AI text detection, human-AI collaborative text, hierarchical representation learning, contrastive learning, out-of-distribution generalization

## TL;DR

This paper proposes DETree, a framework that constructs a Hierarchical Affinity Tree (HAT) to model the hierarchical relationships among diverse human-AI collaborative text generation processes, and designs a Tree-Structured Contrastive Loss (TSCL) to align the representation space. DETree achieves significant advantages in mixed-text detection and OOD generalization scenarios.

## Background & Motivation

With the widespread adoption of LLMs, AI involvement in text generation has become increasingly diverse: humans write and AI polishes, AI generates and humans revise, multiple AI models collaborate, etc. Different contexts have varying tolerances for AI involvement (e.g., acceptable in advertising copy, strictly prohibited in academic writing). Therefore, detection methods must not only determine whether a text involves AI, but also identify the specific mode and degree of AI participation.

Existing methods have notable limitations:

- **Binary classification methods**: Distinguish only between purely human-written and purely AI-generated text, unable to handle the complexity of human-AI collaboration
- **Coarse-grained estimation**: Such as regressing AI involvement degree or classifying content/style features based on predefined prompts, with limited expressive capacity
- **Weak generalization**: Perform well on training distributions but suffer severe performance degradation in OOD scenarios

The authors' key observation is that texts produced by different generation processes naturally form hierarchically clustered structures in representation space. For example, "Llama3_polish_GPT-4o" is more similar to "Claude3.5_paraphrase_GPT-4o" than to "human_polish_Gemini1.5", and both are more similar to each other than to purely human-written text.

## Core Problem

1. How to structurally model the intrinsic relationships among diverse generation processes of human-AI collaborative texts?
2. How to construct a large-scale benchmark dataset covering diverse human-AI collaboration patterns?
3. How to maintain strong generalization in OOD scenarios (new models, new domains)?

## Method

### 1. RealBench Dataset Construction

Raw samples from MAGE, M4, TuringBench, OUTFOX, and RAID are aggregated. Mixed-text construction strategies including paraphrasing, extension, polishing, and translation are introduced, and 11 perturbation-based attack types are integrated. The final dataset covers **1,204 text categories** with approximately **16.4 million text samples**.

### 2. Hierarchical Affinity Tree (HAT) Construction

- **Step 1**: Fine-tune the encoder using supervised contrastive learning, treating each category as an independent class, and compute the inter-class similarity matrix $E \in \mathbb{R}^{N \times N}$
- **Step 2**: Apply agglomerative hierarchical clustering on the similarity matrix to generate an initial binary tree structure
- **Step 3**: Introduce an editable top-down subtree reorganization algorithm. Three prior configurations are predefined (mixed texts assigned to human/AI/independent categories). For each subtree, all possible partitions are enumerated and the optimal partition is selected via Silhouette Score; the process is applied recursively until a stopping criterion is met
- Computational complexity is $\mathcal{O}(N^2 \log N)$

### 3. Tree-Structured Contrastive Loss (TSCL)

The core idea is that categories sharing a closer common ancestor in the HAT should be more similar in the representation space.

For a leaf node $c$ at depth $d_c$, the hierarchical partition set at level $i$ is defined as:

$$H_c^{(i)} = \{x \mid x \in \text{leaf}(f_c^{(i)}) \setminus \text{leaf}(f_c^{(i-1)})\}$$

Based on the equivalence established in Theorem 3.1, the hierarchical similarity constraint is converted into an optimizable contrastive learning objective. At each level $i$, the positive set $P_i = H_c^{(i)}$ and negative set $N_i = \bigcup_{j=i+1}^{d_c} H_c^{(j)}$ are defined. The full TSCL is:

$$\mathcal{L}_{\text{TSCL}}(x;\theta) = \frac{1}{d_c} \sum_{i=0}^{d_c-1} \mathcal{L}_c^{(i)}(x;\theta)$$

### 4. Virtual Class Prototype (VCP)

Given the large number of categories (1,204), a single mini-batch cannot cover all classes. A learnable prototype vector $\mathbf{v}_c \in \mathbb{R}^d$ is introduced for each category as a persistent anchor in contrastive learning, without introducing additional memory overhead.

### 5. Inference and OOD Adaptation

- **K-Means Database Compression**: K-means clustering is applied per category to generate compact representatives, reducing retrieval cost while maintaining balanced category representation
- **Retrieval-Based Few-Shot Adaptation**: A small number of target-domain samples are used to reconstruct the retrieval database and adjust decision boundaries to accommodate domain shift

### Implementation Details

- Encoder: RoBERTa-large + LoRA fine-tuning
- Optimizer: AdamW with cosine annealing, initial learning rate 3e-5, 2000-step linear warmup
- Training: 10 epochs, 8×RTX 4090, batch size 64, maximum input length 512
- Inference: Faiss-GPU for accelerated K-means and KNN, representations taken from layers 17–19, k=5 or 50
- Temperature parameter $\tau = 0.07$

## Key Experimental Results

### Supervised Detection

| Method | MAGE AvgRec | M4-mono AvgRec | TuringBench AvgRec | Avg. AvgRec |
|------|-------------|----------------|--------------------| ------------|
| DeTeCTive | 96.15 | 98.44 | 99.74 | 96.94 |
| DETree (prior1) | **96.87** | **99.86** | **99.74** | **97.88** |

### OOD Generalization (AUROC)

| Dataset | Binoculars | MAGE | DETree zero-shot | DETree 10-shot |
|--------|-----------|------|------------------|----------------|
| MAGE-Unseen | 96.84 | 95.20 | 99.13 | **99.81** |
| MAGE-Paraphrase | 75.87 | 83.35 | 92.66 | **98.77** |
| DetectRL-MultiDomain | 83.95 | 86.67 | 98.94 | **99.88** |
| Beemo-GPT4o-Edited | 78.15 | 67.79 | 83.79 | **88.54** |

### Mixed-Text Detection (HART, AUROC)

| Detector | Level-1 ALL | Level-2 ALL | Level-3 ALL |
|--------|------------|------------|------------|
| HART(Binoculars) | 0.838 | 0.848 | 0.883 |
| DETree (prior1) | **0.998** | **0.992** | **0.988** |

99.32% of triplets satisfy the hierarchical similarity constraint defined in Theorem 3.1, validating the effectiveness of TSCL.

## Highlights & Insights

1. **Novel hierarchical modeling paradigm**: Text detection is elevated from flat classification to tree-structured hierarchical representation learning; HAT autonomously captures relationships among text sources (e.g., clustering by model family or operation strategy)
2. **Solid theoretical foundation**: Theorem 3.1 proves the equivalence of the hierarchical similarity constraint, providing a theoretical basis for the TSCL design
3. **Strong OOD generalization**: Via retrieval-based few-shot adaptation, only 5–10 target-domain samples suffice to substantially improve cross-domain detection performance (+15.55 AUROC on MAGE-Paraphrase)
4. **Large-scale dataset**: RealBench covers 1,204 categories and 16.4 million samples, constituting the most comprehensive human-AI collaborative text benchmark to date
5. **Strong robustness**: Without adversarial training, DETree still outperforms adversarially trained RoBERTa-large in most attack scenarios
6. **Interesting findings**: AI traces in mixed texts are more salient than human features; human editing does not fundamentally alter the underlying AI characteristics of the text

## Limitations & Future Work

1. Adversarial evasion via model fine-tuning has not been explored
2. The current focus is on mixed texts involving up to three authors (two AIs or one AI and one human); scenarios with more collaborators remain to be studied
3. When encountering entirely unseen rare domains, a small number of target-domain samples are still required to adjust decision boundaries; fully zero-shot generalization is not yet achievable
4. The paper's classification under object detection is debatable; it substantively belongs to NLP/AI safety

## Related Work & Insights

| Dimension | Traditional Methods | HART | DETree |
|------|---------|------|--------|
| Classification Granularity | Binary | Three-level risk | Hierarchical category tree |
| Modeling Approach | Flat features | Content/style decoupling | Tree-structured contrastive learning |
| OOD Generalization | Weak | Moderate | Strong (few-shot adaptation) |
| Data Scale | Tens of thousands | Hundreds of thousands | Tens of millions (RealBench) |

Distinction from Hierarchical Text Classification (HTC): In HTC, label hierarchies are predefined and fixed, whereas in mixed-text detection, labels are inherently ambiguous and dynamically evolving; HAT construction is a data-driven adaptive process.

The hierarchical modeling paradigm is generalizable to other tasks requiring fine-grained category relationship modeling, such as malicious code family detection and deepfake video provenance tracing. The inference paradigm of K-Means database compression combined with KNN retrieval holds general value for large-scale classification scenarios. The few-shot retrieval adaptation method provides a new viable direction for training-based detectors in OOD settings. The dataset construction strategy of RealBench (compositional category expansion) is reusable for other research requiring large-scale multi-category datasets.

## Rating
- **Novelty**: 8/10 — HAT and TSCL are pioneering contributions in the AI text detection domain
- **Experimental Thoroughness**: 9/10 — Covers supervised detection, OOD generalization, mixed-text detection, robustness, and compression; ablation studies are comprehensive
- **Writing Quality**: 8/10 — Well-structured, with complete theoretical derivations and information-rich figures and tables
- **Value**: 8/10 — The method is well-designed and practically applicable; the dataset contribution is significant and directly relevant to real-world AI text auditing

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Show, Don't Tell: Detecting Novel Objects by Watching Human Videos](../../CVPR2026/object_detection/show_dont_tell_detecting_novel_objects_by_watching_human_videos.md)
- [\[ICCV 2025\] UPRE: Zero-Shot Domain Adaptation for Object Detection via Unified Prompt and Representation Enhancement](../../ICCV2025/object_detection/upre_zero-shot_domain_adaptation_for_object_detection_via_unified_prompt_and_rep.md)
- [\[CVPR 2026\] PHAC: Promptable Human Amodal Completion](../../CVPR2026/object_detection/phac_promptable_human_amodal_completion.md)
- [\[ICCV 2025\] Augmenting Moment Retrieval: Zero-Dependency Two-Stage Learning](../../ICCV2025/object_detection/augmenting_moment_retrieval_zero-dependency_two-stage_learning.md)
- [\[ICCV 2025\] The Devil is in the Spurious Correlations: Boosting Moment Retrieval with Dynamic Learning](../../ICCV2025/object_detection/the_devil_is_in_the_spurious_correlations_boosting_moment_retrieval_with_dynamic.md)

</div>

<!-- RELATED:END -->
