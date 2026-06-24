---
title: >-
  [Paper Note] Optimal Transport-Guided Source-Free Adaptation for Face Anti-Spoofing
description: >-
  [CVPR 2025][Human Understanding][Face Anti-Spoofing] The OTA framework is proposed: during the training phase, prototype representations are learned to encode the source domain distribution. During the testing phase, the prototypes are transferred to the target domain via optimal transport (OT) in a training-free or lightweight-training manner without accessing source model parameters or training data. Concurrently, geodesic mixup data augmentation is proposed to improve clas…
tags:
  - "CVPR 2025"
  - "Human Understanding"
  - "Face Anti-Spoofing"
  - "Optimal Transport"
  - "Source-Free Domain Adaptation"
  - "Few-Shot Adaptation"
  - "Prototype Learning"
date: 2026-05-08
content_hash: 7cb82d5e379e3170
---

# Optimal Transport-Guided Source-Free Adaptation for Face Anti-Spoofing

**Conference**: CVPR 2025  
**arXiv**: [2503.22984](https://arxiv.org/abs/2503.22984)  
**Code**: None  
**Area**: AI Security  
**Keywords**: Face Anti-Spoofing, Optimal Transport, Source-Free Domain Adaptation, Few-Shot Adaptation, Prototype Learning

## TL;DR

The OTA framework is proposed: during the training phase, prototype representations are learned to encode the source domain distribution. During the testing phase, the prototypes are transferred to the target domain via optimal transport (OT) in a training-free or lightweight-training manner without accessing source model parameters or training data. Concurrently, geodesic mixup data augmentation is proposed to improve classifier learning in low-data scenarios.

## Background & Motivation

**Background**: Face anti-spoofing (FAS) is a critical security component of face recognition systems. Traditional methods cover various real-user interactions and spoof attacks by collecting large-scale datasets, but the diversity of user conditions is virtually limitless. Domain adaptation (DA) requires simultaneous access to both source and target domain data, whereas domain generalization (DG) relies solely on static parameters to generalize to downstream scenarios.

**Limitations of Prior Work**: (1) Legal and privacy constraints limit the amount of reference data and prohibit the host from sharing model parameters and training data with clients; (2) Fine-tuning an independent model for each client is impractical due to high maintenance costs; (3) Clients require rapid customized services to adapt to constantly changing usage scenarios. Although existing SFDA methods do not require source data, they do not restrict access to the source model.

**Key Challenge**: The service provider cannot share model parameters or training data (due to privacy/security constraints), and the client has only a small amount of labeled target domain data. Under such tight constraints, how can effective model adaptation be achieved?

**Goal**: Build a privileged system that allows clients to perform lightweight, user-specific customization at test time using a few labeled samples, without requiring the host to share model parameters or training samples.

**Key Insight**: Prototypes can encode source domain distribution information while remaining compact and privacy-respecting. Optimal Transport (OT) is particularly suited for few-shot scenarios—it leverages Wasserstein distance to effectively utilize the geometric structure of the underlying feature space, faithfully aligning the source and target domain distributions even when data is sparse.

**Core Idea**: Learn multi-center prototypes during training as a compact representation of the source domain distribution, and project the prototypes to the target domain via optimal transport at test time. Two adaptation schemes are provided: training-free (direct prototype transformation via OT) and lightweight training (training a lightweight classifier on geodesic mixup synthetic data).

## Method

### Overall Architecture

The process is divided into a training phase and a testing adaptation phase. Training phase: A feature extractor $f$ and multi-center prototypes $P = \{p^{\text{bona-fide}}, p^{\text{spoof}}\} \in \mathbb{R}^{D \times K \times 2}$ ($K$ sub-centers) are trained on multi-source domain data. Testing phase: The feature extractor $f$ is treated as a black box; only the prototypes $P$ and a small number of target domain samples $\mathbb{D}_t$ are used for adaptation. Two pathways are provided: training-free and lightweight training.

### Key Designs

1. **Prototype-based Framework**:

    - **Function**: Learns a compact representation of the source domain distribution while acting as a classifier.
    - **Mechanism**: $K$ sub-center prototypes are learned for each class (bona fide/spoof). During classification, the average cosine similarity between the embedding $z = f(x)$ and each group of prototypes is calculated. The training loss consists of three parts: (1) a prototype loss $\mathcal{L}_{proto}$ based on an ArcFace variant (adding margin $m$ in angular space); (2) coarse-grained and fine-grained supervised contrastive loss $\mathcal{L}_{con}$; (3) an orthogonal regularization loss $\mathcal{L}_{orth}$ to prevent sub-center collapse. The overall loss is $\mathcal{L} = \mathcal{L}_{proto} + \alpha \mathcal{L}_{con}^{coarse} + \beta \mathcal{L}_{con}^{fine} + \eta \mathcal{L}_{orth}$.
    - **Design Motivation**: Prototypes serve as both classifiers and distribution proxies, which can be transmitted to clients without exposing the model parameters. Multiple sub-centers enhance representational capability.

2. **Training-free Optimal Transport Adaptation**:

    - **Function**: Transfers source domain prototypes to the target domain without any training.
    - **Mechanism**: Formulates an optimal transport problem between the source prototypes ($2K$ prototypes) and target domain few-shot features ($M_t$ features). The cost matrix $M$ is defined based on cosine distance, with Laplacian regularization $\Omega_\alpha$ added to preserve data structures. After solving for the OT plan $\gamma^*$, each prototype is projected to the target domain via barycentric projection: $p^* = \sum_{j=1}^{M_t} \pi_{i,j} z_{t,j}$. The transformation is performed per class to maintain discriminative ability. The transformed prototypes $P^*$ directly serve as the classifier.
    - **Design Motivation**: OT theory is naturally suited for discrete distributions and few-shot scenarios, achieving effective alignment by utilizing the geometric structure of the feature space.

3. **Geodesic Mixup Lightweight Training Adaptation**:

    - **Function**: Generates synthetic training data between the source and target domains to train a lightweight classifier.
    - **Mechanism**: Unlike traditional mixup which performs point-to-point linear interpolation, this method generates synthetic distributions along the geodesic paths defined by OT. Specifically, synthetic data is sampled under different interpolation ratios $t \in [0,1]$ of the OT mapping as $\mu_t = ((1-t)id + tT)\#\mu_s$, where $T$ is the OT mapping. These synthetic data guide the classifier to learn the transition of features between domains, adapting to target domain characteristics while preserving source domain knowledge.
    - **Design Motivation**: Direct training of classifiers in low-data scenarios is prone to overfitting. Synthetic data along the geodesic path captures the structure of the underlying feature manifold much better than linear interpolation.

### Loss & Training

- **Training Phase**: Joint training on multiple source domains using the ArcFace variant loss + dual-granularity contrastive loss + orthogonal regularization.
- **Testing Adaptation**:
    - Training-free: Solve the regularized OT problem + barycentric projection to transform prototypes, which requires no gradient updates.
    - Lightweight training: Train a lightweight MLP classifier on geodesic mixup synthetic data while freezing the feature extractor.
- The margin $m$ is set as a learnable parameter.

## Key Experimental Results

### Main Results (Cross-Domain: OCIM Standard Protocol)

| Method | OCI→M | OMI→C | OCM→I | ICM→O | Avg HTER ↓ |
|------|-------|-------|-------|-------|-----------|
| SSDG-R | 7.38 | 10.44 | 11.71 | 15.61 | 11.28 |
| SSAN-R | 6.67 | 10.00 | 8.88 | 13.72 | 9.81 |
| SA-FAS | 5.95 | 8.78 | 6.58 | 10.00 | 7.82 |
| CFPL | 3.09 | 2.56 | 5.43 | 3.33 | 3.60 |
| OTA (zero-shot) | 2.62 | 2.22 | 5.32 | 3.56 | 3.43 |
| OTA (training-free, 5-shot) | 2.38 | 2.67 | — | — | — |
| ViTAF (5-shot) | 3.42 | 1.40 | 3.74 | 7.17 | 3.93 |

### Ablation Study

- The optimal performance is achieved when the number of sub-centers $K=8$.
- The combination of coarse-grained and fine-grained contrastive losses outperforms their individual use.
- Geodesic mixup outperforms traditional linear mixup (with a significant improvement in HTER).
- Laplacian regularization in OT is crucial for few-shot scenarios.

### Key Findings

- Even in the zero-shot setting (without using target domain data), the prototype method achieves performance close to the SOTA (average HTER 3.43 vs 3.60 of CFPL).
- 5-shot training-free adaptation further improves performance without requiring any training.
- The method achieves a relative improvement of 19.17% in HTER and an 8.58% improvement in AUC, demonstrating its effectiveness.
- The proposed method remains valid in cross-attack settings.
- Even in one-class scenarios (where target domain data contains only a single class), geodesic mixup still provides effective adaptation.

## Highlights & Insights

- **Highly Practical Problem Setting**: The scenario where source model parameters are not shared, source data is inaccessible, and the client only has a small amount of data perfectly aligns with real-world deployment requirements.
- **Elegant Design of Prototypes as Distribution Proxies**: Compact, privacy-friendly, playing a dual role as both classifiers and distribution representations.
- **Natural Adaptation of OT Theory**: The theoretical advantages of OT in few-shot and discrete distribution scenarios are fully leveraged.
- **Innovative Geodesic Mixup**: Generating synthetic data along the OT geodesic path preserves the manifold structure better than point-wise linear interpolation.
- **Flexibility**: Provides both training-free and lightweight training options to flexibly adapt to different requirements.

## Limitations & Future Work

- The performance depends heavily on the quality of the pre-trained feature extractor; if the feature extractor is not sufficiently generalizable, the effectiveness of prototype transfer is limited.
- The number of prototypes ($K$) needs to be predetermined, and the optimal $K$ may vary across different datasets.
- The IPM (Inverse Perspective Mapping) assumption (flatness assumption) may limit performance in certain scenarios.
- Unsupervised target domain adaptation solutions can be explored in future work.
- The framework can be extended to other security-related classification tasks.

## Related Work & Insights

- **ArcFace / Sub-center ArcFace**: Prototype training draws inspiration from the angular margin concept of ArcFace and extends it to multiple sub-centers.
- **SFDA Series**: The key difference is that this work strictly restricts access to source model parameters, whereas most SFDA works allow complete model access.
- **OT in Domain Adaptation**: The innovation of this work lies in using OT for prototype transfer rather than feature alignment.
- **Insight**: For privacy-sensitive scenarios, "lightweight proxy (prototype) + OT transfer" is a more practical paradigm than traditional domain adaptation.

## Rating

| Dimension | Score (1-10) |
|------|------------|
| Novelty | 8 |
| Technical Depth | 8 |
| Experimental Thoroughness | 8 |
| Writing Quality | 8 |
| Value | 9 |
| Overall Rating | 8.2 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] DADM: Dual Alignment of Domain and Modality for Face Anti-Spoofing](../../ICCV2025/human_understanding/dadm_dual_alignment_of_domain_and_modality_for_face_anti-spoofing.md)
- [\[CVPR 2026\] FaceCoT: Chain-of-Thought Reasoning in MLLMs for Face Anti-Spoofing](../../CVPR2026/human_understanding/facecot_cot_reasoning_face_anti_spoofing.md)
- [\[ECCV 2024\] Towards Unified Representation of Invariant-Specific Features in Missing Modality Face Anti-Spoofing](../../ECCV2024/human_understanding/towards_unified_representation_of_invariant-specific_features_in_missing_modalit.md)
- [\[CVPR 2026\] From Intuition to Investigation: A Tool-Augmented Reasoning MLLM Framework for Generalizable Face Anti-Spoofing](../../CVPR2026/human_understanding/from_intuition_to_investigation_a_tool-augmented_reasoning_mllm_framework_for_ge.md)
- [\[ECCV 2024\] TF-FAS: Twofold-Element Fine-Grained Semantic Guidance for Generalizable Face Anti-Spoofing](../../ECCV2024/human_understanding/tf-fas_twofold-element_fine-grained_semantic_guidance_for_generalizable_face_ant.md)

</div>

<!-- RELATED:END -->
