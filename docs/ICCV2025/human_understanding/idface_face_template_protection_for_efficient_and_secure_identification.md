---
title: >-
  [Paper Note] IDFace: Face Template Protection for Efficient and Secure Identification
description: >-
  [ICCV 2025][Human Understanding][face template protection] This paper proposes IDFace, a homomorphic encryption (HE)-based face template protection method that achieves retrieval over 1 million encrypted templates in onl…
tags:
  - "ICCV 2025"
  - "Human Understanding"
  - "face template protection"
  - "homomorphic encryption"
  - "biometric privacy"
  - "large-scale identification"
  - "ternary quantization"
date: 2026-05-08
content_hash: e06b68054842e5f7
---

# IDFace: Face Template Protection for Efficient and Secure Identification

**Conference**: ICCV 2025
**arXiv**: [2507.12050](https://arxiv.org/abs/2507.12050)  
**Code**: N/A  
**Area**: Human Understanding / Face Recognition Security
**Keywords**: face template protection, homomorphic encryption, biometric privacy, large-scale identification, ternary quantization

## TL;DR

This paper proposes IDFace, a homomorphic encryption (HE)-based face template protection method that achieves retrieval over 1 million encrypted templates in only 126ms — incurring merely a 2× overhead compared to unprotected retrieval — through two key techniques: a near-isometric transformation (real-valued vector → ternary vector) and a space-efficient encoding scheme.

## Background & Motivation

**Background**: Face recognition systems (FRS) are widely deployed in airports, checkpoints, and other high-security scenarios. With the emergence of privacy regulations such as GDPR, secure storage of face templates has become increasingly critical. Multiple studies have demonstrated that face images can be reconstructed from unprotected templates — even without knowledge of internal model parameters — and can be leveraged for identity impersonation attacks against commercial systems.

**Limitations of Prior Work**: Homomorphic encryption (HE) theoretically supports arbitrary computation over encrypted data, but HE is fundamentally designed for algebraic operations over polynomial rings, whereas face templates are real-valued vectors ($d=512$) residing on a hypersphere. Naively combining HE with FRS yields prohibitively low efficiency — prior HE-based schemes are hundreds of times slower than unprotected retrieval, rendering them infeasible for million-scale identity databases.

**Key Challenge**: The bit-width of HE message slots far exceeds what is required to represent template values (2048-bit slots vs. 16-bit template values), causing significant spatial waste and unnecessary computational overhead. Moreover, multiplication in the encrypted domain is far more costly than addition.

**Key Insight**: To address the fundamental mismatch between HE and face templates, the paper proposes two complementary techniques: (1) transforming real-valued unit vectors into ternary vectors $\{-1,0,1\}^d$, reducing inner-product computation to pure addition and eliminating encrypted-domain multiplications entirely; and (2) exploiting the fact that each transformed component requires only 1 bit to pack multiple templates into a single message slot, maximizing SIMD utilization.

## Method

### Overall Architecture

IDFace follows a "database encryption" paradigm. During enrollment, the server encrypts all face templates; during query, matching scores are computed entirely in the encrypted domain; and a key server decrypts the results to identify the highest-scoring identity. The system separates a local server (holding the public key and encrypted gallery) from a key server (holding the private key), satisfying the security requirements of ISO 24745.

### Key Designs

1. **Near-Isometric Transformation $T_\alpha$**: Transforms $\mathbf{x} \in \mathbb{S}^{d-1}$ to $\mathcal{Z}_\alpha^d \subset \{-1,0,1\}^d$. Concretely, the $\alpha$ components with the largest absolute values are retained with their signs as $\pm 1$, while the remaining components are set to zero. The authors prove that for $d=512$, $\alpha=341$, the transformation is a $(0.111, o(1), \theta)$-isometric embedding — i.e., the change in inner product between any two vectors exceeds 0.111 with only negligible probability.

   **Design Motivation**: After transformation, the inner product $\langle T_\alpha(\mathbf{x}), T_\beta(\mathbf{y})\rangle$ requires only addition and subtraction (lookup-table operations), avoiding the prohibitively expensive multiplications in the encrypted domain.

   **Mathematical Analysis of $T_\alpha$**: Using order statistics theory, the authors rigorously prove the near-isometric property. For $d=512$, $\alpha=341$, the maximum value of $\epsilon = |\cos\theta - 2P(\theta) \cdot d/\alpha|$ is 0.111, with failure probability $\delta = o(1)$.

2. **Space-Efficient Encoding (Encode/Decode)**: The transformed inner-product values lie in $[-\alpha, \alpha]$. Each ternary vector is decomposed into a positive part $\mathbf{x}^+$ and a negative part $\mathbf{x}^-$ (where $\mathbf{x} = \mathbf{x}^+ - \mathbf{x}^-$) to ensure non-negative inner products. Using base $p > \alpha$, $m$ templates are packed into a single message slot: $\mathbf{x}^\dagger = \sum_{i=1}^m p^{i-1} \cdot \mathbf{x}_i^\dagger$.

   For Paillier (2048-bit message space), up to $m=342$ templates can be packed per slot; for CKKS (50-bit precision), $m=8$. This yields substantial improvements in both storage and computational efficiency.

3. **Query Acceleration ($\beta < \alpha$)**: On the query side, a smaller value of $\beta$ (e.g., 63 or 127) reduces the number of additions (from $2(\alpha-1)$ to $2(\beta-1)$) and increases the packing factor $m$, trading a small accuracy loss for faster retrieval.

### Loss & Training

IDFace is a plug-and-play method that requires no model training. It is directly compatible with existing FRS backbones (ArcFace, AdaFace, etc.) without modifying the feature extractor.

## Key Experimental Results

### Main Results: Efficiency Comparison (1M-Identity Gallery)

| Method | Enrollment Time | Retrieval Time | Storage | Cryptographic Primitive |
|--------|----------------|---------------|---------|------------------------|
| Unprotected | N/A | 0.063s | 2GB | N/A |
| IronMask | 4,416s | 97.9s | 1024GB | FC |
| HERS (512) | 199s | 48.9s | 16.5GB | CKKS |
| MFBR-ID | 1641s | 0.545s | 132.1GB | BFV |
| **IDFace (63, CKKS)** | **72s** | **0.126s** | **4.125GB** | **CKKS** |

IDFace ($\beta=63$) is 97.6× faster than HERS and 4.5× faster than MFBR-ID, with 31.8× less storage. Compared to unprotected retrieval, the overhead is only **2×**.

### Ablation Study: Accuracy Impact

| Dataset | Metric | Unprotected | (341,341) | (341,127) | (341,63) |
|---------|--------|-------------|-----------|-----------|----------|
| LFW | Accuracy | 99.82% | 99.78% | 99.80% | 99.82% |
| CFP-FP | Accuracy | 99.24% | 99.24% | 99.19% | 98.99% |
| AgeDB | Accuracy | 98.00% | 97.97% | 97.80% | 97.23% |
| IJB-C(V) | TAR@1e-3 | 98.39% | 98.27% | 98.14% | 97.78% |

Accuracy degradation on LFW, CFP-FP, and AgeDB remains below 1% even under the most aggressive setting of $\beta=63$. Performance drops on IJB-C identification are slightly larger but remain acceptable.

### Key Findings

- $\alpha=341$ outperforms $\alpha=512$, as discarding low-magnitude components eliminates noise and improves discriminability — validating the intuition that a small subset of dominant components governs the inner product.
- Consistently low accuracy loss is observed across multiple feature extractors (ArcFace, MagFace, SphereFace2, ElasticFace, AdaFace-KPRPE), demonstrating the generality of the approach.
- Security analysis establishes IND-CPA security based on AHE, satisfying all three Biometric Template Protection (BTP) requirements: irreversibility, revocability, and unlinkability.

## Highlights & Insights

- **The 2× Overhead Breakthrough**: IDFace is the first method to bring HE-protected million-scale face retrieval to within practical proximity of unprotected retrieval, enabling real-world deployment.
- **Plug-and-Play Compatibility**: No retraining of the FRS is required, making the method friendly to open-set scenarios where gallery templates and training data are fully independent.
- The **theoretical near-isometry analysis** of the ternary quantization transformation is independently valuable and generalizable to other inner-product-based retrieval tasks.

## Limitations & Future Work

- The current threat model considers only passive adversaries (who steal the encrypted database); the security against active adversaries (who tamper with queries or responses) is not analyzed.
- The two-server model assumes the key server is fully trusted; practical deployments would require hardware security module (HSM) support.
- Accuracy degradation introduced by the transformation becomes more pronounced under extremely low FAR requirements (e.g., $1\text{e-}4$).

## Related Work & Insights

- HERS [Engelsma 2022] introduced the paradigm of column-wise database encryption for matrix-vector product computation; IDFace builds on this by eliminating multiplications through ternarization.
- MFBR [Bassit 2022–2025] also uses lookup tables to avoid multiplications, but incurs prohibitive storage overhead (132GB vs. 4GB).
- The transformation $T_\alpha$ is technically equivalent to the error-correcting code decoding algorithm used in IronMask; however, IDFace is the first to identify and exploit its near-isometric properties.

## Rating

- Novelty: ⭐⭐⭐⭐ — innovative combination of ternary quantization and efficient encoding
- Technical Depth: ⭐⭐⭐⭐⭐ — rigorous theoretical analysis (near-isometry proof, security argumentation)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — multi-method comparison, multiple datasets, multiple feature extractors, dual evaluation of efficiency and accuracy
- Practicality: ⭐⭐⭐⭐⭐ — the 2× overhead makes large-scale secure face retrieval genuinely deployable

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] One-Shot Knowledge Transfer for Scalable Person Re-Identification](one-shot_knowledge_transfer_for_scalable_person_re-identification.md)
- [\[AAAI 2026\] CLIP-FTI: Fine-Grained Face Template Inversion via CLIP-Driven Attribute Conditioning](../../AAAI2026/human_understanding/clip-fti_fine-grained_face_template_inversion_via_clip-driven_attribute_conditio.md)
- [\[ICCV 2025\] RayPose: Ray Bundling Diffusion for Template Views in Unseen 6D Object Pose Estimation](raypose_ray_bundling_diffusion_for_template_views_in_unseen_6d_object_pose_estim.md)
- [\[ICCV 2025\] OpenAnimals: Revisiting Person Re-Identification for Animals Towards Better Generalization](openanimals_revisiting_person_re-identification_for_animals_towards_better_gener.md)
- [\[ICCV 2025\] Weakly Supervised Visible-Infrared Person Re-Identification via Heterogeneous Expert Collaborative Consistency Learning](weakly_supervised_visible-infrared_person_re-identification_via_heterogeneous_ex.md)

</div>

<!-- RELATED:END -->
