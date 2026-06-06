---
title: >-
  [Paper Note] Privacy-Aware Video Anomaly Detection through Orthogonal Subspace Projection
description: >-
  [ICML 2026][Video Understanding][Video anomaly detection] The authors propose OPL (Orthogonal Projection Layer) and its enhanced version G-OPL…
tags:
  - "ICML 2026"
  - "Video Understanding"
  - "Video anomaly detection"
  - "orthogonal projection"
  - "face suppression"
  - "privacy-aware"
  - "subspace disentanglement"
date: 2026-05-08
content_hash: 656405ac5f202442
---

# Privacy-Aware Video Anomaly Detection through Orthogonal Subspace Projection

**Conference**: ICML 2026  
**arXiv**: [2605.08651](https://arxiv.org/abs/2605.08651)  
**Code**: Not explicitly released  
**Area**: Human Understanding / Video Anomaly Detection / Privacy-Preserving Representation Learning  
**Keywords**: Video anomaly detection, orthogonal projection, face suppression, privacy-aware, subspace disentanglement

## TL;DR
The authors propose OPL (Orthogonal Projection Layer) and its enhanced version G-OPL, which use a learnable orthogonal subspace derived from QR decomposition to explicitly project out "task-irrelevant variables" and "facial privacy components" from the feature space of video anomaly detection. Four privacy-aware metrics (SSC/ARD/PD/FPD) are introduced. While maintaining or improving VAD AUC, the accuracy of linear SVM probes for facial prediction drops significantly.

## Background & Motivation

**Background**: Mainstream video anomaly detection (VAD) approaches use backbones like I3D or Swin Transformer to extract spatiotemporal features, followed by weakly supervised heads such as RTFM, MGFN, TEVAD, or EGO for scoring. As models grow larger and AUC improves, deployment in surveillance and public safety scenarios inevitably leads to representations encoding sensitive attributes like faces, clothing, and pose.

**Limitations of Prior Work**: Existing VAD systems lack explicit mechanisms to suppress task-irrelevant or ethically sensitive information. If an attacker accesses intermediate features, identity can be inferred. Current privacy-preserving methods (INLP’s nullspace projection, DAMS, CAE-LSP, OPL-2021) suffer from: (i) reliance on explicit sensitive attribute labels (typical VAD datasets lack face/identity annotations); (ii) unstable adversarial training with gradient reversal; (iii) restriction to static images or low-dimensional settings; (iv) post-hoc auditing (dataset balancing, saliency visualization) that cannot alter representations.

**Key Challenge**: Privacy and utility are entangled at the representation level—simply removing face information risks discarding pose/motion cues crucial for anomaly detection, while adversarial training alone is unstable and lacks interpretability.

**Goal**: (i) Design a differentiable module that does not require sensitive labels or adversarial training, capable of "filtering out a class of information" by inserting a single layer; (ii) Remove facial components while retaining pose/motion without identity supervision; (iii) Provide privacy evaluation metrics tailored for VAD, jointly measuring privacy, utility, and interpretability.

**Key Insight**: Projecting onto the orthogonal complement of a learned low-dimensional subspace offers a geometrically clean, differentiable, and controllable way to delete information—if the subspace captures redundant/sensitive directions, their energy can be precisely removed without affecting other directions.

**Core Idea**: Replace "adversarial training to suppress sensitive attributes" with "geometric projection + cosine alignment weak supervision," parameterizing a sensitive subspace via QR decomposition and projecting onto its orthogonal complement.

## Method

### Overall Architecture
Given an intermediate feature $\bm f\in\mathbb R^d$ (from the backbone or a certain layer), OPL learns a matrix $\bm W\in\mathbb R^{k\times d}$ ($1<k<d$). QR decomposition is performed on $\bm W^\top$ to obtain $\bm W^\top=\bm Q\bm R$, where $\bm Q\in\mathbb R^{d\times k}$ forms an orthogonal basis for the $k$-dimensional nuisance subspace. The projection matrix is $\bm P=\bm I_d-\bm Q\bm Q^\top$, and the purified feature is $\bm f_{\text{proj}}=\bm P\bm f=\bm f-\bm Q\bm Q^\top\bm f$. The entire layer is differentiable and trained jointly with the main task loss. G-OPL adds a cosine alignment loss: an off-the-shelf face detector (RetinaFace) extracts face crops, which are encoded to $\bm f_{\text{face}}$; $\bm Q\bm Q^\top\bm f$ is encouraged to be cosine-similar to $\bm f_{\text{face}}$, actively injecting the "face direction" into the subspace to be discarded. OPL is placed in deeper layers for general nuisance removal, while G-OPL is inserted right after the backbone for early face suppression.

### Key Designs

1. **QR-Decomposition-Parameterized Learnable Orthogonal Projection Layer (OPL)**:

    - **Function**: Differentiably projects features onto the orthogonal complement of a learned low-dimensional subspace, removing "task-irrelevant" components while retaining useful directions—a task-adaptive "feature purifier."
    - **Mechanism**: Explicitly parameterize the subspace to be removed as a trainable matrix $\bm W\in\mathbb R^{k\times d}$. Before each forward pass, QR decomposition on $\bm W^\top$ yields orthogonal basis $\bm Q$, then project with $\bm P=\bm I_d-\bm Q\bm Q^\top$. The process is end-to-end differentiable with respect to the main VAD loss—subspace directions are pushed by task gradients toward "removable without harming detection," akin to a PCA-style geometric method but targeting "maximal task retention + maximal projection deletion."
    - **Design Motivation**: Compared to PCA’s fixed subspace and INLP’s iterative nullspace projection, QR decomposition ensures numerically stable orthogonal bases per forward pass, avoids adversarial gradient reversal, and does not require sensitive attribute labels. The structure is more interpretable than adversarial discriminators (the projected-out component is $\bm Q\bm Q^\top\bm f$, which can be visualized).

2. **Guided OPL (G-OPL) + Cosine Alignment Weakly-Supervised Face Suppression**:

    - **Function**: Without identity labels, inject the "face direction" into the subspace to be removed, explicitly discarding facial biometric components.
    - **Mechanism**: Both original video frames and RetinaFace-detected face crops (averaged over multiple faces, plus 50 segments from Georgia Tech Face DB as controls) are encoded (I3D/SwinT) to $\bm f,\bm f_{\text{face}}$, ensuring they reside in the same latent space. The main VAD loss is augmented as $\mathcal L_{\text{task}}=\mathcal L_{\text{ori}}+\lambda_{\text{face}}(1-\cos(\bm f_{\text{face}},\bm Q\bm Q^\top\bm f))$, forcing $\bm Q\bm Q^\top\bm f$ to align in angle with the face embedding—i.e., attracting the face direction into the subspace to be projected out. This loss is activated only on frames where faces are detected.
    - **Design Motivation**: The approach deliberately avoids adversarial training (unstable, requires discriminator) and explicit attribute classifiers (needs identity labels), using cosine as a geometric "soft label" for unsupervised, stable supervision. RetinaFace provides only binary face-presence and face embedding, so no identity ground truth is needed at deployment. The method generalizes to multiple attributes by replacing/concatenating other weak signals (e.g., torso, clothing) with the face embedding.

3. **Orthogonality Regularization + Privacy-Aware Metric Suite (SSC/ARD/PD-FPD)**:

    - **Function**: (i) Prevents $\bm Q$ from drifting away from orthogonality during training; (ii) Provides quantifiable metrics distinguishing "sensitive subspace capture," "task retention," and "layerwise information decay."
    - **Mechanism**: (a) Orthogonality regularization $\mathcal L_{\text{orth}}=\|\bm Q^\top\bm Q-\bm I_k\|_F^2$, with total loss $\mathcal L_{\text{total}}=\mathcal L_{\text{task}}+\lambda_{\text{orth}}\mathcal L_{\text{orth}}$. (b) Sensitive Subspace Capture $\mathrm{SSC}=\cos(\bm Q\bm Q^\top\bm f_{\text{attr}}^{(i)},\bm f_{\text{attr}}^{(i)})$ measures whether the learned subspace captures sensitive attributes; (c) Anomaly Retention Distance $\mathrm{ARD}=\mathrm{KL}(P_{\text{raw}}(y)\|P_{\text{proj}}(y))$, the KL divergence (via KDE) between anomaly score distributions before/after projection—smaller values indicate better utility retention; (d) Privacy Decay $\{(l,\mathrm{Acc}^{(l)})\}_{l=1}^L$ uses a linear SVM probe after each G-OPL to predict face presence—lower accuracy indicates stronger suppression; first-layer PD (FPD) refers to accuracy after the first G-OPL.
    - **Design Motivation**: QR decomposition yields orthogonal bases per forward pass, but gradient updates can break orthogonality, especially with stacked G-OPLs, necessitating explicit regularization. The SSC/ARD/PD suite fills the gap in VAD privacy evaluation—previously, only AUC was available, which cannot assess face suppression.

### Loss & Training

The total loss is $\mathcal L_{\text{total}}=\mathcal L_{\text{ori}}+\lambda_{\text{face}}(1-\cos(\bm f_{\text{face}},\bm Q\bm Q^\top\bm f))+\lambda_{\text{orth}}\|\bm Q^\top\bm Q-\bm I_k\|_F^2$, where $\mathcal L_{\text{ori}}$ is the original weakly supervised VAD baseline loss (RTFM/MGFN/TEVAD/EGO, respectively). The face alignment term is activated only on frames with detected faces; other frames are skipped.

## Key Experimental Results

### Main Results
On five VAD datasets (ShanghaiTech, UCF-Crime, CUHK Avenue, UCSD Ped2, MSAD), OPL/G-OPL are inserted into four SOTA baselines (RTFM, MGFN, TEVAD, EGO), using unified I3D/Swin Transformer features. Table 1 from the paper shows ShanghaiTech ablation (AUC %):

| $k_{\text{OPL}}\backslash k_{\text{G-OPL}}$ | $2$ | $4$ | $8$ | $16$ | $32$ | $64$ | $128$ |
|---|---|---|---|---|---|---|---|
| $2$ | $95.5$ | $95.9$ | $95.6$ | $94.8$ | $93.9$ | $92.5$ | $91.8$ |
| $4$ | $95.9$ | $\mathbf{97.3}$ | $96.8$ | $95.2$ | $94.0$ | $92.8$ | $91.9$ |
| $8$ | $95.7$ | $97.0$ | $96.5$ | $95.0$ | $93.8$ | $92.6$ | $91.7$ |
| $32$ | $94.5$ | $95.1$ | $94.6$ | $93.8$ | $92.8$ | $91.5$ | $90.8$ |
| $128$ | $92.8$ | $93.1$ | $92.4$ | $91.5$ | $89.8$ | $88.4$ | $87.9$ |

The optimal $k_{\text{OPL}}=k_{\text{G-OPL}}=4$ yields AUC $=97.3\%$, clearly higher than the baseline RTFM (RTFM I3D on ShT in the paper is about $97.0\%$); overly large $k$ (e.g., $128$) excessively removes information, dropping AUC to $87.9\%$.

MSAD multi-anomaly comparison (excerpt from Table 2, AUC %):

| Method | Assault | Explosion | Fighting | Robbery | Shooting | Traffic Acc. | Overall AUC |
|------|---------|-----------|----------|---------|----------|--------------|-------------|
| RTFM (I3D) baseline | $53.9$ | $66.0$ | $79.8$ | … | … | … | … |
| + OPL / G-OPL (Ours) | Comprehensive improvement or parity | — | — | — | — | — | — |

(See the original table for specific OPL/G-OPL gains; the main trend is that utility does not decrease and may slightly increase, while privacy metrics drop significantly.)

### Ablation Study

| Configuration | Key Metrics | Notes |
|------|---------|------|
| baseline RTFM (I3D) | High AUC, but FPD close to baseline backbone (face can be accurately predicted by linear SVM) | No privacy mechanism |
| + OPL | AUC unchanged/slightly increased, UMAP shows nuisance clusters dispersed | General nuisance removed |
| + G-OPL (cosine alignment) | FPD drops sharply, SSC rises significantly, AUC unaffected | Facial components are actively injected into subspace and projected out |
| Remove $\mathcal L_{\text{orth}}$ | After a few epochs, $\bm Q$ deviates from orthogonality, AUC fluctuates | Orthogonality must be maintained |
| $k$ too large ($\ge 64$) | AUC drops sharply | Subspace too wide, useful information lost |

### Key Findings
- Placing G-OPL immediately after the backbone (early stage) is more effective than in deeper layers—the lowest FPD (first-layer post-linear SVM face prediction accuracy) indicates privacy should be intercepted before information diffuses through the network.
- ARD (KL divergence) increases monotonically with $k$, but AUC peaks at $k=4$, indicating utility does not monotonically vary with $k$—removing a small amount of nuisance can reduce overfitting and improve discrimination.
- Using ArcFace as an attacker-style rank-1 re-identification probe, retrieval accuracy drops significantly after G-OPL, showing not only coarse metrics like face presence but also fine-grained identity are suppressed.

## Highlights & Insights
- "Replacing adversarial training with geometric projection" is the most noteworthy paradigm—sensitive components can be removed with stable training and interpretability (directly inspect $\bm Q\bm Q^\top\bm f$), without identity labels. This can be transferred to face anti-spoofing, human keypoint, medical imaging, and other privacy-sensitive tasks.
- Using cosine alignment as a *weak supervision* signal is particularly clever—any off-the-shelf detector providing a "concept vector" (face embedding, clothing embedding) allows geometric injection of "what is sensitive" into $\bm Q$, which can then be plugged into any VAD baseline.
- The SSC/ARD/PD-FPD suite fills the gap in VAD privacy evaluation; future vision tasks (action recognition, re-ID defense) can directly adopt these metrics.

## Limitations & Future Work
- Using RetinaFace as the sole weak supervision source for faces means failure to detect small, occluded, or low-resolution faces causes G-OPL to fail; the paper acknowledges that MSAD is already face-blurred, so only pre-extracted features can be validated.
- Current G-OPL targets only "face" as a sensitive attribute; suppressing multiple attributes (face + clothing + gait) with a shared $\bm Q$ may cause interference among attributes, requiring finer alignment design.
- $k$ is a critical hyperparameter and must be tuned per dataset; the paper provides the best $k=4$ for ShanghaiTech, but cross-dataset generalization is not fully tested.
- No robustness evaluation against *adaptive attackers* (who know OPL exists and train inversion networks); current privacy conclusions are for *non-adaptive* SVM/ArcFace probes.

## Related Work & Insights
- **vs INLP / nullspace projection (Ravfogel 2020)**: INLP uses iterative nullspace removal but requires sensitive attribute ground truth; G-OPL uses cosine weak supervision + differentiable QR for one-shot end-to-end training.
- **vs OPL-2021 (Ranasinghe 2021)**: Previous work used generic decorrelation loss without "sensitive direction" control; this work adds face cosine alignment for controllable, targeted subspace removal.
- **vs Adversarial Training + Gradient Reversal (GANIN, etc.)**: Unstable, requires discriminator training; this work circumvents with geometric methods.
- **vs Data-Level Privacy (face blurring, DP-SGD)**: Data-level methods require pipeline changes; G-OPL is a model-level plug-in without altering data.

## Rating
- Novelty: ⭐⭐⭐⭐ First to combine "differentiable QR orthogonal projection + cosine weak supervision" for VAD; the privacy metric suite is also a new contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ 5 datasets × 4 baselines × multiple $k$ settings + ArcFace inversion attacker validation, broad coverage.
- Writing Quality: ⭐⭐⭐⭐ Clear flow from motivation → OPL → G-OPL → metrics → experiments; appendix supplements theory and practical details.
- Value: ⭐⭐⭐⭐ Engineering-friendly (plug-in module, backbone-agnostic), fixed $\bm Q$ at inference with no extra cost, practically meaningful for deployment in surveillance cameras, etc.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] StegaVAR: Privacy-Preserving Video Action Recognition via Steganographic Domain Analysis](../../AAAI2026/video_understanding/stegavar_privacy-preserving_video_action_recognition_via_steganographic_domain_a.md)
- [\[AAAI 2026\] Balancing Multimodal Domain Generalization via Gradient Modulation and Projection](../../AAAI2026/video_understanding/balancing_multimodal_domain_generalization_via_gradient_modulation_and_projectio.md)
- [\[ICCV 2025\] Aligning Effective Tokens with Video Anomaly in Large Language Models](../../ICCV2025/video_understanding/aligning_effective_tokens_with_video_anomaly_in_large_language_models.md)
- [\[AAAI 2026\] Group Orthogonal Low-Rank Adaptation for RGB-T Tracking](../../AAAI2026/video_understanding/group_orthogonal_low-rank_adaptation_for_rgb-t_tracking.md)
- [\[AAAI 2026\] Learning Topology-Driven Multi-Subspace Fusion for Grassmannian Deep Networks](../../AAAI2026/video_understanding/learning_topology-driven_multi-subspace_fusion_for_grassmannian_deep_network.md)

</div>

<!-- RELATED:END -->
