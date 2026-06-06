---
title: >-
  [Paper Note] Privacy-Aware Video Anomaly Detection through Orthogonal Subspace Projection
description: >-
  [ICML 2026][Video Understanding][Video Anomaly Detection] The authors propose the Orthogonal Projection Layer (OPL) and its enhanced version G-OPL. By utilizing a learnable orthogonal subspace derived from QR decompositi…
tags:
  - "ICML 2026"
  - "Video Understanding"
  - "Video Anomaly Detection"
  - "Orthogonal Projection"
  - "Facial Suppression"
  - "Privacy-Aware"
  - "Subspace Decoupling"
date: 2026-05-08
content_hash: 9fdc5701736d2323
---

# Privacy-Aware Video Anomaly Detection through Orthogonal Subspace Projection

**Conference**: ICML 2026  
**arXiv**: [2605.08651](https://arxiv.org/abs/2605.08651)  
**Code**: Not explicitly disclosed in the paper  
**Area**: Human Understanding / Video Anomaly Detection / Privacy-Preserving Representation Learning  
**Keywords**: Video Anomaly Detection, Orthogonal Projection, Facial Suppression, Privacy-Aware, Subspace Decoupling

## TL;DR
The authors propose the Orthogonal Projection Layer (OPL) and its enhanced version G-OPL. By utilizing a learnable orthogonal subspace derived from QR decomposition, they explicitly project out "task-irrelevant variables" and "facial privacy components" from the video anomaly detection feature space. They also introduce four privacy-aware metrics (SSC/ARD/PD/FPD), demonstrating that facial prediction accuracy by linear SVM probes significantly decreases while VAD AUC is maintained or improved.

## Background & Motivation

**Background**: The dominant approach in Video Anomaly Detection (VAD) utilizes backbones such as I3D or Swin Transformer to extract spatio-temporal features, followed by weakly supervised heads like RTFM, MGFN, TEVAD, or EGO for scoring. While models grow larger and AUC improved, their deployment in surveillance and public safety scenarios inevitably captures sensitive attributes like faces, clothing, and poses within the representation.

**Limitations of Prior Work**: Existing VAD systems lack explicit mechanisms to suppress task-irrelevant or ethically sensitive information. Attackers obtaining intermediate features could potentially reverse-engineer identities. Current privacy preservation methods (e.g., INLP's nullspace projection, DAMS, CAE-LSP, OPL-2021) face several issues: (i) reliance on explicit sensitive attribute labels (VAD datasets typically lack face/identity annotations); (ii) unstable optimization due to adversarial training with gradient reversal; (iii) limitation to static images or low-dimensional scenarios; and (iv) inability to modify representations through post-hoc auditing (dataset balancing, saliency visualization).

**Key Challenge**: Privacy and utility are entangled at the representation level—simply removing facial information often discards pose/motion cues useful for anomaly detection, while adversarial training alone is unstable and uninterpretable.

**Goal**: (i) Design a differentiable module that does not rely on sensitive labels or adversarial training, allowing the filtration of specific information by simply "inserting a layer"; (ii) directionally remove facial components while retaining pose/motion without identity supervision; (iii) establish a suite of privacy evaluation metrics tailored for VAD to measure privacy, utility, and interpretability simultaneously.

**Key Insight**: The authors observe that "projecting onto the orthogonal complement of a learned low-dimensional subspace" is a geometrically clean, differentiable, and controllable way to delete information—provided the subspace learns directions carrying redundant/sensitive components, the corresponding energy can be precisely removed without affecting other directions.

**Core Idea**: Replace "adversarial suppression of sensitive attributes" with "geometric projection + weakly supervised cosine alignment." An orthogonal subspace parameterized by QR decomposition carries the sensitive components, which are then discarded by projecting features onto its orthogonal complement.

## Method

### Overall Architecture
For an intermediate feature $\bm f\in\mathbb R^d$ (from the backbone or a specific layer), OPL learns a matrix $\bm W\in\mathbb R^{k\times d}$ ($1<k<d$). QR decomposition is applied to $\bm W^\top$ to obtain $\bm W^\top=\bm Q\bm R$, where $\bm Q\in\mathbb R^{d\times k}$ provides the orthogonal basis for a $k$-dimensional nuisance subspace. The projection matrix is $\bm P=\bm I_d-\bm Q\bm Q^\top$, and the purified feature is $\bm f_{\text{proj}}=\bm P\bm f=\bm f-\bm Q\bm Q^\top\bm f$. The entire layer is differentiable and trained alongside the primary task loss. G-OPL enhances this by adding a cosine alignment loss: facial crops detected by an off-the-shelf RetinaFace detector are passed through the same encoder to obtain $\bm f_{\text{face}}$, forcing $\bm Q\bm Q^\top\bm f$ to be cosine-similar to $\bm f_{\text{face}}$, thereby actively pushing "facial directions" into the subspace to be discarded. OPL is placed in deeper layers for general nuisance removal, while G-OPL is placed immediately after the backbone to excise facial information early.

### Key Designs

1.  **OPL: Learnable Orthogonal Projection Layer via QR Decomposition**:
    - **Function**: Differentiably projects features into the orthogonal complement of a learned low-dimensional subspace, removing "task-irrelevant" components while preserving task-useful directions. It acts as a task-adaptive "feature purifier."
    - **Mechanism**: Explicitly parameterizes the subspace to be deleted as a trainable matrix $\bm W\in\mathbb R^{k\times d}$. Before each forward pass, QR decomposition on $\bm W^\top$ yields the orthogonal basis $\bm Q$, followed by projection using $\bm P=\bm I_d-\bm Q\bm Q^\top$. The process is backpropagated through the main VAD loss—subspace directions are pushed by task gradients toward directions that "do not affect detection if deleted," similar to a PCA-style geometric method but optimized for "maximum task retention + maximum projection deletion."
    - **Design Motivation**: Compared to fixed subspaces in PCA or iterative nullspace projection in INLP, QR decomposition ensures that $\bm Q$ remains a numerically stable orthogonal basis during every forward pass, avoiding gradient reversal in adversarial training and eliminating reliance on sensitive attribute labels. The structure is more interpretable than adversarial discriminators (the projected content is $\bm Q\bm Q^\top\bm f$, which can be visualized).

2.  **Guided OPL (G-OPL) + Weakly Supervised Facial Suppression via Cosine Alignment**:
    - **Function**: Pushes "facial directions" into the discarded subspace without identity labels, explicitly removing biometric components related to the face.
    - **Mechanism**: Original video frames and face crops detected by RetinaFace (averaged across faces, with 50 segments from Georgia Tech Face DB as source control) are passed through the same encoder (I3D/SwinT) to obtain $\bm f$ and $\bm f_{\text{face}}$ in the same latent space. A loss term $\mathcal L_{\text{task}}=\mathcal L_{\text{ori}}+\lambda_{\text{face}}(1-\cos(\bm f_{\text{face}},\bm Q\bm Q^\top\bm f))$ is added to the primary VAD loss, forcing $\bm Q\bm Q^\top\bm f$ to align angularly with facial embeddings—effectively *attracting* facial directions into the subspace to be projected out. This loss is only active for frames where faces are detected.
    - **Design Motivation**: The authors avoid adversarial training (unstable, requires discriminators) and explicit attribute classifiers (requires identity labels). Using geometric signals like cosine similarity as "soft labels" is both unsupervised and stable. RetinaFace provides only binary face-presence and embeddings, requiring no identity ground truth during deployment. This can be generalized to other attributes by replacing/concatenating facial embeddings with other weak signals (e.g., torso, clothing).

3.  **Orthogonality Regularization + Privacy-Aware Metric Trio (SSC/ARD/PD-FPD)**:
    - **Function**: (i) Prevents $\bm Q$ from drifting away from an orthogonal basis during training; (ii) provides quantifiable metrics to distinguish "sensitive subspace capture," "task retention," and "progressive information decay."
    - **Mechanism**: (a) Orthogonality regularization $\mathcal L_{\text{orth}}=\|\bm Q^\top\bm Q-\bm I_k\|_F^2$ is added to form the total loss $\mathcal L_{\text{total}}=\mathcal L_{\text{task}}+\lambda_{\text{orth}}\mathcal L_{\text{orth}}$. (b) Sensitive Subspace Capture $\mathrm{SSC}=\cos(\bm Q\bm Q^\top\bm f_{\text{attr}}^{(i)},\bm f_{\text{attr}}^{(i)})$ measures if the learned subspace truly captures sensitive attributes. (c) Anomaly Retention Distance $\mathrm{ARD}=\mathrm{KL}(P_{\text{raw}}(y)\|P_{\text{proj}}(y))$, where KDE estimates the KL divergence between anomaly score distributions before and after projection; lower values indicate better utility retention. (d) Privacy Decay $\{(l,\mathrm{Acc}^{(l)})\}_{l=1}^L$ uses linear SVM probes after each G-OPL to predict face presence; lower accuracy indicates stronger suppression. First-layer PD (FPD) specifically refers to accuracy after the first G-OPL.
    - **Design Motivation**: QR decomposition yields an orthogonal basis per forward pass, but gradient updates can destroy orthogonality, especially when stacking multiple G-OPL layers. The SSC/ARD/PD trio fills the gap in VAD privacy evaluation, as prior work focused solely on AUC.

### Loss & Training
The total loss is $\mathcal L_{\text{total}}=\mathcal L_{\text{ori}}+\lambda_{\text{face}}(1-\cos(\bm f_{\text{face}},\bm Q\bm Q^\top\bm f))+\lambda_{\text{orth}}\|\bm Q^\top\bm Q-\bm I_k\|_F^2$, where $\mathcal L_{\text{ori}}$ is the original loss of the integrated weakly supervised VAD baseline (e.g., RTFM, MGFN, TEVAD, or EGO). The facial alignment term is only active when faces are detected, automatically skipped for other frames.

## Key Experimental Results

### Main Results
OPL/G-OPL were integrated into 4 SOTA baselines (RTFM, MGFN, TEVAD, EGO) using I3D / Swin Transformer features across 5 VAD datasets (ShanghaiTech, UCF-Crime, CUHK Avenue, UCSD Ped2, MSAD). Decoupling ablation on ShanghaiTech (AUC %):

| $k_{\text{OPL}}\backslash k_{\text{G-OPL}}$ | $2$ | $4$ | $8$ | $16$ | $32$ | $64$ | $128$ |
|---|---|---|---|---|---|---|---|
| $2$ | $95.5$ | $95.9$ | $95.6$ | $94.8$ | $93.9$ | $92.5$ | $91.8$ |
| $4$ | $95.9$ | $\mathbf{97.3}$ | $96.8$ | $95.2$ | $94.0$ | $92.8$ | $91.9$ |
| $8$ | $95.7$ | $97.0$ | $96.5$ | $95.0$ | $93.8$ | $92.6$ | $91.7$ |
| $32$ | $94.5$ | $95.1$ | $94.6$ | $93.8$ | $92.8$ | $91.5$ | $90.8$ |
| $128$ | $92.8$ | $93.1$ | $92.4$ | $91.5$ | $89.8$ | $88.4$ | $87.9$ |

At optimal $k_{\text{OPL}}=k_{\text{G-OPL}}=4$, the AUC reaches $97.3\%$, significantly higher than the RTFM baseline (approx. $97.0\%$ with I3D on ShT). Excessive $k$ (e.g., $128$) causes over-deletion of information, dropping AUC to $87.9\%$.

Multi-anomaly type comparison on MSAD (Table 2 excerpt, AUC %):

| Method | Assault | Explosion | Fighting | Robbery | Shooting | Traffic Acc. | Overall AUC |
|------|---------|-----------|----------|---------|----------|--------------|-------------|
| RTFM (I3D) baseline | $53.9$ | $66.0$ | $79.8$ | ... | ... | ... | ... |
| + OPL / G-OPL (Ours) | Comprehensive gain/parity | — | — | — | — | — | — |

(Specific gains for OPL/G-OPL show the trend of maintaining or slightly increasing utility while significantly reducing privacy metrics.)

### Ablation Study

| Configuration | Key Metrics | Description |
|------|---------|------|
| Baseline RTFM (I3D) | High AUC, but FPD close to baseline backbone | No privacy mechanism in place. |
| + OPL | AUC stable/slightly higher; UMAP shows nuisance clusters dispersing. | General nuisance components successfully removed. |
| + G-OPL (Cosine Alignment) | FPD drops sharply, SSC increases significantly, AUC maintained. | Facial components directionally absorbed into the subspace and discarded. |
| W/O $\mathcal L_{\text{orth}}$ | $\bm Q$ deviates from orthogonality after a few epochs; AUC fluctuates. | Orthogonal basis must be explicitly preserved. |
| $k$ too large ($\ge 64$) | Sharp decrease in AUC. | Discarded subspace is too wide, losing useful information. |

### Key Findings
- Placing G-OPL immediately after the backbone (early stage) is more effective than in deeper layers—FPD (face prediction accuracy by linear SVM after the first layer) is lowest, indicating privacy should be intercepted before information diffuses through the network.
- ARD (KL divergence) increases monotonically with $k$, but AUC peaks at $k=4$, indicating that utility does not change monotonically with $k$. Removing a small amount of nuisance can reduce overfitting and improve discriminative power.
- Using ArcFace as an attacker-style rank-1 re-identification probe, retrieval accuracy significantly decreases after G-OPL, validating that fine-grained identity is suppressed alongside coarse facial presence.

## Highlights & Insights
- "Geometric projection in place of adversarial training" is a paradigm worth adopting—it removes sensitive components with stable training, interpretability (visualizing $\bm Q\bm Q^\top\bm f$), and no need for identity labels. This is applicable to face anti-spoofing, human pose estimation, and medical imaging.
- Using cosine alignment as a *weakly supervised* signal is ingenious—as long as an off-the-shelf detector provides "concept vectors" (face or clothing embeddings), "what is sensitive" can be injected into $\bm Q$ geometrically and plugged into any VAD baseline.
- The SSC/ARD/PD-FPD trio fills the gap in VAD privacy evaluation and can be directly applied to other vision tasks like action recognition or re-ID defense/privacy.

## Limitations & Future Work
- Relying on RetinaFace as the sole face-weak-supervision source means G-OPL fails when faces are small, occluded, or low-resolution; the authors admit MSAD results rely on pre-extracted features since faces are already blurred.
- Current G-OPL targets only "facial" sensitive attributes. Simultaneous suppression of multiple attributes (face + clothing + gait) sharing one $\bm Q$ might lead to interference and requires more refined alignment designs.
- $k$ is a critical hyperparameter that requires per-dataset tuning; while $k=4$ worked for ShanghaiTech, cross-dataset generalization remains under-explored.
- Robustness against *adaptive attackers* (who know OPL exists and train specialized inversion networks) is not evaluated; current conclusions apply to *non-adaptive* SVM/ArcFace probes.

## Related Work & Insights
- **vs INLP / Nullspace Projection (Ravfogel 2020)**: INLP uses iterative nullspace erasure but requires attribute ground truth. G-OPL uses cosine weak supervision + differentiable QR for one-stage end-to-end training.
- **vs OPL-2021 (Ranasinghe 2021)**: Prior work used generic decorrelation losses without the concept of "sensitive directions." This work adds face cosine alignment to make the subspace controllable and directional.
- **vs Adversarial Training + Gradient Reversal (Ganin et al.)**: Eliminates the instability and the need to train discriminators by using geometric methods directly.
- **vs Data-level Privacy (Facial Blurring, DP-SGD)**: Data-level methods require pipeline overhauls; G-OPL is a model-level plug-in that does not alter raw data.

## Rating
- Novelty: ⭐⭐⭐⭐ Combining "differentiable QR orthogonal projection + cosine weak supervision" for VAD is a first; the trio of privacy metrics is also a new contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ 5 datasets × 4 baselines × multiple $k$ configurations + ArcFace inversion attacker validation provides broad coverage.
- Writing Quality: ⭐⭐⭐⭐ Clear logical progression from motivation → OPL → G-OPL → metrics → experiments. Appendices provide theoretical and operational details.
- Value: ⭐⭐⭐⭐ Engineering-friendly (plug-in module, immutable backbone), with no extra overhead during inference once $\bm Q$ is fixed. Highly practical for deployment in surveillance cameras.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] VSCD: Video Scene Change Detection in Unaligned Scenarios](vscd_video-based_scene_change_detection_in_unaligned_scenes.md)
- [\[AAAI 2026\] StegaVAR: Privacy-Preserving Video Action Recognition via Steganographic Domain Analysis](../../AAAI2026/video_understanding/stegavar_privacy-preserving_video_action_recognition_via_steganographic_domain_a.md)
- [\[AAAI 2026\] Balancing Multimodal Domain Generalization via Gradient Modulation and Projection](../../AAAI2026/video_understanding/balancing_multimodal_domain_generalization_via_gradient_modulation_and_projectio.md)
- [\[ICCV 2025\] Aligning Effective Tokens with Video Anomaly in Large Language Models](../../ICCV2025/video_understanding/aligning_effective_tokens_with_video_anomaly_in_large_language_models.md)
- [\[AAAI 2026\] Group Orthogonal Low-Rank Adaptation for RGB-T Tracking](../../AAAI2026/video_understanding/group_orthogonal_low-rank_adaptation_for_rgb-t_tracking.md)

</div>

<!-- RELATED:END -->
