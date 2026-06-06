---
title: >-
  [Paper Note] Divide and Conquer: Reliable Multi-View Evidential Learning for Deepfake Detection
description: >-
  [ICML 2026][Image Generation][Deepfake Detection] This paper proposes the DiCoME framework, which utilizes geometric orthogonal projection to explicitly decouple CLIP semantic features and forgery artifact features into…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "Deepfake Detection"
  - "Multi-View Learning"
  - "Evidential Learning"
  - "Geometric Orthogonal Decomposition"
  - "Uncertainty Quantification"
date: 2026-05-08
content_hash: 70a77ac8adecabf6
---

# Divide and Conquer: Reliable Multi-View Evidential Learning for Deepfake Detection

**Conference**: ICML 2026  
**arXiv**: [2606.01885](https://arxiv.org/abs/2606.01885)  
**Code**: https://github.com/kxl0825/DiCoME.git (Available)  
**Area**: AI Security / Deepfake Detection  
**Keywords**: Deepfake Detection, Multi-View Learning, Evidential Learning, Geometric Orthogonal Decomposition, Uncertainty Quantification

## TL;DR
This paper proposes the DiCoME framework, which utilizes geometric orthogonal projection to explicitly decouple CLIP semantic features and forgery artifact features into two complementary "expert views." It then employs Dempster–Shafer evidential fusion to model "epistemic conflict" between these views, outputting reliable uncertainty. DiCoME improves average AUC from 0.923 backlines to 0.939 (cross-dataset) and 0.976 (cross-manipulation) on deepfake detection benchmarks.

## Background & Motivation

**Background**: With the advancement of GANs and diffusion models, deepfakes have become nearly indistinguishable from real content at the semantic level; forensics must rely on extremely subtle structural anomalies. Current mainstream solutions follow a single-view holistic paradigm, using a backbone (increasingly vision foundation models like CLIP) to encode both semantic content and forgery traces into entangled representations, followed by Softmax binary classification.

**Limitations of Prior Work**: The authors identify the failure mode of this paradigm as the **Semantic Masking Effect**. Since vision backbones are inherently optimized for semantic invariance, high-amplitude identity features dominate the feature space ($\|f_s\| \gg \|f_a\|$), causing weak artifact signals to be ignored as shortcut noise during gradient descent. Consequently, models become overconfident on seen manipulations but fail on unseen methods or cross-domain data, with Softmax disguising these failures as high-confidence predictions.

**Key Challenge**: The essence of deepfake detection is "finding subtle anomalies against a strong semantic background." However, conventional backbones entangle semantic and artifact dimensions. Feature-level fusion (e.g., RGB+Frequency, Global+Local) merely concatenates two highly correlated views. The "Structure Expert" often redundantizes the "Content Expert," lacking true complementarity and failing to provide calibrated uncertainty during conflicts. Prior efforts (e.g., ICML'25) attempted SVD-based orthogonalization but relied on global low-rank assumptions, which are insensitive to sample-level artifacts.

**Goal**: (1) Completely strip artifacts from semantics at the feature level to obtain two decorrelated and complementary pieces of evidence; (2) Explicitly model conflicts between "expert opinions" as high uncertainty at the decision level, rather than being suppressed into a false high-confidence prediction by Softmax.

**Key Insight**: Artifacts are strictly defined as the "orthogonal complement of the semantic subspace." This geometric hard constraint is more thorough than soft regularization (e.g., contrastive learning). Furthermore, Subjective Logic is used to parameterize the output of each view as a Dirichlet distribution, allowing the "conflict coefficient" to be calculated via Dempster–Shafer theory.

**Core Idea**: Use geometric orthogonal projection to subtract the semantic leakage from the CLIP semantic residual to obtain a pure artifact view. Evidential fusion is then used to let view conflicts directly increase epistemic uncertainty, allowing the model to "admit ignorance" when facing unknown attacks.

## Method

### Overall Architecture
Inputting a face image, DiCoME follows a two-stage pipeline:

1.  **Divide (Multi-View Construction)**: First, semantic features $f_s$ are extracted using a LoRA-fine-tuned CLIP ViT-L/14. Then, a lightweight $\beta$-VAE projects $f_s$ onto the "semantic manifold" in the feature space (rather than pixel space) to obtain a reconstruction $f_c$. The residual $f_r = f_s - f_c$ represents the raw anomaly signal but remains contaminated with semantic leakage. Finally, **geometric orthogonal projection** is applied to subtract the component of $f_r$ along the $f_s$ direction, resulting in pure artifact features $f_a$.
2.  **Conquer (Evidential Fusion Decision)**: Both $f_s$ and $f_a$ are fed into separate evidential heads (MLP+Softplus) to generate Dirichlet parameters $\alpha^v$, deriving subjective opinions $T^v = \{b^v, u^v\}$ (belief + uncertainty). The Dempster–Shafer orthogonal sum $T^1 \oplus T^2$ fuses these into global belief and uncertainty. During inference, $p_k = b_k + u/K$ serves as the class probability, and $u$ as the risk rejection score.

### Key Designs

1.  **Geometric View Purification**:
    *   **Function**: Hard-strips "semantic leakage masquerading as anomalies" from the CLIP semantic residual $f_r$, obtaining pure artifact features $f_a$ strictly orthogonal to $f_s$.
    *   **Mechanism**: A $\beta$-VAE reconstructs a "semantic manifold version" $f_c$ in the CLIP feature space (using cosine distance for reconstruction loss as CLIP embeddings are spherical). Given the residual $f_r = f_s - f_c$, the component of $f_r$ along $f_s$ is calculated via $f_r^{\parallel} = \frac{\langle f_r, f_s \rangle}{\|f_s\|_2^2} f_s$. Subtracting this yields $f_a = f_r - f_r^{\parallel}$, ensuring $\langle f_a, f_s \rangle = 0$, meaning $f_a$ resides in the orthogonal complement $S^{\perp}$ of the semantic subspace.
    *   **Design Motivation**: Previous multi-view methods (e.g., attention-guided FDML, global SVD in Effort) utilize soft or coarse constraints that cannot guarantee decorrelation on a per-sample basis. Defining "artifacts = semantic orthogonal complement" as a hard geometric constraint provides the "Structure Expert" a feature space immune to semantic interference, addressing the Semantic Masking Effect at its root.

2.  **Evidential Opinion Generation**:
    *   **Function**: Converts features from each view into subjective opinions over Dirichlet distributions, enabling the model to provide both class probabilities and epistemic uncertainty.
    *   **Mechanism**: For view $v \in \{1, 2\}$, the evidential head outputs a non-negative evidence vector $e^v = \text{Softplus}(W^v f_v + C^v)$. Dirichlet parameters are $\alpha_k^v = e_k^v + 1$, with total strength $S^v = \sum_k \alpha_k^v$. This derives belief $b_k^v = e_k^v / S^v$ and uncertainty $u^v = K/S^v$, satisfying $\sum_k b_k^v + u^v = 1$. When a view lacks discriminative evidence ($e_k \to 0$), $u^v \to 1$. Training uses an evidential loss $\mathcal{L}_{fit}$ in Bayes risk form, with KL regularization to push evidence for non-target classes toward a uniform prior.
    *   **Design Motivation**: Softmax compresses different types of uncertainty into a single confidence score, leading to overconfidence on unseen attacks. Dirichlet parameterization explicitly encodes "how much evidence" into $S^v$, providing mathematically well-defined belief/uncertainty for fusion.

3.  **Conflict-Aware Evidential Fusion**:
    *   **Function**: Combines opinions from both experts using Dempster–Shafer aggregation rules, explicitly converting inter-view conflicts into uncertainty.
    *   **Mechanism**: The fusion formula is $b_k = \frac{1}{1-\mathcal{C}}(b_k^1 b_k^2 + b_k^1 u^2 + b_k^2 u^1)$ and $u = \frac{u^1 u^2}{1-\mathcal{C}}$, where the conflict coefficient $\mathcal{C} = \sum_{i \ne j} b_i^1 b_j^2$ quantifies the disagreement. This handles three scenarios: (a) **Consensus**: Both are confident and agree, increasing global belief while $u$ decays; (b) **Complementarity**: One is confident while the other is ignorant, allowing the confident view to "vote" via the cross-term; (c) **Conflict**: Views disagree, significantly increasing $\mathcal{C}$, which suppresses belief and raises $u$ as a rejection signal.
    *   **Design Motivation**: Standard mean or concatenation fusion "averages out" conflict, making models more confident on hard samples. DS fusion embeds logic for handling evidence conflict, perfectly leveraging the decorrelated views.

### Loss & Training
Total loss: $\mathcal{L}_{total} = \mathcal{L}_{edl} + \lambda_{vae} \mathcal{L}_{vae}$, where $\mathcal{L}_{vae} = \mathcal{L}_{align} + \beta \mathcal{L}_{kld}$. $\mathcal{L}_{align} = 1 - \cos(f_s, f_c)$ forces the $\beta$-VAE to align CLIP features in direction rather than magnitude. The backbone is CLIP ViT-L/14 with LoRA; trained only on FaceForensics++ c23 and zero-shot transferred to all evaluation sets.

## Key Experimental Results

### Main Results
Comparison with 12 baselines across 6 cross-datasets and 8 cross-manipulations (DF40 subset).

| Setting | Metric | DiCoME | Effort (ICML'25) | GenD (WACV'26) | ProDet (NIPS'24) |
|----------|------|--------|------------------|----------------|------------------|
| Cross-dataset Avg AUC | video AUC | **0.939** | 0.907 | 0.923 | 0.821 |
| Cross-manipulation Avg AUC | video AUC | **0.976** | 0.940 | 0.958 | 0.867 |
| CDFv2 | video AUC | **0.977** | 0.956 | 0.960 | 0.926 |
| DFDC | video AUC | **0.882** | 0.843 | 0.871 | 0.707 |

DiCoME outperforms the runner-up GenD by 1.6 points on average cross-dataset and 1.8 points on cross-manipulation. The improvement on the challenging DFDC dataset is nearly 4 points.

### Ablation Study

| Config | CDFv2 | DFDC | MFS | Avg |
|------|-------|------|-----|-----|
| (A) $f_s$ only (Single-view CLIP) | 0.927 | 0.856 | 0.933 | 0.905 |
| (C) $f_s + f_r$ (Raw residual, no orthogonalization) | 0.956 | 0.860 | 0.951 | 0.922 |
| **Ours $f_s + f_a$ (Orthogonal Purification)** | **0.977** | **0.882** | **0.956** | **0.938** |
| (D) $\mathcal{L}_{ce}$ + Mean Fusion | 0.956 | 0.874 | 0.942 | 0.924 |
| **Ours $\mathcal{L}_{edl}$ + DS-Conflict** | **0.977** | **0.882** | **0.956** | **0.938** |

### Key Findings
- **Orthogonal projection is the primary contributor**: Moving from raw residual (C) to purified $f_a$ (Ours) improves average AUC by 1.6, confirming that raw residuals suffer from heavy semantic leakage.
- **DS fusion outperforms Mean fusion**: Switching from Mean to DS-Conflict adds 1.1 points, indicating that the gain stems from explicit conflict modeling.
- **Angular alignment > Magnitude alignment**: Removing $\mathcal{L}_{align}$ causes a larger drop than removing KL, validating that CLIP spherical embeddings store information in directions.
- **Uncertainty is actionable**: Risk-coverage curves show that rejecting the 10% most uncertain samples in DFD significantly boosts accuracy.

## Highlights & Insights
- **Defining "Artifacts" as the Semantic Orthogonal Complement**: This is a clean inductive bias more thorough than attention or contrastive decoupling. It applies per-sample via a single vector projection line.
- **Conflict as Uncertainty**: DS fusion's strength lies in converting disagreement between experts into a mathematically well-defined uncertainty source ($\mathcal{C}$), which is highly effective for detecting unseen attacks.
- **Feature-space VAE Reconstruction for Foundation Models**: Reconstructing in the CLIP feature space avoids the difficulties of pixel-space VAEs while leveraging robust semantic priors. This paradigm is easily transferable to other tasks like OOD or anomaly detection.
- **Single-Domain Training**: All results are obtained via zero-shot transfer from FF++ c23 training, reflecting true generalization capability in open-world scenarios.

## Limitations & Future Work
- **Orthogonalization based on a single direction $f_s$**: Collapsing the entire semantic subspace into one vector is a coarse approximation of complex factors like identity and lighting.
- **Dependency on $\beta$-VAE**: If $f_c$ reconstruction is poor, non-semantic noise might be incorrectly labeled as artifacts in $f_a$.
- **Task Specificity**: Validated only on facial deepfakes; applicability to full-body deepfakes or temporal artifacts is untested.
- **Binary Limitation**: In binary classification, Dirichlet degrades into a Beta distribution; the advantages of conflict sensing would be more pronounced in multi-class forgery source identification.

## Related Work & Insights
- **vs Effort (ICML'25)**: Both isolate artifacts via orthogonalization, but Effort uses global SVD which lacks sample-level granularity. DiCoME's per-sample projection and evidential uncertainty improve cross-dataset Avg from 0.907 to 0.939.
- **vs RECCE**: RECCE performs reconstruction in pixel space, which is difficult for high-res faces. DiCoME performs reconstruction in CLIP feature space, inheriting semantic robustness.
- **vs Standard EDL**: While standard EDL is a single-view uncertainty tool, this work upgrades it to a "multi-view evidence conflict arbitrator."

## Rating
- Novelty: ⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] From Talking to Singing: A New Challenge for Audio-Visual Deepfake Detection](from_talking_to_singing_a_new_challenge_for_audio-visual_deepfake_detection.md)
- [\[ICML 2026\] ViewMask-1-to-3: Multi-View Consistent Image Generation via Multimodal Discrete Diffusion Models](viewmask-1-to-3_multi-view_consistent_image_generation_via_multimodal_discrete_d.md)
- [\[ICCV 2025\] DeepShield: Fortifying Deepfake Video Detection with Local and Global Forgery Analysis](../../ICCV2025/image_generation/deepshield_fortifying_deepfake_video_detection_with_local_and_global_forgery_ana.md)
- [\[ICML 2026\] Gradient Preconditioning for Efficient and Reliable Reward-Guided Generation](gradient_preconditioning_for_efficient_and_reliable_reward-guided_generation.md)
- [\[ICML 2026\] Offline Multi-agent Reinforcement Learning via Sequential Score Decomposition](offline_multi-agent_reinforcement_learning_via_sequential_score_decomposition.md)

</div>

<!-- RELATED:END -->
