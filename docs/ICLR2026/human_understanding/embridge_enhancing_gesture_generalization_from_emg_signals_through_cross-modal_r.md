---
title: >-
  [Paper Note] EMBridge: Enhancing Gesture Generalization from EMG Signals Through Cross-modal Representation Learning
description: >-
  [ICLR 2026][Human Understanding][Surface Electromyography (sEMG)] EMBridge proposes using hand poses as high-quality anchors. Through a triple mechanism of Q-Former, Masked Pose Reconstruction Loss (MPRL), and Community-Aware Soft Contrastive Learning (CASCLe), it aligns the representation space of noisy sEMG signals with a semantically structured pose space, achieving zero-shot EMG gesture classification on wearable devices for the first time.
tags:
  - "ICLR 2026"
  - "Human Understanding"
  - "Surface Electromyography (sEMG)"
  - "Cross-modal Representation Learning"
  - "Gesture Recognition"
  - "Zero-shot Generalization"
  - "Q-Former"
date: 2026-05-08
content_hash: b04b350ebf2ab428
---

# EMBridge: Enhancing Gesture Generalization from EMG Signals Through Cross-modal Representation Learning

**Conference**: ICLR 2026  
**Code**: None  
**Area**: Human Understanding / Gesture Recognition / Wearable Devices  
**Keywords**: Surface Electromyography (sEMG), Cross-modal Representation Learning, Gesture Recognition, Zero-shot Generalization, Q-Former

## TL;DR
EMBridge proposes using hand poses as high-quality anchors. Through a triple mechanism of Q-Former, Masked Pose Reconstruction Loss (MPRL), and Community-Aware Soft Contrastive Learning (CASCLe), it aligns the representation space of noisy sEMG signals with a semantically structured pose space, achieving zero-shot EMG gesture classification on wearable devices for the first time.

## Background & Motivation

**Background**: Gesture recognition is widely demanded in scenarios such as rehabilitation, human-computer interaction, and prosthetic control. Vision-based solutions (video/image/skeleton) are mature but suffer from high power consumption, privacy risks, and instability during occlusion. Surface electromyography (sEMG) offers low power consumption, continuous acquisition, and suitability for integration into wristband-type wearables, making it a promising alternative.

**Limitations of Prior Work**: sEMG signals are inherently noisy with significant individual differences, and publicly available paired datasets are limited in scale. Representation spaces learned solely through EMG self-supervised pre-training (e.g., MAE) exhibit chaotic semantic structures and poor inter-class discriminability. The authors visualize this by contrasting the "scattered cloud" state of EMG embeddings after MAE pre-training with the "clear clusters" of pose embeddings trained similarly.

**Key Challenge**: Learning discriminative features purely from EMG signals is extremely difficult, whereas high-quality hand pose data (motion capture) provides rich semantic supervision. However, during inference, only EMG signals are available without pose data. This raises the question: how to utilize poses during training while relying solely on EMG during inference?

**Goal**: To use poses as a "teacher" to guide EMG representation learning during the pre-training phase, ultimately enabling the EMG encoder to recognize unseen gestures during testing based only on the signal itself (zero-shot generalization).

**Key Insight**: Cross-modal representation alignment—freezing a high-quality pose encoder as a fixed anchor and optimizing only the EMG encoder (asymmetric design) avoids "pulling down" the structured pose space to the noise level of the EMG.

**Core Idea**: Utilizing a frozen pose encoder as an anchor, EMBridge drives the alignment of EMG representations toward the pose semantic space via Q-Former, masked reconstruction, and community-aware soft contrastive objectives to achieve zero-shot EMG gesture recognition.

## Method

### Overall Architecture

EMBridge adopts a two-stage design. In the first stage, MAE is used for uni-modal pre-training of the EMG encoder $E_x$ and the pose encoder $E_p$ to obtain high-quality initial representations. In the second stage, $E_p^*$ is frozen as a fixed anchor, and a Q-Former is attached. Three joint optimization objectives are used to pull EMG representations toward the pose semantic space. At inference time, only $E_x$ is required, with no pose data needed.

```mermaid
flowchart TD
    A[sEMG Sequence x] --> B[EMG Encoder E_x\nLearnable, MAE Pre-trained]
    B --> C[Q-Former F_ϕ\n4×self-attn + 2×cross-attn\nInit from Pose-MAE]
    C --> D[Query Embeddings Q' ∈ R^{M×d}]
    
    E[Hand Pose Sequence p] --> F[Pose Encoder E_p*\nFrozen, MAE Pre-trained]
    F --> G[Pose Embedding v ∈ R^d]
    
    D -- InfoNCE --> H[Instance-level Alignment]
    D & G -- CASCLe --> I[Community-level Soft Contrastive Alignment]
    D & E -- MPRL --> J[Masked Pose Reconstruction]
    
    H & I & J --> K[Total Loss L]
```

### Key Designs

**1. Q-Former Asymmetric Alignment: Extracting Pose-related EMG Features**

Standard CLIP/BLIP models update both encoders symmetrically, which can "contaminate" the representation space of the high-quality modality. EMBridge chooses to freeze the pose encoder $E_p^*$ and optimize only the Q-Former $F_\phi$ and $E_x$ on the EMG side. The Q-Former maintains $M$ learnable queries $Q^{(0)} \in \mathbb{R}^{M \times d}$. Through 4 layers of self-attention blocks (with cross-attention layers inserted every other layer), it extracts pose-related information from the EMG encoder's output, producing updated queries $Q' \in \mathbb{R}^{M \times d}$. The self-attention layers are initialized by the pre-trained Pose-MAE, granting the queries an inherent ability to understand pose semantics; cross-attention layers are randomly initialized to learn how to "query" pose information from EMG features. The InfoNCE objective drives the optimal query $u_i$ for each sample $i$ (the one with the highest cosine similarity to the corresponding pose embedding $v_i$) closer to $v_i$ and further from other intra-batch samples:

$$L_{\text{InfoNCE}} = -\frac{1}{B}\sum_{i=1}^{B}\sum_{j=1}^{B} I_{ij} \log \frac{\exp(u_i^\top v_j / \tau)}{\sum_{k=1}^{B} \exp(u_i^\top v_k / \tau)}$$

Since gradients only exist on the EMG side, the pose representation space maintains its superior semantic structure, and the EMG encoder is "lifted" unidirectionally.

**2. Masked Pose Reconstruction Loss (MPRL): Forcing Queries to Carry Structured Pose Semantics**

Relying solely on contrastive loss might result in queries aligning with overall semantics while ignoring fine-grained pose structures. MPRL requires the queries to reconstruct masked pose tokens without direct access to EMG features. Specifically, $Q'$ is obtained in a first forward pass. In a second pass, masked pose tokens $\tilde{P}$ are concatenated with $Q'$ and fed into the Q-Former's self-attention layers (an attention mask ensures pose tokens cannot access EMG features via cross-attention and must retrieve information from $Q'$). The reconstruction loss is defined as:

$$L_{\text{MPRL}} = \frac{1}{|\mathcal{M}|}\sum_{m \in \mathcal{M}} \left\| g\left(H_P[m]\right) - P[m] \right\|_2^2$$

This "forced dependency" mechanism ensures that queries actively extract and encode latent pose information from the EMG output, thereby enriching EMG embeddings with pose semantics and aiding generalization to unseen poses.

**3. Community-Aware Soft Contrastive Learning (CASCLe): Aligning Relative Geometric Structures in Latent Space**

Standard InfoNCE treats all non-matching intra-batch samples as equivalent "negatives." However, the pose space is continuous—poses of two different gestures might be spatially very close, and forcing them apart creates harmful gradients and model confusion. CASCLe replaces hard one-hot targets with community-level soft targets. Offline $k$-means is performed on Pose-MAE embeddings to obtain $N_c$ centroids $C$. For each pose embedding in a batch, affinity vectors with centroids $S_{p,c} = PC^\top$ are calculated and sparsified to retain the top-$k_c$ nearest centroids (filtering out irrelevant communities). A community-aware pose-pose similarity matrix $S_{p,p} = S_{p,c} S_{p,c}^\top$ is then computed via outer product. After removing the diagonal and applying softmax normalization, soft targets $\tilde{y}_{ij}$ are obtained, representing the "probability that pose $v_j$ is a semantic neighbor of $v_i$ in a fixed pose relationship graph." CASCLe minimizes the cross-entropy between the EMG-pose similarity distribution and the soft targets:

$$L_{\text{CASCLe}} = -\frac{1}{B}\sum_{i=1}^{B}\sum_{j \neq i}^{B} \tilde{y}_{ij} \log q_{ij}$$

Compared to SoftCLIP (based on instance-level similarity) and label smoothing, CASCLe utilizes more stable clustering structure information, performing better in zero-shot scenarios—ablation studies show that replacing InfoNCE with CASCLe improves ZS unseen performance from 0.511 to 0.528.

## Key Experimental Results

### Main Results (emg2pose dataset, Balanced Accuracy)

| Method | LP Seen | ZS Seen | LP Unseen | ZS Unseen |
|------|------------|------------|------------|------------|
| EMG-MAE (Uni-modal Baseline) | 0.347 | — | 0.334 | — |
| emg2pose (Supervised Baseline) | 0.734 | — | 0.405 | — |
| CPEP (Symmetric Contrastive) | 0.782 | 0.757 | 0.536 | 0.481 |
| Q-Former (w/o MPRL/CASCLe) | 0.782 | 0.763 | 0.493 | 0.498 |
| **EMBridge** | **0.785** | **0.777** | 0.505 | **0.528** |
| Upper Bound (Pose Encoder LP) | 0.851 | — | 0.649 | — |

On NinaPro, EMBridge achieved 0.692 / 0.447 for ZS Seen/Unseen gestures, a significant improvement over CPEP (0.604 / 0.413).

### Ablation Study (emg2pose ZS Unseen gestures)

| Configuration | LP Seen | ZS Seen | LP Unseen | ZS Unseen |
|------|---------|---------|---------|---------|
| EMBridge w/o Q-Former | 0.793 | 0.763 | 0.538 | 0.494 |
| EMBridge w/o MPRL | 0.783 | 0.764 | 0.494 | 0.516 |
| EMBridge w/o CASCLe | 0.784 | 0.764 | 0.485 | 0.509 |
| Label Smoothing replacing CASCLe | 0.777 | 0.759 | 0.489 | 0.511 |
| SoftCLIP replacing CASCLe | 0.788 | 0.760 | 0.490 | 0.510 |
| **EMBridge (Full)** | **0.785** | **0.777** | **0.505** | **0.528** |

### Key Findings
- EMBridge's ZS Seen performance (0.777) exceeds the LP performance of all uni-modal baselines (max 0.734), indicating that cross-modal alignment indeed enhances the discriminative power of EMG representations.
- Even with only 40% of paired pre-training data, EMBridge's zero-shot performance still surpasses uni-modal baselines trained on the full dataset, highlighting its data efficiency.
- Regarding per-person ZS performance on unseen users, EMBridge improves F1 by an average of 16.0% compared to CPEP, demonstrating robustness to individual variances.

## Highlights & Insights
- **Necessity of Asymmetric Design**: Freezing the high-quality modality encoder as a fixed anchor is a critical design choice. In symmetric training, gradients from noisy EMG would degrade the semantic structure of the pose space. Simultaneously, a fixed pose encoder allows for independent pre-training using large volumes of unpaired pose data, which can significantly enhance supervision quality in the future without requiring more paired data.
- **Effectiveness of Community-Aware Soft Targets**: The continuity of the hand pose space makes hard negative penalties harmful. CASCLe finds natural semantic neighborhoods through clustering, which is more stable than instance-level similarity (SoftCLIP) and consistently outperforms other soft target solutions in zero-shot generalization.
- **Q-Former Trade-off**: Q-Former maximizes generalization in zero-shot settings (due to its flexible query mechanism), but its linear probing performance is slightly lower than CPEP (which uses the CLS token directly). This is a common trade-off between "representation flexibility vs. feature determinism."

## Limitations & Future Work
- The framework relies on paired EMG-pose data for pre-training, but the scarcity of high-quality paired datasets remains a practical bottleneck. Future work could explore pre-training pose encoders with large-scale unpaired pose data and then performing EMBridge alignment with small amounts of paired data.
- Currently, only the EMG-pose modality combination has been explored. Extending alignment to RGB-EMG or Video-EMG (utilizing pre-trained vision encoders) is a natural progression to further improve the quality of supervision signals.
- Pose community modeling currently uses hard $k$-means. Future iterations could use Gaussian Mixture Models (GMM) to introduce soft probabilistic community membership, making the calculation of structural similarity more continuous and smooth.

## Related Work & Insights
- **vs. CLIP/BLIP-2**: CLIP symmetrically aligns two large-scale encoders, requiring massive amounts of paired data. EMBridge employs an asymmetric Q-Former design, achieving cross-modal alignment with high-quality uni-modal pre-training and limited paired data, which is better suited for the small-data scenarios common in biosignals.
- **vs. CPEP (Prior Work)**: CPEP uses a projection layer + InfoNCE for simple alignment, failing to exploit the multi-scale temporal structure within EMG. EMBridge's Q-Former can selectively extract pose-relevant features from EMG via multi-head cross-attention, leading to stronger generalization.
- **vs. SoftCLIP / Label Smoothing**: SoftCLIP uses instance-level similarity as soft targets, while CASCLe utilizes clustered community structures. The latter leverages a more globally stable semantic topology, making it more robust to noise.

## Rating
- Novelty: ⭐⭐⭐⭐ First to jointly apply Q-Former + masked reconstruction + community-level soft contrastive learning to EMG cross-modal alignment, with clear motivation for the asymmetric architecture.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers two datasets, multiple evaluation protocols (ZS/LP), detailed ablations, and hyperparameter sensitivity analysis. Data efficiency experiments add practical conviction.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, complete derivation of methodological motivation, and good alignment between figures and text.
- Value: ⭐⭐⭐⭐ Zero-shot EMG gesture recognition has clear application scenarios in VR/AR and prosthetic control. The methodological framework provides a reference for other biosignal cross-modal research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] CPEP: Contrastive Pose-EMG Pre-training Enhances Gesture Generalization on EMG Signals](../../NeurIPS2025/human_understanding/cpep_contrastive_pose-emg_pre-training_enhances_gesture_generalization_on_emg_si.md)
- [\[ICLR 2026\] From Pixels to Semantics: Unified Facial Action Representation Learning for Micro-Expression Analysis](from_pixels_to_semantics_unified_facial_action_representation_learning_for_micro.md)
- [\[AAAI 2026\] New Synthetic Goldmine: Hand Joint Angle-Driven EMG Data Generation Framework for Micro-Gesture Recognition](../../AAAI2026/human_understanding/new_synthetic_goldmine_hand_joint_angle-driven_emg_data_generation_framework_for.md)
- [\[ICLR 2026\] SesaHand: Enhancing 3D Hand Reconstruction via Controllable Generation with Semantic and Structural Alignment](sesahand_enhancing_3d_hand_reconstruction_via_controllable_generation_with_seman.md)
- [\[ICLR 2026\] Cross-Domain Policy Optimization via Bellman Consistency and Hybrid Critics](cross-domain_policy_optimization_via_bellman_consistency_and_hybrid_critics.md)

</div>

<!-- RELATED:END -->
