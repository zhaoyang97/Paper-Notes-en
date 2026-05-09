---
title: >-
  [Paper Note] Plug-and-Play Parameter-Efficient Tuning of Embeddings for Federated Recommendation
description: >-
  [AAAI 2026][AI Safety][Federated Recommendation] This paper proposes a plug-and-play federated recommendation framework that introduces PEFT (Parameter-Efficient Fine-Tuning) concepts into item embeddings. By freezing pre-trained full embeddings and transmitting only lightweight compressed embeddings (LoRA / Hash / RQ-VAE), the framework significantly reduces communication overhead while improving recommendation accuracy.
tags:
  - AAAI 2026
  - AI Safety
  - Federated Recommendation
  - Parameter-Efficient Fine-Tuning
  - Embedding Compression
  - Communication Efficiency
  - Privacy Protection
date: 2026-05-08
content_hash: 8249d0e08d1255f6
---

# Plug-and-Play Parameter-Efficient Tuning of Embeddings for Federated Recommendation

**Conference**: AAAI 2026
**arXiv**: [2512.13734](https://arxiv.org/abs/2512.13734)
**Code**: [https://github.com/young1010/FedPEFT](https://github.com/young1010/FedPEFT)
**Area**: AI Safety
**Keywords**: Federated Recommendation, Parameter-Efficient Fine-Tuning, Embedding Compression, Communication Efficiency, Privacy Protection

## TL;DR

This paper proposes a plug-and-play federated recommendation framework that introduces PEFT (Parameter-Efficient Fine-Tuning) concepts into item embeddings. By freezing pre-trained full embeddings and transmitting only lightweight compressed embeddings (LoRA / Hash / RQ-VAE), the framework significantly reduces communication overhead while improving recommendation accuracy.

## Background & Motivation

Federated Recommendation (FR) is the mainstream framework for distributed recommendation training under user privacy constraints. The core idea is to keep user data on local clients and upload only model parameters to a central server for aggregation. However, **item embeddings** in recommendation models grow linearly with the number of items and typically account for the vast majority of model parameters. In large-scale item scenarios, transmitting full embeddings at each communication round becomes a severe bottleneck.

Existing solutions fall into two categories:

**Direct embedding compression** (low-rank decomposition, hashing, quantization, etc.): While reducing parameter count, these methods typically lead to a notable drop in recommendation accuracy.

**Introducing complex auxiliary models** (meta-learning, SENet, etc.): These partially compensate for accuracy loss but suffer from poor robustness, with unstable performance across different FR models and settings.

These limitations motivate the authors to ask: **Can the PEFT paradigm from NLP be adopted to combine full embeddings with compressed embeddings?** Specifically, the idea is to pre-train high-quality full embeddings on the server, freeze them, and then fine-tune and transmit only lightweight compressed embeddings during federated training. This approach retains the rich semantics of full embeddings while substantially reducing communication costs.

## Method

### Overall Architecture

The framework consists of three phases:

1. **Pre-training Phase**: The server pre-trains an autoencoder (AE) on item attributes to obtain high-quality full embeddings $E = \{e_i \in \mathbb{R}^k\}_{i=1}^n$.
2. **Warm-up Phase**: Full embeddings are distributed to clients for a small number of federated training rounds (< 20 rounds / 1000 rounds) to stabilize optimization.
3. **PEFT Training Phase**: Full embeddings are frozen; compressed embeddings are initialized and distributed, and only compressed embeddings are trained and transmitted thereafter. The final item embedding = frozen full embedding + trainable compressed embedding.

### Key Designs

#### 1. **Pre-trained Full Embeddings**

Item attributes are encoded into 768-dimensional input embeddings using Sentence-T5, then passed through an AE (encoder: [768, 512, 256, 128, 32]; decoder: symmetric) to learn 32-dimensional latent representations as full embeddings. The loss function is the reconstruction loss:

$$\mathcal{L}_{AE} = \|x - \hat{x}_{AE}\|^2$$

Pre-training is performed entirely on the server without involving user data, thus preserving privacy.

#### 2. **LoRA Strategy**

A low-dimensional embedding table $A = \{\mathbf{a}_i \in \mathbb{R}^{k_L}\}_{i=1}^n$ ($k_L \ll k$) and a projection matrix $B \in \mathbb{R}^{k \times k_L}$ are introduced. The compressed embedding is obtained via matrix multiplication:

$$\mathbf{e}_i = B(\mathbf{a}_i)$$

The final embedding is $\mathbf{E} = \{e_i + B(\mathbf{a}_i)\}_{i=1}^n$. $B$ is initialized to zero, ensuring that PEFT embeddings do not alter the output of full embeddings at the start of training. Communication cost is reduced from $O(k \cdot n)$ to $O(k_L \cdot (n + k))$.

#### 3. **Hash Strategy**

A family of universal hash functions $\mathcal{H}$ maps item IDs to vectors in a shared embedding table $H = \{v_i\}_{i=1}^{d_H}$ ($d_H \ll n$). Each item is represented by the combination of $h$ hash vectors. Two aggregation modes are provided:

- **Mean Pooling**: $\mathbf{e}_i = \frac{1}{h} \sum_{j=1}^h v_{\mathcal{H}_j(i)}$
- **SENet Attention Weighting**: A squeeze-excitation network dynamically computes the weight of each hash vector

Communication cost is only $O(d_H)$, independent of the number of items.

#### 4. **RQ-VAE Strategy (Novel Contribution)**

This work is the first to introduce Residual-Quantized Variational Autoencoders (RQ-VAE) into federated recommendation as a PEFT strategy. The core idea is:

- Maintain $l$ shared codebooks $(C_0, \ldots, C_{l-1})$, each of size $d_R$
- Each item is represented by a semantic code of length $l$: $\mathbf{c}_i = (c_0, \ldots, c_{l-1})$
- Quantized representation: $\hat{z} = \sum_{j=0}^{l-1} C_j(c_j)$

The **pre-training loss** consists of reconstruction loss and RQ-VAE loss:

$$\mathcal{L} = \|x - \hat{x}\|^2 + \sum_{j=0}^{l-1}\left(\|\text{sg}[r_j] - o_{j,c_j}\|^2 + \beta\|r_j - \text{sg}[o_{j,c_j}]\|^2\right)$$

During federated training, semantic codes are frozen on the client side, and only codebooks are optimized. Communication cost is $O(d_R \cdot l)$, with a representation space of $(d_R)^l$ that far exceeds the number of items.

### Loss & Training

- The recommendation task uses standard BPR loss or BCE loss (depending on the backbone model)
- Full embeddings participate in optimization during the warm-up phase (< 20 rounds) and are then frozen
- Each client performs 2 local epochs per round, with a 10% sampling rate over 1000 total rounds
- Differential privacy experiments use the Laplace mechanism, evaluated under both CDP and LDP settings

## Key Experimental Results

### Main Results

Comprehensive evaluation across 4 backbone models × 3 datasets (representative results shown):

| Model + Dataset | Method | N@10 | H@10 | vs. Full |
|---|---|---|---|---|
| FedMF-ML1M | Full | 33.98 | 58.44 | - |
| FedMF-ML1M | P-LoRA | **37.98** | **59.79** | +4.00/+1.35 |
| FedMF-ML1M | P-RQ-VAE | 33.59 | 58.96 | -0.39/+0.52 |
| FedNCF-ML1M | Full | 38.80 | 61.29 | - |
| FedNCF-ML1M | P-RQ-VAE | **39.75** | **60.91** | +0.95/-0.38 |
| PFedRec-ML1M | Full | 38.63 | 60.48 | - |
| PFedRec-ML1M | P-LoRA | **39.48** | **61.35** | +0.85/+0.87 |
| FedPerGNN-Industrial | P-RQ-VAE | **12.08** | **22.08** | +3.43/+7.27 |

**Key Finding**: PEFT embeddings outperform or match full embeddings in the vast majority of settings, while communication volume decreases by 50–90%.

### Ablation Study

| Configuration (PFedRec-ML1M) | N@10 | H@10 | Comm. (KB) | Notes |
|---|---|---|---|---|
| Full Embedding | 38.63 | 60.48 | 482.4 | Full baseline |
| P-LoRA ($k_L=2$) | 38.16 | 59.19 | 30.1 | Dimension too low |
| P-LoRA ($k_L=4$) | **39.48** | **61.35** | 60.3 | Optimal |
| P-LoRA ($k_L=6$) | 37.88 | 58.19 | 90.5 | Over-parameterized |

The optimal latent dimension for LoRA is $k_L = 4$, with communication volume at only 12.5% of full embeddings. For RQ-VAE, $d_R = 256, l = 4$ is the optimal configuration; excessively large codebooks introduce redundancy.

### Key Findings

1. **PEFT > Pure Compression**: Compressed embeddings alone (C-LoRA, C-Hash, etc.) exhibit poor robustness, but combining them with frozen full embeddings yields stable performance gains.
2. **Unique Advantage of RQ-VAE**: RQ-VAE performs best under LDP settings (performance even improves as noise increases), while LoRA is more robust under CDP.
3. **SENet + MLP Synergy**: SENet yields gains only for models that include MLPs (FedNCF, PFedRec), and actually degrades performance in pure embedding models.

## Highlights & Insights

- **Plug-and-Play Design**: The framework is decoupled from FR backbone models and can be seamlessly integrated into any embedding-based FR method.
- **First Application of RQ-VAE to FR**: The multi-level codebook quantization mechanism decouples embedding size from the number of items.
- **Comprehensive DP Analysis**: The framework's robustness is validated under both CDP and LDP privacy mechanisms.
- **Thorough Communication Analysis**: Three strategies are systematically compared across four dimensions: communication volume, storage, computation, and representational capacity.

## Limitations & Future Work

1. **No Universally Optimal Strategy**: The three compression strategies each have distinct trade-offs and no single strategy consistently dominates across all settings.
2. **Pre-training Depends on Item Attributes**: The approach requires server-side access to item attribute information (e.g., textual descriptions) and is not applicable when attributes are unavailable.
3. **Warm-up Phase Still Requires Full Embedding Transmission**: Although limited to fewer than 20 rounds, the initial distribution of full embeddings still incurs non-trivial overhead.
4. **Cold-Start Not Addressed**: How to efficiently update pre-trained embeddings when new items are added is not discussed.

## Related Work & Insights

- The evolution in federated recommendation from "full transmission" to "communication-efficient" approaches follows a clear trajectory.
- RQ-VAE originates from generative retrieval-based recommendation (TIGER, OneRec); its approach of encoding items as discrete semantic codes warrants further exploration.
- The transfer of NLP PEFT methods (LoRA, Adapter) to recommendation systems shows broad application prospects.

## Rating

- Novelty: ⭐⭐⭐⭐ (Applying RQ-VAE to FR is novel, though the overall framework draws on established PEFT ideas)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (4 models × 3 datasets × multiple strategies, including DP analysis)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure, comprehensive analysis)
- Value: ⭐⭐⭐⭐ (Addresses a practical FR communication bottleneck with strong applicability)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Fine-Grained DINO Tuning with Dual Supervision for Face Forgery Detection](fine-grained_dino_tuning_with_dual_supervision_for_face_forgery_detection.md)
- [\[AAAI 2026\] Reference Recommendation based Membership Inference Attack against Hybrid-based Recommender Systems](reference_recommendation_based_membership_inference_attack_against_hybrid-based_.md)
- [\[ICCV 2025\] LoRA-FAIR: Federated LoRA Fine-Tuning with Aggregation and Initialization Refinement](../../ICCV2025/ai_safety/lora-fair_federated_lora_fine-tuning_with_aggregation_and_initialization_refinem.md)
- [\[AAAI 2026\] Detect All-Type Deepfake Audio: Wavelet Prompt Tuning for Enhanced Auditory Perception](detect_all-type_deepfake_audio_wavelet_prompt_tuning_for_enhanced_auditory_perce.md)
- [\[CVPR 2026\] SubFLOT: Submodel Extraction for Efficient and Personalized Federated Learning via Optimal Transport](../../CVPR2026/ai_safety/subflot_submodel_extraction_for_efficient_and_personalized_federated_learning_vi.md)

</div>

<!-- RELATED:END -->
