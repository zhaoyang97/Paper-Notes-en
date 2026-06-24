---
title: >-
  [Paper Note] Plug-and-Play Parameter-Efficient Tuning of Embeddings for Federated Recommendation
description: >-
  [AAAI 2026][AI Safety][Federated Recommendation] This paper proposes a plug-and-play federated recommendation framework. By introducing the concept of PEFT (Parameter-Efficient Fine-Tuning) to item embeddings, it freezes the pre-trained full embeddings and transmits only the lightweight compressed embeddings (LoRA / Hash / RQ-VAE), substantially reducing communication overhead while improving recommendation accuracy.
tags:
  - "AAAI 2026"
  - "AI Safety"
  - "Federated Recommendation"
  - "Parameter-Efficient Fine-Tuning"
  - "Embedding Compression"
  - "Communication Efficiency"
  - "Privacy Protection"
date: 2026-05-08
content_hash: 8e8aa97b97e5cd24
---

# Plug-and-Play Parameter-Efficient Tuning of Embeddings for Federated Recommendation

**Conference**: AAAI 2026  
**arXiv**: [2512.13734](https://arxiv.org/abs/2512.13734)  
**Code**: [https://github.com/young1010/FedPEFT](https://github.com/young1010/FedPEFT)  
**Area**: AI Safety  
**Keywords**: Federated Recommendation, Parameter-Efficient Fine-Tuning, Embedding Compression, Communication Efficiency, Privacy Protection

## TL;DR

This paper proposes a plug-and-play federated recommendation framework. By introducing the concept of PEFT (Parameter-Efficient Fine-Tuning) to item embeddings, it freezes the pre-trained full embeddings and transmits only the lightweight compressed embeddings (LoRA / Hash / RQ-VAE), substantially reducing communication overhead while improving recommendation accuracy.

## Background & Motivation

Federated Recommendation (FR) is a mainstream framework for distributed recommendation training while addressing user privacy protection needs. Its core concept is that user data remains on local clients, and only model parameters are uploaded to a central server for aggregation. However, the scale of **item embeddings** in recommendation models grows linearly with the number of items, often accounting for the vast majority of model parameters. In large-scale item scenarios, transmitting full embeddings in each communication round becomes a severe bottleneck.

Existing solutions are mainly categorized into two types:

**Direct embedding compression** (low-rank factorization, hashing, quantization, etc.): Although this reduces the parameter size, it typically leads to a significant decrease in recommendation accuracy.

**Introducing complex auxiliary models** (meta-learning, SENet, etc.): While this can partially compensate for accuracy loss, the robustness is poor, showing unstable performance across different FR models and settings.

These limitations motivated the authors to consider: **Can the PEFT concept from the NLP field be borrowed to combine full embeddings with compressed embeddings?** Specifically, high-quality full embeddings are first pre-trained on the server side and then frozen, while only the lightweight compressed embeddings are fine-tuned and transmitted during federated training. This approach preserves the rich semantics of full embeddings while significantly reducing communication overhead.

## Method

### Overall Architecture

The framework consists of three stages:

1. **Pre-training Stage**: The server pre-trains item attributes using an autoencoder (AE) to obtain high-quality full embeddings $E = \{e_i \in \mathbb{R}^k\}_{i=1}^n$.
2. **Warm-up Stage**: The full embeddings are distributed to clients, followed by a few rounds (< 20 rounds out of 1000 rounds) of federated training to stabilize optimization.
3. **PEFT Training Stage**: The full embeddings are frozen, and compressed embeddings are initialized and distributed. Subsequently, only the compressed embeddings are trained and transmitted. The final item embedding is the sum of the frozen full embedding and the trainable compressed embedding.

### Key Designs

#### 1. **Pre-training Full Embeddings**

Sentence-T5 is used to encode item attributes into 768-dimensional input embeddings, which then pass through an AE (encoder [768, 512, 256, 128, 32], symmetrical decoder) to learn 32-dimensional latent representations as full embeddings. The loss function is the reconstruction loss:

$$\mathcal{L}_{AE} = \|x - \hat{x}_{AE}\|^2$$

Pre-training is completed on the server side without involving user data, thereby not compromising privacy.

#### 2. **LoRA Strategy**

A low-dimensional embedding table $A = \{\mathbf{a}_i \in \mathbb{R}^{k_L}\}_{i=1}^n$ ($k_L \ll k$) and a projection matrix $B \in \mathbb{R}^{k \times k_L}$ are introduced. The compressed embedding is obtained by matrix multiplication:

$$\mathbf{e}_i = B(\mathbf{a}_i)$$

The final embedding is represented as $\mathbf{E} = \{e_i + B(\mathbf{a}_i)\}_{i=1}^n$. $B$ is initialized as a zero matrix to ensure that the PEFT embedding does not alter the output of the full embedding at the beginning of training. The communication overhead is reduced from $O(k \cdot n)$ to $O(k_L \cdot (n + k))$.

#### 3. **Hash Strategy**

A family of universal hash functions $\mathcal{H}$ is used to map item IDs to vectors in a shared embedding table $H = \{v_i\}_{i=1}^{d_H}$ ($d_H \ll n$). Each item is composed of $h$ concatenated hash vectors. Two aggregation methods are provided:

- **Mean Pooling**: $\mathbf{e}_i = \frac{1}{h} \sum_{j=1}^h v_{\mathcal{H}_j(i)}$
- **SENet Attention Weighting**: Squeeze-excitation network is used to dynamically calculate the weights of each hash vector.

The communication overhead is only $O(d_H)$, independent of the total number of items.

#### 4. **RQ-VAE Strategy (Novelty)**

This work is the first to introduce Residual Quantized Variational Autoencoder (RQ-VAE) into federated recommendation as a PEFT strategy. The core ideas are:

- Maintain $l$ shared codebooks $(C_0, \ldots, C_{l-1})$, each of size $d_R$.
- Each item is represented by a semantic code $\mathbf{c}_i = (c_0, \ldots, c_{l-1})$ of length $l$.
- Quantized representation: $\hat{z} = \sum_{j=0}^{l-1} C_j(c_j)$.

The **pre-training loss** is the reconstruction loss + RQ-VAE loss:

$$\mathcal{L} = \|x - \hat{x}\|^2 + \sum_{j=0}^{l-1}\left(\|\text{sg}[r_j] - o_{j,c_j}\|^2 + \beta\|r_j - \text{sg}[o_{j,c_j}]\|^2\right)$$

During federated training, the semantic codes are frozen on the client side, and only the codebooks are optimized. The communication overhead is $O(d_R \cdot l)$, while the representation space size is $(d_R)^l$, significantly exceeding the number of items.

### Loss & Training

- Standard BPR loss or BCE loss (depending on the backbone model) is used for the recommendation task.
- During the warm-up stage, full embeddings participate in optimization (< 20 rounds) and are frozen afterwards.
- Clients perform 2 local epochs per round with a sampling rate of 10%, for a total of 1000 rounds.
- In differential privacy experiments, the Laplace mechanism is used to evaluate two settings: CDP (Central Differential Privacy) and LDP (Local Differential Privacy).

## Key Experimental Results

### Main Results

Comprehensive evaluation on 4 backbone models × 3 datasets (showing partial representative results):

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

**Key Findings**: PEFT embeddings outperform or match full embeddings in the vast majority of settings, while communication volume decreases by 50-90%.

### Ablation Study

| Configuration (PFedRec-ML1M) | N@10 | H@10 | Comm. (KB) | Description |
|---|---|---|---|---|
| Full Embedding | 38.63 | 60.48 | 482.4 | Full Baseline |
| P-LoRA ($k_L=2$) | 38.16 | 59.19 | 30.1 | Dimension too low |
| P-LoRA ($k_L=4$) | **39.48** | **61.35** | 60.3 | Optimal |
| P-LoRA ($k_L=6$) | 37.88 | 58.19 | 90.5 | Over-parameterized |

The optimal latent dimension for LoRA is $k_L = 4$, with a communication volume of only $12.5\%$ of the full embedding. For RQ-VAE, $d_R = 256, l = 4$ is the optimal configuration; an excessively large codebook introduces redundancy instead.

### Key Findings

1. **PEFT > Pure Compression**: Compressed embeddings (C-LoRA, C-Hash, etc.) exhibit poor robustness when used alone, but their performance consistently improves when combined with frozen full embeddings.
2. **Unique Advantage of RQ-VAE**: RQ-VAE performs best under the LDP setting (with performance even improving as noise increases), whereas LoRA is more robust under CDP.
3. **SENet + MLP Synergy**: SENet yields gains only on models containing MLPs (FedNCF, PFedRec), while deteriorating performance on pure embedding models.

## Highlights & Insights

- **Plug-and-Play Design**: The framework is decoupled from the FR backbone models and can be seamlessly integrated into any embedding-based FR method.
- **First to Introduce RQ-VAE to FR**: Utilizes a multi-level codebook quantization mechanism to decouple embedding size from the number of items.
- **Comprehensive DP Analysis**: Validates the robustness of the framework under both CDP and LDP privacy mechanisms.
- **Thorough Communication Analysis**: Systematically compares the three strategies across four dimensions: communication volume, storage, computation, and representation capacity.

## Limitations & Future Work

1. **No Single Optimal Strategy**: The three compression strategies have their own pros and cons, failing to achieve consistent optimal performance across all settings.
2. **Pre-training Depends on Item Attributes**: Requires the server-side to have access to item attributes (such as text descriptions), which is inapplicable in attribute-missing scenarios.
3. **Full Embeddings Still Require Transmission During Warm-up**: Though lasting for less than 20 rounds, the initial distribution of full embeddings still incurs some overhead.
4. **Cold Start Unconsidered**: How to efficiently update pre-trained embeddings when new items arrive is not discussed.

## Related Work & Insights

- The evolutionary path from "full transmission" to "efficient communication" in the field of federated recommendation is clear.
- RQ-VAE originates from generative retrieval recommendation (TIGER, OneRec); its approach of encoding items into discrete semantic codes warrants further exploration.
- The transfer application of PEFT methods from the NLP field (e.g., LoRA, Adapter) to recommendation systems holds promising prospects.

## Rating

- Novelty: ⭐⭐⭐⭐ (Using RQ-VAE for FR is novel, but the overall framework draws from PEFT concepts)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (4 models × 3 datasets × multiple strategies, including DP analysis)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure, comprehensive analysis)
- Value: ⭐⭐⭐⭐ (Addresses practical FR communication bottlenecks with strong practicality)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Gradient Inversion Attacks on Parameter-Efficient Fine-Tuning](../../CVPR2025/ai_safety/gradient_inversion_attacks_on_parameter-efficient_fine-tuning.md)
- [\[AAAI 2026\] Fine-Grained DINO Tuning with Dual Supervision for Face Forgery Detection](fine-grained_dino_tuning_with_dual_supervision_for_face_forgery_detection.md)
- [\[AAAI 2026\] Reference Recommendation based Membership Inference Attack against Hybrid-based Recommender Systems](reference_recommendation_based_membership_inference_attack_against_hybrid-based_.md)
- [\[ICML 2026\] From Parameter Dynamics to Risk Scoring: Quantifying Sample-Level Safety Degradation in LLM Fine-tuning](../../ICML2026/ai_safety/from_parameter_dynamics_to_risk_scoring_quantifying_sample-level_safety_degradat.md)
- [\[AAAI 2026\] Detect All-Type Deepfake Audio: Wavelet Prompt Tuning for Enhanced Auditory Perception](detect_all-type_deepfake_audio_wavelet_prompt_tuning_for_enhanced_auditory_perce.md)

</div>

<!-- RELATED:END -->
