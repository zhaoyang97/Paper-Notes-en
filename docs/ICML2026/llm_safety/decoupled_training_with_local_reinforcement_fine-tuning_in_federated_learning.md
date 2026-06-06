---
title: >-
  [Paper Note] Decoupled Training with Local Reinforcement Fine-Tuning in Federated Learning
description: >-
  [ICML 2026][LLM Safety][Federated Learning] FedDTL maintains CLIP's image encoder on the client while moving the text encoder to the server as a "global semantic anchor." It employs a two-stage local fine-tuning process…
tags:
  - "ICML 2026"
  - "LLM Safety"
  - "Federated Learning"
  - "CLIP"
  - "LoRA"
  - "GRPO"
  - "Decoupled Training"
date: 2026-05-08
content_hash: 4ce79dc87c6102b8
---

# Decoupled Training with Local Reinforcement Fine-Tuning in Federated Learning

**Conference**: ICML 2026  
**arXiv**: [2605.27900](https://arxiv.org/abs/2605.27900)  
**Code**: To be confirmed  
**Area**: Federated Learning / Vision-Language Models / Reinforcement Learning Fine-Tuning  
**Keywords**: Federated Learning, CLIP, LoRA, GRPO, Decoupled Training

## TL;DR
FedDTL maintains CLIP's image encoder on the client while moving the text encoder to the server as a "global semantic anchor." It employs a two-stage local fine-tuning process (SFT warm-up followed by GRPO-style RL) to simultaneously mitigate inter-client optimization inconsistency and intra-client overfitting in heterogeneous and full-data federated scenarios.

## Background & Motivation

**Background**: In Federated Learning (FL), incorporating pre-trained VLMs like CLIP into downstream tasks has become mainstream. Typically, the backbone is frozen, and Parameter-Efficient Fine-Tuning (PEFT) such as prompts, adapters, or LoRA is performed locally on clients, followed by parameter averaging aggregation on the server.

**Limitations of Prior Work**: Under Non-IID and full-data conditions, pure local optimization over long trajectories leads to two concurrent issues: (i) **Inter-client optimization inconsistency**—local objectives are misaligned and gradient directions differ, preventing the formation of coherent global semantics after parameter averaging; (ii) **Intra-client over-specialization**—local PEFT parameters become biased by skewed label frequencies and feature statistics, degrading generalization to unseen classes/domains.

**Key Challenge**: Existing methods mostly rely on "pure local optimization + server averaging" with additional regularization or alignment losses. These still depend on parameter averaging for cross-client knowledge transfer and **cannot systematically address representation-level client drift**. Furthermore, most evaluations are conducted only in few-shot settings, which masks the amplified severity of these issues in full-data scenarios.

**Goal**: To **simultaneously** improve global task adaptation (base classes) and generalization (novel classes) under diverse federated data distributions, including few-shot, full-data, label skew, and feature shift.

**Key Insight**: The authors observe that CLIP's "modal decoupling + alignment" is structurally isomorphic to the "server-client broadcast" in FL—images must be processed locally for privacy, while the text encoder, which only processes class names, is naturally suited for placement on the server. Additionally, insights from Chu et al. (2025) suggesting "SFT tends toward memorization while RL favors generalization" imply that RL can substitute for auxiliary regularization.

**Core Idea**: **Encoder decoupling across devices + SFT → RL two-stage local fine-tuning**. The server trains the text encoder to provide a unified semantic "anchor," while clients perform LoRA fine-tuning only on the visual side. Local training starts with an SFT warm-up before switching to GRPO-style RL to suppress over-specialization.

## Method

### Overall Architecture
With $K$ clients and a central server, each client $k$ holds private data $\mathcal{D}_k=\{(x_i,y_i)\}_{i=1}^{N_k}$. The pipeline for each global round $t$ proceeds as follows:

1.  **Downlink Broadcast**: The server broadcasts the global visual LoRA parameters $\Delta\mathbf{W}_g^{t-1}$ and global text embeddings for all classes $\{\bar z_{\text{text}}^{c,t-1}\}_{c=1}^{C}$ to each client.
2.  **Local Fine-tuning**: Clients encode local images into $\bar z_v$ using the LoRA-tuned image encoder $\mathcal{V}_k$, and calculate classification via cosine similarity with the received global text embeddings. The first $M$ rounds comprise the SFT stage (cross-entropy warm-up), followed by the RL stage.
3.  **Uplink Communication**: Clients upload only the local visual LoRA parameters $\Delta\mathbf{W}_k$ and normalized image class embeddings $\bar z_{v,k}$ (class tokens only; patch tokens are excluded, and subsets may be sampled in full-data scenarios).
4.  **Server Aggregation + Global Training**: The server performs sample-weighted averaging of visual LoRA parameters. Simultaneously, it uses the uploaded image embeddings as supervision to train the global text encoder $\mathcal{T}_g$ (also via LoRA) on the server side, completing a global round.

This structure relies on "client visual decoupling + server text unification" combined with "two-stage local fine-tuning" to suppress inter-client inconsistency and intra-client overfitting without extra regularization terms.

### Key Designs

1.  **Decoupled Encoder Training**:
    *   **Function**: Splits CLIP’s dual-tower structure based on data access—the image encoder remains on the client to protect privacy, while the text encoder moves to the server for global semantic alignment.
    *   **Mechanism**: Clients use LoRA to fine-tune the last $L-l$ layers of the image encoder ($W=W_0+BA$, $r\ll d$), aligning with global text embeddings using $p(\hat y=c|x)=\frac{\exp(\text{sim}(\bar z_v,\bar z_{\text{text}}^c)/\tau)}{\sum_j\exp(\text{sim}(\bar z_v,\bar z_{\text{text}}^j)/\tau)}$. The server, upon receiving class-token embeddings, trains the text encoder LoRA $\Delta\mathbf{W}_{\text{text}}$ using cross-entropy, mapping "a photo of a [classname]" to a unified text embedding space aligned with the global visual space.
    *   **Design Motivation**: Compared to methods where clients train fully locally, the global text encoder acts as a semantic "anchor" independent of specific client distributions. This forces visual representations across all clients to converge toward a unified coordinate system, fundamentally suppressing representation-level client drift. Uploading highly compressed class-token embeddings also minimizes privacy exposure and communication bandwidth.

2.  **SFT-Warmed Task Adaptation Stage**:
    *   **Function**: Uses supervised learning to rapidly pull the image encoder to a stable task-related initialization before RL, preventing policy sampling from an underfitted state.
    *   **Mechanism**: Clients perform standard cross-entropy $\mathcal{L}_{ce}=-\frac{1}{N_k}\sum_{(x_i,y_i)}\sum_c y_i\log p(\hat y=c|x_i)$ for the first $M$ global rounds, aiming to $\min_{\Delta\mathbf{W}_k}\mathcal{L}_{ce}([\mathbf{W}_0,\Delta\mathbf{W}_k];\{\bar z_{\text{text}}^c\},\mathcal{D}_k)$, with $T_e=2$ local epochs per round.
    *   **Design Motivation**: Pure RL has low sample efficiency in classification fine-tuning, while pure SFT on long trajectories becomes biased by local distributions. Sequencing SFT then RL assigns "fast adaptation" and "anti-overfitting" to the optimization paradigms best suited for each.

3.  **GRPO-Inspired RL Generalization Enhancement Stage**:
    *   **Function**: Uses reinforcement learning instead of additional regularization to actively suppress intra-client over-specialization during long-trajectory local training.
    *   **Mechanism**: The SFT-converged image encoder is treated as policy $\pi_{\theta_k}$. To address the deterministic output of CLIP-style encoders, small Gaussian noise $\varepsilon\sim\mathcal{N}(0,\sigma^2 I)$ is injected into latent embeddings to create controllable randomness, sampling $G=3$ actions per image. Classification correctness serves as a 0/1 reward, and relative advantage $A_{i,j}$ is calculated via intra-group normalization. The GRPO $\epsilon$-clip policy gradient $\mathcal{L}_p=\min[\rho_{i,j}A_{i,j},\text{clip}(\rho_{i,j},1-\epsilon,1+\epsilon)A_{i,j}]$ is applied, with an unbiased KL estimate $\mathbb{D}_{\text{KL}}$ against a hybrid reference model (weighted 0.5 SFT model and 0.5 latest global policy). The final goal is $\mathcal{L}_{rl}=-\frac{1}{G}\sum_j\frac{1}{bs}\sum_i(\mathcal{L}_p-\beta\mathbb{D}_{\text{KL}})$ with $\beta=0.5$.
    *   **Design Motivation**: Unlike direct GRPO application, noise injection during sampling (while keeping the model deterministic during updates) preserves intra-group optimization while maintaining stability. The hybrid reference model provides a task-aware anchor for KL, preventing the policy from collapsing or drifting excessively compared to a single reference model.

### Loss & Training
Key hyperparameters: ViT-B/16 backbone, LoRA rank $r=4$ inserted from layer $l=10$; Adam optimizer, $\eta=1e-3$, batch size 64; $T=20$ global rounds, each with $T_e=2$ (SFT) / $3$ (RL) local epochs, $K=5$ clients. For RL: $\sigma=0.1$, $G=3$, $\epsilon=0.2$, $\beta=0.5$. Only class-token embeddings are uploaded.

## Key Experimental Results

### Main Results
Average accuracy across 9 label skew benchmarks (CIFAR10/100, EuroSAT, TinyImageNet, OxfordPet, Flower102, Caltech101/256, Food101), focusing on Base (task adaptation) and Novel (generalization) classes:

| Setup | Method | Base | Novel |
| :--- | :--- | :--- | :--- |
| Few-shot Non-IID | FedMaPLe | 83.63 | 77.56 |
| Few-shot Non-IID | **Ours** | **89.58** | **83.01** |
| Few-shot Dir(0.1) | FedMaPLe | 84.05 | 77.69 |
| Few-shot Dir(0.1) | **Ours** | **90.95** | **82.64** |
| Full-data Non-IID | FedMaPLe | 80.56 | 69.41 |
| Full-data Non-IID | **Ours** | **91.64** | **77.72** |
| Full-data Dir(0.1) | FedMaPLe | 89.27 | 70.10 |
| Full-data Dir(0.1) | **Ours** | **92.40** | **76.59** |

For Feature shift (DomainNet, Full-one / Full-Dir(0.1)): Ours achieved 93.38 / 93.47, compared to FedMaPLe's 91.94 / 90.51.

### Ablation Study
Means across 7 datasets, showing Base / Novel / harmonic mean (HM):

| Configuration | Few_Non-IID Base / Novel / HM | Full_Non-IID Base / Novel / HM |
| :--- | :--- | :--- |
| FedLoRA (Baseline) | 78.32 / 78.86 / 78.56 | 58.11 / 70.51 / 63.12 |
| + Decoupled Encoder Training | 86.42 / 79.52 / 82.60 | 86.68 / 73.57 / 79.20 |
| + Two-stage Local FT | 79.46 / 83.84 / 81.47 | 47.91 / 76.43 / 57.86 |
| **FedDTL (Ours)** | **90.06 / 83.58 / 86.51** | **90.58 / 80.62 / ≈85** |

### Key Findings
*   Adding decoupled encoder training alone increases Base accuracy in Full_Non-IID from 58 to 87 (+28), indicating that inter-client inconsistency is primarily addressed by this module; however, Novel gains remain limited without RL.
*   Two-stage fine-tuning alone actually crashes Base accuracy to 47.91 in Full_Non-IID, showing that pure RL is unstable under heterogeneity without a global semantic anchor. The two components must work in tandem.
*   While multiple baselines experience significant Novel accuracy drops when moving from few-shot to full-data (e.g., pFedMMA drops from 74.91 to 65.56 under Dir(0.1)), Ours remains stable across all settings, demonstrating successful suppression of long-trajectory overfitting.

## Highlights & Insights
*   **Analogy between CLIP modal decoupling and FL broadcasting**: This serves as the conceptual "hook"—the requirement for images to remain local (privacy) and text to remain global (class names) aligns naturally with FL physical constraints, creating a "global semantic anchor" without new complex components.
*   **Using RL as an alternative to regularization in FL**: This is not a direct port of GRPO. The authors solve the engineering problem of zero-advantage in deterministic encoders via "noise injection during sampling + deterministic model for updates + hybrid reference model," providing a robust GRPO variant applicable to any vision RL scenario using classification heads as policies.
*   **Emphasis on full-data evaluation**: Most federated VLM works report only few-shot results. Full-data environments truly expose the composite effects of inter-client inconsistency and intra-client over-specialization. Reporting both sets of results highlights the vulnerability of existing baselines.

## Limitations & Future Work
*   Communication costs scale linearly with the number of classes $C$ (broadcasting $C$ global text embeddings per round). While uplink communication can be subsampled, no compression scheme is proposed for the downlink broadcast.
*   Experiments are based exclusively on CLIP ViT-B/16. It remains unverified if decoupled training can suppress drift in larger backbones (ViT-L, Eva-CLIP). The LoRA rank $r=4$ and insertion layer $l=10$ are fixed and tightly coupled to the backbone.
*   The RL stage requires $G$ times more forward passes ($G=3$), placing significantly higher computational pressure on clients than pure SFT, which may not be feasible for compute-constrained edge FL.
*   Privacy analysis is qualitative (focusing on class-token embeddings). Formal Differential Privacy (DP) guarantees or empirical tests against embedding inversion attacks are missing.

## Related Work & Insights
*   **vs FedMaPLe / PromptFL**: These local-optimization+averaging methods rely on complex prompt tuning for knowledge transfer. Ours shifts to server-side text encoder training, fundamentally changing the "averaging-as-knowledge-transfer" paradigm.
*   **vs FedPGP / pFedMMA**: These depend on additional alignment/regularization losses within SFT. Ours replaces regularization with an RL stage, avoiding the "Frankenstein" feel of multiple loss terms.
*   **vs FedPPO / AFedPG**: While these also use RL in FL, they focus on system heterogeneity (stragglers, asynchronous policies) and optimize FL scheduling. Ours applies RL directly to model parameters to solve statistical heterogeneity and generalization.

## Rating
*   **Novelty**: ⭐⭐⭐⭐ The structural alignment of CLIP decoupling with FL broadcasting is elegant, and the noise-injected GRPO adaptation for deterministic encoders is a practical new solution.
*   **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers 9 label-skew and 2 feature-shift datasets across 5 distributions and both few-shot/full-data settings. Ablations accurately attribute gains to the core modules.
*   **Writing Quality**: ⭐⭐⭐⭐ Effectively explains the composite motivation of inter-client and intra-client issues. Formulas are well-structured, though RL implementation details may be dense for readers unfamiliar with GRPO.
*   **Value**: ⭐⭐⭐⭐ Addresses the genuine pain point of stability in Federated VLM under full-data heterogeneity. The decoupling + two-stage fine-tuning combination can be widely adopted by future FL+RL research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] FedALT: Federated Fine-Tuning through Adaptive Local Training with Rest-of-World LoRA](../../AAAI2026/llm_safety/fedalt_federated_fine-tuning_through_adaptive_local_training_with_rest-of-world_.md)
- [\[ICML 2026\] FedTreeLoRA: Reconciling Statistical and Functional Heterogeneity in Federated LoRA Fine-Tuning](fedtreelora_reconciling_statistical_and_functional_heterogeneity_in_federated_lo.md)
- [\[ICML 2026\] Towards Fine-Grained Robustness: Attention-Guided Test-Time Prompt Tuning for Vision-Language Models](towards_fine-grained_robustness_attention-guided_test-time_prompt_tuning_for_vis.md)
- [\[NeurIPS 2025\] Adaptive LoRA Experts Allocation and Selection for Federated Fine-Tuning](../../NeurIPS2025/llm_safety/adaptive_lora_experts_allocation_and_selection_for_federated_fine-tuning.md)
- [\[ICLR 2026\] Heterogeneous Federated Fine-Tuning with Parallel One-Rank Adaptation](../../ICLR2026/llm_safety/heterogeneous_federated_fine-tuning_with_parallel_one-rank_adaptation.md)

</div>

<!-- RELATED:END -->
