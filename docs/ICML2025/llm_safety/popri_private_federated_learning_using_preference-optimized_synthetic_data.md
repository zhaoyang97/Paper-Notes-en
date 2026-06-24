---
title: >-
  [Paper Note] POPri: Private Federated Learning using Preference-Optimized Synthetic Data
description: >-
  [ICML2025][LLM Safety][Differential Privacy] The differentially private federated learning synthetic data generation problem is reformulated as an LLM policy optimization (DPO) problem. By utilizing client DP feedback to construct preference pairs for fine-tuning the LLM, this approach achieves larger improvements than traditional Private Evolution—narrowing the privacy-performance gap by 58% under $\epsilon=1$.
tags:
  - "ICML2025"
  - "LLM Safety"
  - "Differential Privacy"
  - "Federated Learning"
  - "Synthetic Data"
  - "Preference Optimization"
  - "DPO"
  - "Private Evolution"
date: 2026-05-08
content_hash: ce6281be6e38f0c5
---

# POPri: Private Federated Learning using Preference-Optimized Synthetic Data

**Conference**: ICML2025  
**arXiv**: [2504.16438](https://arxiv.org/abs/2504.16438)  
**Code**: [meiyuw/POPri](https://github.com/meiyuw/POPri)  
**Area**: Federated Privacy / Differentially Private Synthetic Data  
**Keywords**: Differential Privacy, Federated Learning, Synthetic Data, Preference Optimization, DPO, Private Evolution

## TL;DR

The differentially private federated learning synthetic data generation problem is reformulated as an LLM policy optimization (DPO) problem. By utilizing client DP feedback to construct preference pairs for fine-tuning the LLM, this approach achieves larger improvements than traditional Private Evolution—narrowing the privacy-performance gap by 58% under $\epsilon=1$.

## Background & Motivation

Differentially Private Federated Learning (DP-FL) is the mainstream approach for training models from on-device private data, but it faces two major bottlenecks:

**Model Scale Limitations**: LLMs are too large to be stored and trained on client devices, making it difficult for standard DP-FL methods (e.g., DP-FedAvg, DP-FTRL) to directly leverage LLM capabilities.

**Limitations of Synthetic Data Methods**: Existing Private Evolution (PE)-based methods (such as PrE-Text) rely solely on prompting to guide LLMs in generating synthetic data; discarding low-scoring samples also wastes valuable information.

Key Insight: Client feedback collected by PE (similarity scores between synthetic samples and private data) can inherently be viewed as **RL reward signals**. Therefore, preference optimization (such as DPO) can replace simple prompt engineering to fine-tune the LLM.

## Method

### Overall Architecture

POPri is divided into four phases executed iteratively for $T$ rounds:

**Phase 1: Synthetic Sample Generation**
The server generates $K$ prompts. Each prompt is used by the LLM $\Psi$ to independently generate $J$ synthetic samples (yielding $K \times J$ samples in total). These samples are encoded using a sentence embedding model $\Gamma$ and then transmitted to the clients.

**Phase 2: DP Client Scoring**
Each client $i$ computes the average cosine similarity between each synthetic sample and its private samples to obtain a score vector. This vector is clipped to have an $L_2$ norm of 1 and then perturbed with Gaussian noise:

$$\text{Scores}_{i,t} + \mathcal{N}(0, \sigma^2 I / L)$$

The scores are aggregated across all clients via secure aggregation: $\text{Scores}_t = \frac{1}{L}\sum_{i \in \mathcal{S}^t} \text{Scores}_{i,t}$

**Phase 3: LLM Preference Optimization (Core Innovation)**
For each prompt $\eta_k$, its $J$ responses are sorted based on the aggregated scores:
- Highest-scoring sample $\rightarrow$ chosen ($y_\omega$)
- $\ell$-th highest-scoring sample $\rightarrow$ rejected ($y_r$)

After constructing the preference dataset, the LLM is fine-tuned using the DPO loss:

$$\min_\Psi \mathbb{E}_{x, y_\omega, y_r} \left[ -\log \sigma \left( \tau \log \frac{\Psi(y_\omega|x)}{\Psi(y_r|x)} - \tau \log \frac{\Psi_{\text{ref}}(y_\omega|x)}{\Psi_{\text{ref}}(y_r|x)} \right) \right]$$

where $\Psi_{\text{ref}}$ represents the frozen reference LLM weights, and $\tau$ controls the degree of deviation. LoRA (rank=4, $\alpha=8$) is utilized to reduce GPU memory footprint.

**Phase 4: Downstream Fine-Tuning**
The final-round LLM $\Psi_T$ generates large-scale synthetic data $S_{\text{syn},T+1}$, which is used to fine-tune the downstream model $\Phi$ that will be deployed to client devices.

### Privacy Guarantees

- Each client's score vector is clipped to an $L_2$ norm of 1 $\rightarrow$ Sensitivity is 1.
- Gaussian Mechanism: $\mathcal{N}(0, \sigma^2 I)$ noise is added.
- RDP Privacy Accountant (OPACUS) is used to track the $(\epsilon, \delta)$-DP budget.

### Key Differences from PE

| Dimension | Private Evolution | POPri |
|------|------------------|-------|
| Feedback Utilization | In-context learning (Prompting) | DPO fine-tuning of weights |
| Low-score Sample Handling | Discarded directly | Utilized as rejected samples |
| Similarity Metric | Nearest-neighbor histogram | Cosine similarity |

## Key Experimental Results

### Main Results: Next-Word Prediction Accuracy (%)

| Dataset | Method | $\epsilon=\infty$ | $\epsilon=7$ | $\epsilon=1$ | $\epsilon=0$ |
|--------|------|-----|-----|-----|-----|
| bioRxiv | DP-FedAvg | 41.5 | 29.0 | 28.3 | 27.9 |
| | PE | — | 31.0 | 31.1 | — |
| | PE+SFT | — | 28.6 | 28.6 | — |
| | **POPri** | — | **34.4** | **34.8** | — |
| Congress | DP-FedAvg | 35.7 | 29.1 | 29.0 | 26.9 |
| | PE | — | 27.3 | 27.0 | — |
| | **POPri** | — | **30.6** | **30.4** | — |

### Centralized DP Benchmarks

| Dataset | Method | $\epsilon=\infty$ | $\epsilon=1$ |
|--------|------|-----|-----|
| PubMed | PE (Llama-2-7b) | 47.6 | 27.5 |
| | **POPri** | — | **29.4** |
| OpenReview | PE (Llama-2-7b) | 50.8 | 37.0 |
| | **POPri** | — | **40.2** |

### Key Performance Indicators

- bioRxiv $\epsilon=1$: POPri narrows the privacy-performance gap by **58%**, compared to only 23% for PE and 3% for DP-FL.
- POPri achieves state-of-the-art downstream performance across all datasets and tasks.

### Additional Contribution: LargeFedBench

Introduces a new federated benchmark, LargeFedBench, comprising Congressional Records (134K clients) and bioRxiv abstracts (57K clients). It is updated periodically to prevent LLM training data contamination.

## Highlights & Insights

1. **Elegant Problem Formulation**: Reconceptualizing client score feedback from PE as RL rewards naturally leads to the DPO optimization paradigm, offering clear theoretical motivation.
2. **No Waste of Low-Scoring Samples**: Whereas PE discards low-scoring samples, POPri utilizes them as rejected pairs for DPO training, conveying rich information about "what is undesirable."
3. **Cosine Similarity Outperforms Histograms**: Ablation studies demonstrate that cosine similarity scoring is crucial for the policy optimization of POPri.
4. **Pareto Improvements in Privacy-Utility**: POPri significantly outperforms baselines even under strong privacy guarantees ($\epsilon=1$), while incurring lower communication and computation costs compared to standard DP-FL.
5. **Practical Value of LargeFedBench**: A large-scale, periodically updated, and contamination-free federated benchmark that fills a crucial gap in the community.

## Limitations & Future Work

1. **Evaluation Limited to Text Tasks**: The method relies on LLMs for synthetic data generation and has not yet been validated on other modalities such as images or tabular data.
2. **Dependency on LLMs**: Deploying a LLaMA-3-8B class model on the server side is required, demanding substantial computational resources.
3. **Score Quality Affected by Noise**: DP noise may lead to inaccurate preference pair construction, particularly under extremely low $\epsilon$ or with a small number of clients.
4. **Sensitivity to LoRA Hyperparameters**: Whether the fixed configuration of rank=4, $\alpha=8$ generalizes well across different tasks remains unexplored.
5. **Lack of Comparison with Newer Policy Optimization Methods**: Only DPO has been utilized, while alternatives like KTO and IPO remain unexplored.

## Related Work & Insights

- **Private Evolution Lineage**: PE $\rightarrow$ Aug-PE $\rightarrow$ PrE-Text represents the main trajectory of synthetic data methods. POPri builds on this by introducing weight fine-tuning.
- **DPO (Rafailov et al., 2023)**: Direct Preference Optimization serves as the core technical foundation.
- **DP-SGD / DP-FL**: Traditional privacy-preserving optimization methods serve as vital baselines.
- **Insights**: This framework can be generalized to other scenarios requiring private feedback paired with generative model optimization, such as privacy-preserving personalized recommendation.

## Rating

- Novelty: ⭐⭐⭐⭐ — The reformulation perspective from PE to DPO is highly novel, though the individual technical components (DPO + LoRA) are already established.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Evaluated across multiple datasets and settings with complete ablation studies, though it lacks cross-modal validation on non-text tasks.
- Writing Quality: ⭐⭐⭐⭐ — The motivation is clear, the algorithm description is detailed, and the diagrams are intuitive.
- Value: ⭐⭐⭐⭐ — It provides tangible advancements to the field of private federated learning, and LargeFedBench holds long-term value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] FedSVD: Adaptive Orthogonalization for Private Federated Learning with LoRA](../../NeurIPS2025/llm_safety/fedsvd_adaptive_orthogonalization_for_private_federated_learning_with_lora.md)
- [\[ICML 2025\] Reward-Augmented Data Enhances Direct Preference Alignment of LLMs](reward-augmented_data_enhances_direct_preference_alignment_of_llms.md)
- [\[ICLR 2026\] Secret-Protected Evolution for Differentially Private Synthetic Text Generation](../../ICLR2026/llm_safety/secret-protected_evolution_for_differentially_private_synthetic_text_generation.md)
- [\[NeurIPS 2025\] Differentially Private Federated Low Rank Adaptation Beyond Fixed-Matrix](../../NeurIPS2025/llm_safety/differentially_private_federated_low_rank_adaptation_beyond_fixed-matrix.md)
- [\[ACL 2026\] Differentially Private Synthetic Text Generation for Retrieval-Augmented Generation (RAG)](../../ACL2026/llm_safety/differentially_private_synthetic_text_generation_for_retrieval-augmented_generat.md)

</div>

<!-- RELATED:END -->
