---
title: >-
  [Paper Note] Federated Data-Efficient Instruction Tuning for Large Language Models
description: >-
  [ACL 2025 (Findings)][LLM Alignment][Federated Learning] Proposed FedHDS (Federated Hierarchical Data Selection) to eliminate intra- and inter-client data redundancy in federated learning via a two-level hierarchical data selection mechanism, combined with multi-layer Transformer feature fusion to improve coreset quality. Utilizing less than 1.5% of the data, it achieves an average Rouge-L improvement of 10.72% over the state-of-the-art full-data federated baseline…
tags:
  - "ACL 2025 (Findings)"
  - "LLM Alignment"
  - "Federated Learning"
  - "Data-Efficient"
  - "Instruction Tuning"
  - "Inter-layer Feature Fusion"
  - "Hierarchical Data Selection"
  - "Coreset"
  - "HDBSCAN"
date: 2026-05-08
content_hash: 3b5397e918c9f45e
---

# Federated Data-Efficient Instruction Tuning for Large Language Models

**Conference**: ACL 2025 (Findings)  
**arXiv**: [2410.10926](https://arxiv.org/abs/2410.10926)  
**Code**: [GitHub](https://github.com/zhenqincn/FedHDS)  
**Area**: LLM Alignment  
**Keywords**: Federated Learning, Data-Efficient, Instruction Tuning, Inter-layer Feature Fusion, Hierarchical Data Selection, Coreset, HDBSCAN

## TL;DR

Proposed FedHDS (Federated Hierarchical Data Selection) to eliminate intra- and inter-client data redundancy in federated learning via a two-level hierarchical data selection mechanism, combined with multi-layer Transformer feature fusion to improve coreset quality. Utilizing less than 1.5% of the data, it achieves an average Rouge-L improvement of 10.72% over the state-of-the-art full-data federated baseline, while accelerating training efficiency by up to 48.8 times.

## Background & Motivation

**Background**: Instruction tuning is a crucial step to align pre-trained LLMs with human instructions. Federated Learning (FL) has emerged as a popular paradigm for LLM fine-tuning by leveraging private instruction data scattered across edge devices to improve data diversity.

**Limitations of Prior Work**:
   - **(Efficiency Issue)** Existing federated fine-tuning methods (e.g., FedIT, FlexLoRA) use the full dataset on clients for local training, leading to massive computational overhead. Edge devices have limited GPU capacity and often require hybrid CPU+GPU computing, making the time cost of traversing the full dataset unaffordable.
   - **(Overfitting Issue)** In FL, each client usually covers only limited domains. Repeatedly training on redundant data leads to overfitting on local data, harming generalization performance on unseen tasks.
   - **(Incompatibility of Centralized Methods with FL)** Existing data-efficient methods (such as coreset selection) require concurrent access to all data. This directly violates the FL privacy principle of "data staying local" and cannot detect data redundancy across different clients.
   - **(Suboptimal Feature Representation)** Existing coreset methods rely solely on features from the last Transformer layer to distinguish data samples. However, experiments show that no single layer is optimal across all evaluation metrics.

**Key Challenge**: How to efficiently select representative data subsets while preserving privacy under the federated learning framework? How to eliminate both intra-client and inter-client data redundancy simultaneously?

**Key Insight**: A two-level hierarchical selection is proposed: first, HDBSCAN clustering is applied locally on clients to identify intra-client redundancy; then, approximate cluster centers (which do not correspond to real samples) are sent to the server for inter-client clustering to filter cross-client redundancy. Concurrently, features from all Transformer layers are fused to obtain better data representations.

**Core Idea**: Two-level hierarchical coreset selection in federated scenarios + full-layer feature fusion = exceeding full-data federated fine-tuning using <1.5% of the data.

## Method

### Overall Architecture

In each round of FL, FedHDS introduces an "I. Data Selection" module prior to regular local training. The workflow is as follows:

1. Clients download the latest global model $\mathbf{w}^r$
2. **Intra-client selection** (steps ①-④): Feature extraction $\rightarrow$ Fusion $\rightarrow$ Local clustering $\rightarrow$ Send cluster centers to the server
3. **Inter-client selection** (steps ⑤-⑦): Server clustering $\rightarrow$ Filtering $\rightarrow$ Notify clients which groups are selected
4. Clients perform local training solely on the coreset $\tilde{\mathcal{D}}_i$
5. Server performs FedAvg aggregation

### Key Designs

#### 1. Full-Layer Feature Fusion

For each data sample $\mathbf{x}$, the hidden states of the last token from all $l$ Transformer layers are extracted and concatenated as:

$$\mathbf{h}_j = [\mathbf{h}_j^{1,-1}, \mathbf{h}_j^{2,-1}, \ldots, \mathbf{h}_j^{l,-1}]$$

Then, dimension reduction to $k=2$ dimensions is performed via t-SNE (Barnes-Hut implementation):

$$\{\tilde{\mathbf{h}}_1, \tilde{\mathbf{h}}_2, \ldots\} = P(\{\mathbf{h}_1, \mathbf{h}_2, \ldots\})$$

**Design Motivation**: Experiments demonstrate that different layers provide features at various abstraction levels. Relying on the last layer alone is suboptimal across clustering quality metrics (CH Index, F1-Score, Silhouette Coefficient). t-SNE is more effective in non-linear spaces than PCA and can automatically suppress the influence of low-variance dimensions.

#### 2. Intra-Client Data Selection

HDBSCAN (density-based hierarchical clustering) is applied to the fused 2D feature space to partition the local data into groups $\{\mathcal{G}_1, \mathcal{G}_2, \ldots\}$. Each group corresponds to an approximate cluster center $\mathbf{c}_j$ (not corresponding to a real sample).

Reason for **choosing HDBSCAN over K-means**: HDBSCAN automatically determines the number of clusters without needing a preset $k$, and adaptively handles data with varying density distributions.

#### 3. Inter-Client Data Selection

The server collects cluster centers sent from all active clients and performs global clustering again using HDBSCAN:

$$\mathbb{G}^{II} = \text{HDBSCAN}(\{\mathbf{c}_1, \mathbf{c}_2, \ldots\})$$

Within each global group $\mathcal{G}_j^{II}$, the first-level group closest to its center is selected as the representative. The corresponding clients are then notified to select the real data samples closest to their group centers to be included in the coreset.

**Privacy Guarantee**: Only 2D cluster centers (a few dozen bytes) are transmitted, which do not correspond to any real samples. This can be further scaled via tanh + Gaussian noise to achieve $(\varepsilon, \delta)$-Differential Privacy.

#### 4. FedHDS-Turbo Acceleration Version

A lightweight proxy model (GPT-2, ~124M parameters) replaces the LLM for feature extraction, significantly reducing inference time with negligible loss of accuracy.

### Loss & Training

- PEFT: LoRA (rank=8, alpha=16, dropout=0.05)
- Optimizer: Adam, learning rate $3 \times 10^{-4}$ (NI) or $3 \times 10^{-5}$ (Dolly-15K)
- Cross-device FL settings: 5% client participation rate per round
- Communication overhead: In addition to LoRA parameters, only extra dozens of bytes for 2D centers and cluster indices are transmitted

## Key Experimental Results

### Main Results (Table 2: Rouge-L %)

| Method | NI (1.3B) | NI (3B) | Dolly (α=0.5, 1.3B) | Dolly (α=0.5, 3B) | Dolly (α=5.0, 1.3B) | Dolly (α=5.0, 3B) |
|------|-----------|---------|----------------------|--------------------|----------------------|--------------------|
| FedAvg (Full-parameter, Ref) | 22.08 | 24.40 | 30.18 | 31.90 | 30.18 | 31.90 |
| FedIT (LoRA full-data) | 21.29 | 28.00 | 30.55 | 32.49 | 30.57 | 32.02 |
| FlexLoRA | 21.67 | 27.35 | 30.31 | 32.89 | 31.60 | 33.24 |
| Random (Best ratio) | 24.86 | 28.80 | 31.41 | 33.44 | 31.09 | 32.86 |
| Coreset-Cent (Centralized) | 31.36 | 34.81 | 33.27 | 35.48 | 33.27 | 35.48 |
| **FedHDS** | **25.45** | **28.77** | **31.82** | **33.66** | **32.12** | **33.81** |
| **FedHDS-Turbo** | 24.77 | 28.42 | 31.47 | 33.80 | 31.67 | 33.57 |

- FedHDS-Turbo achieves an average Rouge-L improvement of **10.72%** compared to FedIT.
- FedHDS/FedHDS-Turbo utilizes **<1.5%** of the data.

### Efficiency Comparison (Table 3: Speedup)

| Method | NI (1.3B) Speedup | NI (3B) Speedup | Dolly (1.3B) Speedup | Dolly (3B) Speedup |
|------|----------------|--------------|-------------------|-----------------|
| FedIT (Full-data) | 1× | 1× | 1× | 1× |
| Random | 37.2× | 23.0× | 12.89× | 4.54× |
| **FedHDS** | 17.8× | 19.9× | 16.06× | 7.90× |
| **FedHDS-Turbo** | **48.8×** | **40.4×** | **18.86×** | **6.66×** |

### Communication and Memory Overhead (Table 5)

| Method | Model Comm. | Extra Comm. | GPU Memory |
|------|----------|----------|----------|
| FedIT | 12 MB | 0 | 10.56 GB |
| FedHDS | 12 MB | 44 Bytes | 9.40 GB |
| FedHDS-Turbo | 12 MB | 76 Bytes | 9.32 GB |

Extra communication overhead is almost negligible (only dozens of bytes for 2D cluster centers).

### Ablation Study

- **Two-Level Selection vs. Single-Level**: Removing inter-client selection (FedHDS‡) performs worse than full FedHDS in all scenarios, validating the necessity of hierarchical selection.
- **Global Selection vs. Hierarchical**: Direct transmission of all features to the server for global clustering (GlobalSelect) yields worse results, potentially due to suboptimal clustering performance on large-scale data.
- **Feature Fusion Methods**: t-SNE > PCA $\approx$ Kernel PCA, with the performance gap being more pronounced on the complex NI dataset.
- **Differential Privacy**: FedHDS still outperforms the Random baseline when the noise variance is $\le 0.1$.
- **Convergence and Overfitting**: The test loss of FedHDS continuously decreases, whereas the test loss of FedIT stops dropping early (while training loss continues to decrease), confirming that full-data training leads to overfitting.

## Highlights & Insights

- **First Work on Federated Data-Efficient Instruction Tuning**: Fills the gap in coreset selection within the FL paradigm. The qualitative comparison in Table 1 clearly demonstrates the differences from existing methods.
- **Exquisitely Designed Two-Level Hierarchical Selection**: Intra-client selection utilizes density clustering to automatically determine the number of groups, and inter-client selection transmits only 2D centers (a few dozen bytes), achieving both efficiency and privacy preservation.
- **Strong Empirical Support for Multi-Layer Feature Fusion**: Validates the hypothesis that "no single layer is a silver bullet" via multiple metrics such as CH Index, F1-Score, and Silhouette Coefficient, thereby justifying the necessity of feature fusion.
- **Counter-Intuitive Discovery of "Less is More"**: Selecting less data does not hurt but improves performance, as eliminating redundancy and overfitting leads to stronger generalization. This yields valuable insights for FL scenarios.
- **Proxy Model Concept in FedHDS-Turbo**: Employs GPT-2 (124M) instead of the LLM for feature extraction, achieving comparable accuracy with massive speedup, demonstrating strong practical engineering value.

## Limitations & Future Work

1. **Sole Focus on Data Representativeness, Ignoring Quality**: Low-quality samples might be preserved as independent domains; introducing quality-based filtering could yield further improvements.
2. **Experiments Limited to 1.3B / 3B Models**: The scalability and advantage have not been validated on larger-scale models (e.g., 7B, 13B).
3. **Feature Extraction Relies on a Full Forward Pass**: Even with FedHDS-Turbo, inference over the entire dataset is required in each round, which still incurs overhead when the dataset scales up.
4. **Potential Instability of HDBSCAN on High-Dimensional Sparse Data**: Although t-SNE reduction to 2D alleviates this, the non-deterministic nature of t-SNE may affect reproducibility.
5. **Evaluation Restricted to NLP Tasks**: Validation on other instruction tuning scenarios such as code generation and multi-modal tasks is yet to be explored.
6. **Centralized Coreset-Cent Still Outperforms FedHDS on Complex Datasets**: This indicates that privacy constraints inevitably cause performance degradation, suggesting room for improvement in federated coreset selection.

## Related Work & Insights

- **Relationship with FedIT (Zhang et al., 2024)**: FedIT serves as the federated instruction tuning baseline utilizing FedAvg + LoRA. FedHDS introduces a data selection module on top of it.
- **Connection to LIMA (Zhou et al., 2023)**: "Less is More for Alignment" demonstrates that a small set of high-quality data is sufficient for LLM alignment; FedHDS validates a similar conclusion in federated environments.
- **Comparison with Chen et al. (2023)**: The latter adopts K-means with final-layer features for centralized coreset selection, failing to handle cross-client redundancy in FL environments.
- **Insights**: (1) Data-quality scoring can be integrated with representativeness selection to achieve joint quality-representativeness optimization in FL; (2) The concept of using lightweight proxy models for feature extraction can be extended to other FL scenarios (e.g., Federated RAG).

## Rating

- Novelty: ⭐⭐⭐⭐ (First work on federated data-efficient instruction tuning, featuring hierarchical selection + full-layer fusion)
- Theoretical Depth: ⭐⭐⭐ (Provides convergence guarantees and DP proofs, though relatively straightforward)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (2 datasets $\times$ 2 models $\times$ various non-IID settings, thorough ablation studies, and comprehensive analysis of efficiency and privacy)
- Value: ⭐⭐⭐⭐ (Directly applicable to edge-side FL scenarios, with FedHDS-Turbo being highly engineering-friendly)
- Overall Recommendation: ⭐⭐⭐⭐ (Solid work in federated instruction tuning with comprehensive experiments)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Instruction Tuning of Large Language Models for Tabular Data Generation—in One Day](../../ICML2025/llm_alignment/instruction_tuning_of_large_language_models_for_tabular_data_generation-in_one_d.md)
- [\[ACL 2025\] Call for Rigor in Reporting Quality of Instruction Tuning Data](call_for_rigor_in_reporting_quality_of_instruction_tuning_data.md)
- [\[ACL 2025\] Measuring Data Diversity for Instruction Tuning: A Systematic Analysis and A Reliable Metric](measuring_data_diversity_for_instruction_tuning_a_systematic_analysis_and_a_reli.md)
- [\[AAAI 2026\] Importance-Aware Data Selection for Efficient LLM Instruction Tuning](../../AAAI2026/llm_alignment/importance-aware_data_selection_for_efficient_llm_instruction_tuning.md)
- [\[ACL 2025\] Rethinking Table Instruction Tuning](rethinking_table_instruction_tuning.md)

</div>

<!-- RELATED:END -->
