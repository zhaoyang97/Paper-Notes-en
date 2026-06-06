---
title: >-
  [Paper Note] Neural Estimation of Pairwise Mutual Information in Masked Discrete Sequence Models
description: >-
  [ICML2026][Computational Biology][Masked diffusion models] This paper starts from the hidden states of a pre-trained masked diffusion model (MDM) to train a lightweight "mutual information prediction head." This head out…
tags:
  - "ICML2026"
  - "Computational Biology"
  - "Masked diffusion models"
  - "Mutual information"
  - "Parallel sampling"
  - "ESM-C"
  - "Sudoku"
date: 2026-05-08
content_hash: eb44334962a4c088
---

# Neural Estimation of Pairwise Mutual Information in Masked Discrete Sequence Models

**Conference**: ICML2026  
**arXiv**: [2605.20187](https://arxiv.org/abs/2605.20187)  
**Code**: To be confirmed  
**Area**: Computational Biology  
**Keywords**: Masked diffusion models, Mutual information, Parallel sampling, ESM-C, Sudoku

## TL;DR
This paper starts from the hidden states of a pre-trained masked diffusion model (MDM) to train a lightweight "mutual information prediction head." This head outputs the full conditional mutual information matrix between all token pairs in a single forward pass. Based on this, it selects a "conditionally independent" subset of tokens for parallel decoding, reducing inference NFE by 3-5x on Sudoku and protein (ESM-C) tasks while maintaining or exceeding the quality of sequential decoding.

## Background & Motivation
**Background**: Masked diffusion models (MDM) are now considered powerful alternatives to autoregressive (AR) models for discrete sequence generation. They model the generation process as a reverse diffusion process that "gradually unmasks from a full mask." To avoid $L$ steps, the dominant acceleration strategy is Mask-Predict, which is based on "marginal confidence (entropy)"—unmasking the top-$k$ tokens with the highest confidence (lowest entropy) simultaneously at each step.

**Limitations of Prior Work**: Selecting parallel tokens based solely on marginal confidence fails on structured data. If two tokens that are both high-confidence but highly correlated are sampled simultaneously, violations of hard constraints occur (e.g., putting two "3"s in the same row in Sudoku) or global consistency is destroyed in protein generation. Works like EB-Sampler attempt to use the sum of marginal entropies to upper bound joint entropy, but this approach compresses the dependency structure into a single aggregated uncertainty value, failing to characterize conditional and asymmetric dependencies between tokens.

**Key Challenge**: During training, MDMs only explicitly learn the marginal $p(x^i \mid x_t)$ and do not directly provide the joint $p(x^i, x^j \mid x_t)$. However, the correctness of parallel decoding requires knowing which positions are conditionally independent—exactly the information that marginals cannot see.

**Goal**: (1) "Read out" the conditional mutual information $I(X_i; X_j \mid C)$ for any two positions $i, j$ from the hidden states of the MDM. (2) Use this MI matrix to guide parallel decoding, unmasking only tokens with low mutual information (conditionally independent) in the same step.

**Key Insight**: The hidden layers of an MDM themselves encode joint information—the head simply does not output it explicitly. Therefore, a lightweight predictor head can be trained by treating the MDM as a "feature extractor," with supervision signals coming from the "ground-truth MI" defined by the model itself (calculated via brute-force conditional probing).

**Core Idea**: Use a neural mutual information estimator to predict the full $N \times N$ MI matrix in one forward pass, then use a budgeted greedy algorithm based on entropy + $\lambda \cdot \text{MI}$ to select token subsets for parallel unmasking.

## Method

### Overall Architecture
The method consists of three components: (1) **Ground-truth MI Calculation**: Perform exhaustive conditional probing on the pre-trained MDM to obtain the $I(X_i; X_j \mid C)$ matrix that the model itself "believes" as the supervision signal; (2) **Neural MI Estimator**: Attach a prediction head $f_\phi$ on top of the frozen MDM. The input is the MDM's final hidden state $h \in \mathbb{R}^{N \times D}$, and the output is a symmetric matrix $\hat{I} \in \mathbb{R}^{N \times N}$, fitted to the ground truth using MSE; (3) **MI-Guided Parallel Sampling**: During inference, a single forward pass provides both marginal entropy $h_i$ and the MI matrix $\hat{I}$, followed by a budgeted greedy algorithm to select a token subset $U$ for parallel unmasking.

### Key Designs

1. **Ground-truth MI via Conditional Probing**:
    - **Function**: Back-calculate the joint distribution from an MDM that only outputs marginals to compute an accurate conditional MI matrix as the supervision target for the estimator.
    - **Mechanism**: Use the identity $I(X_i; X_j \mid C) = H(X_j \mid C) - H(X_j \mid X_i, C)$ to transform MI into "marginal entropy - conditional entropy." Specifically, perform $1 + N \cdot |V|$ forward passes: 1 base pass to get $P(X_i \mid C)$ and calculate $H(X_j \mid C)$; then, for each position $i$ and each vocabulary element $v \in V$, perform a pass where $X_i$ is temporarily fixed to $v$ to obtain $P(X_j \mid X_i = v, C)$. These are weighted by $P(X_i = v \mid C)$ and summed to get $H(X_j \mid X_i, C)$. Essentially, the MDM is treated as a probabilistic black box that can be "conditioned by punching holes."
    - **Design Motivation**: This supervision signal represents the "model's own belief" rather than the true data distribution. Thus, the estimator learns "how much dependency the model believes exists between $i$ and $j$." This matches both the estimator input (model hidden states) and the downstream use (guiding this specific model's parallel decoding). Although the $O(N \cdot |V|)$ inference cost is significant, it is only paid during the training phase.

2. **Lightweight MI Predictor Head on Hidden States**:
    - **Function**: Replace expensive conditional probing by directly outputting the entire $N \times N$ MI matrix in one forward pass.
    - **Mechanism**: Treat the frozen MDM as a backbone, take the final hidden state $h \in \mathbb{R}^{N \times D}$, and feed it to a small head $f_\phi$ to output a symmetric matrix $\hat{I} = f_\phi(\mathrm{MDM}(X_t))$. The training objective is the Frobenius MSE on masked positions: $\mathcal{L}_{MI} = \|M_{GT} - \hat{I}\|_F^2$. Training samples are generated by randomly sampling noise levels $t \sim \mathcal{U}[0,1]$ and masking the original sequence. Sudoku uses a head with ~100K parameters, while ESM-C (300M) uses a ~810K parameter head; both are extremely small.
    - **Design Motivation**: The MDM hidden layers inherently need to model joint relationships between tokens to predict masked positions, so "it knows but doesn't say"—one just needs a head to read out this internal belief. Converting MI estimation into a single forward pass makes it practical to use within the inference loop.

3. **Budgeted MI-Guided Parallel Sampling**:
    - **Function**: Select a set of "conditionally independent" tokens to unmask simultaneously in each denoising step, ensuring parallelization does not destroy global consistency.
    - **Mechanism**: Sort masked indices by marginal entropy $h_i$ in ascending order (highest confidence first). Initialize a budget $B = \gamma$. For each candidate $i$, calculate a dependency cost relative to the already selected set $U$: $d(i \mid U) = \sum_{j \in U} \hat{I}_{i,j}$. The total cost is $\text{cost} = h_i + \lambda \cdot d(i \mid U)$. If $\text{cost} \le B$, add $i$ to $U$ and deduct from the budget. Finally, sample all tokens in $U$ synchronously based on their marginals. $\gamma$ controls "how much to parallelize per step," and $\lambda$ is the dependency penalty coefficient.
    - **Design Motivation**: Selecting purely by entropy ($\lambda = 0$) degenerates into Mask-Predict, which might unmask highly correlated tokens simultaneously. By introducing $\lambda \cdot d(i \mid U)$, if $\hat{I}_{i,j}$ is large, the algorithm is forced to delay $j$ to the next step. The distribution of $j$ will then be updated because $i$ has been unmasked, thus preserving "chained conditional dependencies" while parallelizing only where tokens are truly independent.

### Loss & Training
The estimator only calculates Frobenius MSE on masked positions, while the original MDM remains frozen throughout. The Sudoku MDM has 4.16M parameters + 0.10M head, trained for 10 epochs on 100K random Sudokus. ESM-C uses the 300M parameter pre-trained ESM-Cambrian + 810K head, trained for 5 epochs on 10K proteins.

## Key Experimental Results

### Main Results: Sudoku Parallel Decoding

| Method | Avg. NFE | Solve Acc. |
|------|---------|-----------|
| Sequential | 53.9 | 61.6% |
| Naive ($k=4$) | 14.9 | 52.4% |
| Naive ($k=7$) | 9.0 | 36.8% |
| EB-Sampler ($\gamma=0.2$) | 15.3 | 61.0% |
| EB-Sampler ($\gamma=0.5$) | 9.9 | 51.2% |
| **MI-Guided ($\gamma=0.3$)** | 15.2 | **63.6%** |
| **MI-Guided ($\gamma=0.6$)** | 9.7 | 56.2% |

In low NFE regions, MI-Guided not only outperforms EB-Sampler but also achieves 2 percentage points higher accuracy than Sequential decoding using only 1/3 of the NFE at the $\gamma=0.3$ configuration.

### Main Results: ESM-C Protein Generation

| Method | Avg. NFE | JSD to Ref. ↓ |
|------|---------|---------------|
| Sequential | 74.8 | 0.093 |
| Naive ($k=4$) | 19.1 | 0.185 |
| Naive ($k=8$) | 9.8 | 0.196 |
| Naive ($k=12$) | 6.2 | 0.218 |
| **MI-guided ($\gamma=2, \lambda=1$)** | 15.3 | **0.136** |
| **MI-guided ($\gamma=4, \lambda=1$)** | 10.0 | 0.174 |

For 500 unconditional generations of length 50–100, Jensen-Shannon divergence (JSD) was calculated relative to the UniRef50 reference set. MI-guided achieved significantly lower JSD than Naive methods at similar NFE levels.

### Key Findings
- **Interpretability Byproduct**: On Sudoku, the MI matrix naturally recovers hard constraints for rows, columns, and 3×3 blocks—the model was never explicitly told the rules, but its internal belief already encoded these structures (Fig. 1).
- **Gap between MI and Entropy widens with structural complexity**: On strongly constrained tasks like Sudoku, EB-Sampler performs decently. However, in scenarios with obvious long-range dependencies like proteins, the quality of Naive/Entropy-based methods collapses, amplifying the advantage of MI-guided sampling.
- **$\lambda$ is the critical switch**: $\lambda = 0$ degenerates into Mask-Predict; if $\lambda$ is too large, it leads to excessive serialization. Experiments showed $\lambda = 1$ reaches the "sweet spot."

## Highlights & Insights
- **"Model's own MI" as a supervision signal**: Instead of using the data distribution to calculate ground-truth MI, the model's conditional probing results are used. This is clever as it eliminates the distribution gap between "data vs. model" and naturally fits the downstream goal of "guiding the decoding of this specific model." It is a sophisticated adaptation of the MINE series for generative models.
- **Treating acceleration as a "hidden state reading" problem**: The problem of "which tokens to select for parallel decoding," which usually relies on heuristics or additional sampling estimation, is transformed into "adding a prediction head on hidden states." This essentially acknowledges that MDMs are already computing joint info, just not exposing it. This perspective can be transferred to any marginal-only output model.
- **Intrinsic interpretability**: Visualizing the MI matrix reveals the "dependency structures understood by the model," such as row/col/box constraints in Sudoku. This approach can be used to probe what "structural knowledge" any MDM has acquired.

## Limitations & Future Work
- **High training overhead**: Calculating ground-truth MI requires $O(N \cdot |V|)$ forward passes, which is expensive for long sequences and large vocabularies (manageable for the 20-token protein alphabet, but could explode for ESM or text LLM vocabularies). The authors acknowledge this as a primary limitation and suggest curriculum learning or modified training strategies as future work.
- **Limited predictor precision**: MI estimation is essentially a regression task. The paper uses a head with ~100K–810K parameters and MSE; there is no systematic comparison with deeper architectures or contrastive learning objectives (like CLUB or INFOLOG).
- **Validation limited to Sudoku and ESM-C**: It has not been evaluated on large-scale text MDMs (e.g., SEDD, MD4) for language modeling parallel decoding. Generalization needs confirmation on more general language or molecular tasks.
- **Dependency on frozen MDM**: The head is added post-hoc. There is no exploration of whether "joint training of the MI head + MDM" could further improve the quality of MI representations.

## Related Work & Insights
- **vs Mask-Predict (Ghazvininejad et al., 2019)**: Mask-Predict is pure entropy-based parallelization ($\lambda = 0$). This work effectively adds an MI penalty term $\lambda \cdot d(i \mid U)$, theoretically correcting the error of unmasking highly correlated tokens simultaneously.
- **vs EB-Sampler (Ben-Hamu et al., 2025)**: EB-Sampler uses the sum of marginal entropies to upper bound joint entropy to delay related tokens, but it only reflects aggregate uncertainty. This paper directly predicts explicit pairwise MI, capturing conditional and asymmetric dependencies invisible to EB-Sampler, outperforming it in Sudoku experiments.
- **vs MINE / CLUB (Belghazi 2018, Cheng 2020)**: Classical neural MI estimation focuses on "maximizing/minimizing MI during representation learning." This paper is the first to embed a neural MI estimator into the inference loop of a generative model, with supervision coming from the model itself rather than data.
- **vs Sequential AR Decoding**: AR is forced into $O(L)$ steps. MI-guided parallelization achieves 56.2% at 9.7 NFE on Sudoku (vs. 61.6% at 53.9 NFE) and JSD 0.174 at 10 NFE on proteins (vs. JSD 0.093 at 74.8 NFE), compressing NFE to 1/5–1/7 while maintaining acceptable quality.

## Rating
- Novelty: ⭐⭐⭐⭐ The perspective of embedding neural MI estimation into the MDM inference loop is very novel, and using the model's own conditional probabilities as ground truth is a clever design.
- Experimental Thoroughness: ⭐⭐⭐ Comparisons were made in both Sudoku and ESM-C domains, but it lacked large-scale text MDMs and protein biological metrics beyond PCA/JSD were sparse.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clearly derived, algorithm descriptions are precise, and the Sudoku MI visualization in Fig. 1 is very convincing.
- Value: ⭐⭐⭐⭐ MI-guided parallel decoding is a general recipe with good reuse potential alongside the trend of MDM inference acceleration (e.g., SEDD).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Is Sequence Information All You Need for Bayesian Optimization of Antibodies?](../../NeurIPS2025/computational_biology/is_sequence_information_all_you_need_for_bayesian_optimization_of_antibodies.md)
- [\[CVPR 2026\] Cell-Type Prototype-Informed Neural Network for Gene Expression Estimation from Pathology Images](../../CVPR2026/computational_biology/cell-type_prototype-informed_neural_network_for_gene_expression_estimation_from_.md)
- [\[ICML 2026\] TD3B: Transition-Directed Discrete Diffusion for Allosteric Binder Generation](td3b_transition-directed_discrete_diffusion_for_allosteric_binder_generation.md)
- [\[ICML 2026\] CARD: Coarse-to-fine Autoregressive Modeling with Radix-based Decomposition for Transferable Free Energy Estimation](card_coarse-to-fine_autoregressive_modeling_with_radix-based_decomposition_for_t.md)
- [\[ICML 2026\] DNAChunker: Learnable Tokenization for DNA Language Models](dnachunker_learnable_tokenization_for_dna_language_models.md)

</div>

<!-- RELATED:END -->
