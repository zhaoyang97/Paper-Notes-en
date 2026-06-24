---
title: >-
  [Paper Note] DavIR: Data Selection via Implicit Reward for Large Language Models
description: >-
  [ACL 2025][LLM Pretraining][Data Selection] The DavIR data selection method is proposed, which effectively eliminates the sequence-length dependency in the RHO objective through **reference model loss normalization** (rather than token-count normalization) of the loss difference between the base and reference models. This allows a model trained on only **6%** of the Alpaca dataset (3K/52K) to outperform one trained on the full dataset, while extending the normalization concep…
tags:
  - "ACL 2025"
  - "LLM Pretraining"
  - "Data Selection"
  - "Coreset Selection"
  - "Implicit Reward"
  - "DPO"
  - "Instruction Tuning"
  - "Length Bias"
  - "Learnability"
date: 2026-05-08
content_hash: 12df16eec1ddfc67
---

# DavIR: Data Selection via Implicit Reward for Large Language Models

**Conference**: ACL 2025  
**arXiv**: [2310.13008](https://arxiv.org/abs/2310.13008)  
**Code**: Not publicly available  
**Authors**: Haotian Zhou, Tingkai Liu, Qianli Ma, Yufeng Zhang, Jianbo Yuan, Pengfei Liu, Yang You, Hongxia Yang  
**Institutions**: ByteDance, Cold Spring Harbor Laboratory, Shanghai Jiao Tong University, National University of Singapore  
**Area**: LLM Pre-training  
**Keywords**: Data Selection, Coreset Selection, Implicit Reward, DPO, Instruction Tuning, Length Bias, Learnability

## TL;DR

The DavIR data selection method is proposed, which effectively eliminates the sequence-length dependency in the RHO objective through **reference model loss normalization** (rather than token-count normalization) of the loss difference between the base and reference models. This allows a model trained on only **6%** of the Alpaca dataset (3K/52K) to outperform one trained on the full dataset, while extending the normalization concept to DPO to yield DavIR-DPO, improving Zephyr's alignment performance on AlpacaEval by 8%.

## Background & Motivation

**Importance of SFT Data Selection**: According to the "Superficial Alignment Hypothesis," a small amount of selected high-quality data is sufficient to guide a pre-trained LLM to exhibit instruction-following capabilities (e.g., Zhou et al. 2023 LIMA using only 1K samples).

**Limitations of Prior Work**:
   - **Data-centric methods**: AlpaGasus uses ChatGPT scoring for filtering (quality-oriented), and LIMA relies on manual annotation (diversity-oriented), but both **neglect the capabilities of the base model itself**.
   - **Dependence on external teacher models**: ChatGPT annotations introduce safety and cost concerns.
   - **Model-centric but flawed**: RHO (Reducible Holdout Loss) is theoretically sound, but exhibits a high correlation with sequence length when directly applied to language modeling.

**Key Findings**: In language modeling, the Spearman correlation between token-level entropy/loss and sequence length is as high as **-0.97** (Albert on Alpaca). The RHO objective inherits this correlation, causing data selection to degenerate into an **approximate sorting by length**.

**Key Designs**: A subtle but critical change in normalization—using the reference model loss as the denominator instead of the token count—drastically reduces length dependency.

## Method

### Core Formula: From RHO-LM to DavIR

**RHO-LM (Baseline)**: Generalizing Reducible Holdout Loss to causal language modeling:

$$S_{\text{RHO-LM}}(x,y) = \mathcal{L}_{\text{base}}(y|x) - \mathcal{L}_{\text{ref}}(y|x)$$

Where $\pi_{\text{base}}$ is the pre-trained base model, and $\pi_{\text{ref}}$ is the reference model fine-tuned on the full dataset.

**Issue**: $S_{\text{RHO-LM}}$ is highly correlated with sequence length (Pearson correlation of 0.64-0.83, see Table 2). This occurs because in autoregressive language modeling, longer sequences provide more contextual constraints that restrict subsequent token distributions, resulting in systematically lower average losses for longer sequences.

**DavIR (Ours)**:

$$S_{\text{DavIR}}(x_i, y_i) = \frac{\mathcal{L}_{\text{base}}(x_i, y_i) - \mathcal{L}_{\text{ref}}(x_i, y_i)}{\mathcal{L}_{\text{base}}(x_i, y_i)}$$

- Key Difference: The denominator uses **the base model's own loss** instead of the token count.
- Mathematically equivalent to: $1 - \mathcal{L}_{\text{ref}} / \mathcal{L}_{\text{base}}$
- Intuition: Measures the "proportion learned" by the model—denominator normalization eliminates differences in absolute loss magnitudes across data of different lengths.
- Using either the base or reference model loss as the denominator does not affect the ranking (simple proof provided in Appendix C).

### Relationship with Implicit Reward

The implicit reward function of DPO: $r(x,y) = \beta \log \frac{\pi(y|x)}{\pi_{\text{base}}(y|x)} = \beta \cdot [\mathcal{L}_{\text{base}} - \mathcal{L}]$

- The scoring function of RHO-LM is exactly a constant multiple of the DPO implicit reward.
- DavIR can be viewed as a **normalized implicit reward**, selecting data with the "highest relative learning potential reward".

### DavIR Algorithmic Process

1. Fine-tune the base model on the full dataset $D_{\text{full}}$ to obtain $\pi_{\text{ref}}$.
2. For each $(x_i, y_i) \in D_{\text{full}}$, compute $\mathcal{L}_{\text{base}}$ and $\mathcal{L}_{\text{ref}}$.
3. Compute $S_{\text{DavIR}}$ and rank them.
4. Select the top-k highest-scoring data to form the training set.
5. Re-fine-tune $\pi_{\text{base}}$ on the top-k data.

### DavIR-DPO Extension

Extending the normalization concept to the DPO training objective:

$$\mathcal{L}_{\text{DavIR-DPO}} = -\mathbb{E}\left[\log \sigma\left(\beta \frac{\log \pi_\theta(y_w|x) / \pi_{\text{ref}}(y_w|x)}{|\log \pi_{\text{ref}}(y_w|x)|} - \beta \frac{\log \pi_\theta(y_l|x) / \pi_{\text{ref}}(y_l|x)}{|\log \pi_{\text{ref}}(y_l|x)|}\right)\right]$$

- Normalize winning and losing responses separately using their respective reference model losses.
- Goal: Reduce the dependency of the DPO objective on variations in output/response length.

## Key Experimental Results

### Length Dependency Analysis

| Dataset | Model | RHO-LM Spearman | DavIR Spearman |
|--------|------|-----------------|----------------|
| Alpaca | gemma-2b | 0.75 | **0.30** |
| Alpaca | gemma-2-2b | 0.83 | **0.47** |
| GSM8K | gemma-2b | 0.58 | **0.06** |
| LIMA | gemma-2b | 0.20 | **0.02** |

- DavIR significantly reduces correlation with length (dropping from 0.58 to 0.06 in the best case).

### SFT Data Selection: 16x Compression Rate

Results on the Alpaca dataset using LLaMA-7B/13B (Figure 1):
- **3K/52K = only 6%** of the data is sufficient to outperform full-dataset training.
- Both GPT-4 evaluation and human evaluation confirm the superiority of DavIR.
- The performance of random sampling grows logarithmically with data volume, falling far behind DavIR.

### Comparison with Other Coreset Selection Methods (Gemma-2B, AlpacaEval)

| Method | 3K | 5K | 7K | 10K |
|------|-----|-----|-----|------|
| Random | 10.6 | 15.9 | 17.0 | 17.6 |
| Full (52K) | - | - | - | ~18 |
| EL2N (Highest) | 10.0 | 11.3 | 12.4 | 14.3 |
| Forgetting (Highest) | 9.5 | 13.4 | 16.7 | 18.2 |
| DataInf (Highest) | 10.3 | 15.9 | 18.7 | **18.8** |
| RHO (Highest) | 9.9 | 14.5 | 15.8 | ~16 |
| **DavIR** | **~12** | **~17** | **~19** | **~19** |

- DavIR is the **only method that consistently outperforms the full-dataset baseline across all data volumes**.
- The gap is small in the low-data regime (<5K), but the advantage is pronounced at higher data volumes.

### DavIR-DPO Results

| DPO Variant | Pearson Correlation with Response Length Difference |
|----------|------------------------|
| Vanilla DPO | 0.38 |
| IPO | -0.10 |
| AOT | 0.12 |
| **DavIR-DPO** | **0.07** |

- DavIR-DPO has the lowest dependency on length difference (0.07 vs 0.38).
- On Zephyr-7B-SFT, DavIR-DPO achieves an **8%** relative improvement on AlpacaEval (length-controlled).

### Data Mixing Experiments

- Mixing the DavIR-selected Alpaca subset with GSM8K data for training effectively balances open-domain QA and mathematical reasoning capabilities.
- Mixing the full Alpaca dataset with GSM8K, conversely, leads to conflicts in capabilities.

## Highlights & Insights

1. **A tiny normalization change yields immense gains**: Changing the denominator from token count to the reference model loss appears simple but produces dramatic results, reflecting a deep understanding of the problem's essence.
2. **Theory-practice closed loop**: Establishing theoretical connections from DPO implicit reward $\rightarrow$ discovering the length dependency issue $\rightarrow$ proposing the normalization solution $\rightarrow$ extending normalization back to DPO.
3. **Precise quantification of "learnability"**: The DavIR score directly reflects how much the model can learn through training (relative to its existing capability), serving as a model-centric selection standard.
4. **Computationally efficient**: Only requires two forward inference passes (losses of base and reference models), without needing gradients/Hessians (unlike DataInf) or the ChatGPT API.
5. **Statistically rigorous**: Estimates 95% confidence intervals via bootstrap sampling and performs t-tests, providing ample evidence of statistical significance.

## Limitations & Future Work

1. **Requires full training of a reference model**: A reference model $\pi_{\text{ref}}$ must first be trained on $D_{\text{full}}$, which increases upfront computational costs.
2. **Limited evaluation scope**: Mainly validated on English instruction-following datasets such as Alpaca/LIMA, leaving multilingual or long-context scenarios unexplored.
3. **Fewer DavIR-DPO experiments**: Validated only on a single model (Zephyr); the robustness of its advantages requires confirmation through more extensive experimentation.
4. **Hypothesis dependency**: The superficial alignment hypothesis may not hold in all scenarios (e.g., domains requiring deep knowledge acquisition).
5. **Iterative DavIR unexplored**: Theoretically, it can be executed iteratively (selection $\rightarrow$ training $\rightarrow$ updating reference $\rightarrow$ re-selection), but this is not experimented with in the paper.

## Related Work & Insights

- **Coreset Selection**: RHO (Mindermann et al. 2022), CRAIG, DataInf, EL2N, Forgetting Score, etc.
- **LLM Post-training Data Selection**: LIMA (Zhou et al. 2023) manual annotation, AlpaGasus (Chen et al. 2023) GPT scoring, Instruction Mining validation loss.
- **DPO Variants**: IPO (Azar et al. 2023), EXO (Ji et al. 2024), AOT (Melnyk et al. 2024), SPPO (Wu et al. 2024), etc.
- **Pre-training Data Selection**: DoReMi (Xie et al. 2023), DSIR (Xie et al. 2023), DRO (Oren et al. 2019).

## Rating

⭐⭐⭐⭐⭐ (5/5)

- **Novelty**: ⭐⭐⭐⭐⭐ The normalization change is simple but insightful. The theoretical chain of RHO $\rightarrow$ DavIR $\rightarrow$ DavIR-DPO is elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multiple model families, datasets, baseline comparisons, and rigorous statistical testing.
- **Writing Quality**: ⭐⭐⭐⭐ The problem definition is clear, but notations are heavy and the text is slightly verbose.
- **Value**: ⭐⭐⭐⭐⭐ Low computational cost with significant effect, directly applicable to any LLM post-training data selection scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Meta-rater: A Multi-dimensional Data Selection Method for Pre-training Language Models](metarater_a_multidimensional_data_selection_method.md)
- [\[ACL 2025\] Large Vocabulary Size Improves Large Language Models](large_vocabulary_size_improves_large_language_models.md)
- [\[ACL 2025\] Retrofitting Large Language Models with Dynamic Tokenization](retrofitting_large_language_models_with_dynamic_tokenization.md)
- [\[ACL 2025\] Data Whisperer: Efficient Data Selection for Task-Specific LLM Fine-Tuning via Few-Shot In-Context Learning](data_whisperer_data_selection.md)
- [\[ACL 2025\] Stealing Training Data from Large Language Models in Decentralized Training through Activation Inversion Attack](stealing_training_data_from_large_language_models_in_decentralized_training_thro.md)

</div>

<!-- RELATED:END -->
