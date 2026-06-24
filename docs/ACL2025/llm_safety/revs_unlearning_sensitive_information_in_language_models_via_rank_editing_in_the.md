---
title: >-
  [Paper Note] REVS: Unlearning Sensitive Information in Language Models via Rank Editing in the Vocabulary Space
description: >-
  [ACL 2025][LLM Safety][Machine Unlearning] This paper proposes REVS, a gradient-free model editing method. By locating neurons in the FF2 layer that are most strongly associated with sensitive tokens and projecting them into the vocabulary space, it iteratively lowers the rank of target tokens. On three types of sensitive data (SSN/Email/URL), its Unlearning Score significantly outperforms six baselines (89.58 vs 36.98) with almost zero cost to general capabilities (MMLU 61.0…
tags:
  - "ACL 2025"
  - "LLM Safety"
  - "Machine Unlearning"
  - "Vocabulary Space Editing"
  - "Rank Manipulation"
  - "Gradient-Free"
  - "Sensitive Information Protection"
date: 2026-05-08
content_hash: 699472d7886e0f0d
---

# REVS: Unlearning Sensitive Information in Language Models via Rank Editing in the Vocabulary Space

**Conference**: ACL 2025  
**arXiv**: [2406.09325](https://arxiv.org/abs/2406.09325)  
**Code**: [https://github.com/tomerashuach/REVS](https://github.com/tomerashuach/REVS)  
**Area**: LLM Safety / Machine Unlearning  
**Keywords**: Machine Unlearning, Vocabulary Space Editing, Rank Manipulation, Gradient-Free, Sensitive Information Protection

## TL;DR

This paper proposes REVS, a gradient-free model editing method. By locating neurons in the FF2 layer that are most strongly associated with sensitive tokens and projecting them into the vocabulary space, it iteratively lowers the rank of target tokens. On three types of sensitive data (SSN/Email/URL), its Unlearning Score significantly outperforms six baselines (89.58 vs 36.98) with almost zero cost to general capabilities (MMLU 61.05 $\rightarrow$ 60.87), while remaining highly robust to Logit-Lens and Delta extraction attacks.

## Background & Motivation

**Background**: LLMs memorize sensitive information in training data (emails, URLs, SSNs, etc.), making it possible to reconstruct complete PII given relevant prompts. Existing defenses fall into two categories: data-side approaches such as differential privacy or data cleaning (which require retraining), and model-side approaches like machine unlearning or model editing (which modify parameters via post-processing).

**Limitations of Prior Work**: (1) Differential privacy degrades model quality, and data cleaning is costly and requires retraining whenever new PII is discovered. (2) In-context learning (ICL) unlearning and gradient ascent (GA) only superficially suppress generation while retaining information in weights, remaining vulnerable to white-box extraction attacks. (3) Model editing methods like MEMIT are designed for "factual modification" and perform poorly when applied to "unlearning". (4) Existing evaluations of PII unlearning lack datasets that contain naturally memorized sensitive information.

**Key Challenge**: Gradient-based methods involve broad modifications that easily impair general capabilities, whereas precise editing methods are not designed for unlearning objectives.

**Key Insight**: The FF2 weight of the Transformer MLP serves as a "key-value memory" (Geva et al., 2021), where each column (neuron) projected to the vocabulary space corresponds to the logit contribution of specific tokens. If neurons that promote sensitive tokens can be precisely identified and their ranks directly lowered in the vocabulary space, gradient computation can be bypassed.

**Core Idea**: Instead of optimizing a loss function, this method directly manipulates token ranks in the vocabulary space—lowering target sensitive tokens from rank 1 to a designated target rank.

## Method

### Overall Architecture

Input: Model $M$, prompt $P$ containing sensitive information, target token $t$, target rank $r_d$, and maximum number of edited neurons $n_{max}$. Output: Edited model, where the rank of $t$ at the position corresponding to $P$ is reduced below $r_d$. The entire process consists of two stages: **Localization** (identifying which layers/neurons are responsible for $t$) $\rightarrow$ **Editing** (reducing the logit rank of $t$ in the vocabulary space). Prior to editing, there is a **target token selection** step, which selects only the 2 rarest tokens in the sequence as targets for unlearning.

### Key Designs

1. **Layer Selection**

    - **Function**: Screening layers from all Transformer layers that are strongly associated with the target token.
    - **Mechanism**: For each layer $l$, the FF2 output $\vec{h}_l$ is projected into the vocabulary space via the unembedding matrix $U$, and the rank of the target token $t$ is computed as $r_{t,l} = \text{rank}(t, U\vec{h}_l)$. Layers where the rank is below a threshold $r_h$ are selected.
    - **Design Motivation**: A higher rank (smaller numerical value) indicates that the layer makes a greater contribution to generating $t$. Editing only these layers minimizes perturbation to the model.

2. **Neuron Selection**

    - **Function**: Identifying the columns (neurons) of FF2 inside the selected layers that are most strongly associated with $t$.
    - **Mechanism**: A two-step hybrid screening: (a) First, select the top-$k$ neurons with the highest activation values $a_j$ (context relevance); (b) Among these $k$ neurons, sort them by $\text{rank}(t, U\vec{n}_j)$ and incrementally select them until the overall layer rank $r_t > r_h$ or $n_{max}$ is reached.
    - **Design Motivation**: Selecting solely by activation or solely by token association is insufficient. Activations ensure context relevance (the neuron is indeed activated by the current prompt), while token association ensures semantic relevance (the neuron indeed encodes the target token). Ablation studies verify that the hybrid strategy achieves an Unlearning Score of 75.2, outperforming pure activation (68.8), pure rank (22.4), gradient-based (71.1), and random selection (0).

3. **Neuron Editing**

    - **Function**: Iteratively adjusting the selected neurons $\vec{n}_j$ in the vocabulary space until the rank of $t$ falls below $r_n$.
    - **Mechanism**: For each selected neuron $\vec{n}_j$: (a) Project to the vocabulary space $\vec{v} = U\vec{n}_j$; (b) Set the logit of $t$ to a low value $l_t$ (initially $-10$); (c) Project back to the neuron space using the pseudo-inverse $\vec{n}_j^* = U^\dagger \vec{v}$; (d) Project back to the vocabulary space to check the rank of $t$. Since $U$ is not invertible, the pseudo-inverse introduces approximation errors, necessitating an iterative process: if the rank is not low enough, $l_t \leftarrow l_t \times 1.3$; if it is too low, $l_t \leftarrow l_t \times 0.8$, until convergence within an $\epsilon$ tolerance.
    - **Design Motivation**: Directly zeroing out neurons (as in DEPN) destroys their contributions to other tokens, leading to degraded specificity. REVS only alters the rank of $t$, preserving the logit distribution of other tokens.

4. **Target Token Selection**

    - **Function**: Selecting the minimal token subset $T \subseteq S$ from a sensitive sequence $S$ to unlearn.
    - **Mechanism**: Select $|T|=2$ rarest tokens (using token IDs to approximate rarity, where larger IDs mean rarer tokens), excluding common parts like `@email.com`, `http`, `://`, etc.
    - **Design Motivation**: Reconstructing a complete sensitive information sequence requires generating the entire token sequence; if two key rare tokens cannot be generated, the whole sequence becomes unrecoverable. This significantly reduces the editing footprint.

### Loss & Training

REVS is completely gradient-free. In each experiment, the model performs unlearning edits for all sensitive sequences first, and is then evaluated. Hyperparameters are optimized by maximizing the Unlearning Score (k=100) on the SSN and Email datasets. On Llama-3-8B, the maximum number of edited neurons per layer is 130 (SSN) / 45 (Email), with activation top-k=1000, and target rank $r_n = 105000$ (with a vocabulary size of 128,256).

## Key Experimental Results

### Main Results (Llama-3-8B, k=100)

| Method | Unlearning Score ↑ | Efficacy@100 ↑ | Generality@100 ↑ | Specificity ↑ | MMLU | GSM8K |
|------|---:|---:|---:|---:|---:|---:|
| FT-L | 36.98 | 63.88 | 50.35 | 24.33 | 60.99 | 46.62 |
| MEMIT | 24.72 | 30.70 | 23.90 | 22.67 | 61.02 | 46.17 |
| NPO-KL | 11.95 | 38.78 | 36.13 | 6.33 | 61.01 | 47.23 |
| RMU | 16.42 | 13.47 | 16.67 | 38.67 | 60.83 | 48.21 |
| Max-Entropy | 5.12 | 5.17 | 3.92 | 1.40 | 61.06 | 47.46 |
| Head-Projection | 2.98 | 3.08 | 2.95 | 4.17 | 61.06 | 46.92 |
| **REVS** | **89.58** | **98.88** | **89.67** | **82.17** | 60.87 | 44.20 |

*Results on the SSN dataset. Unlearning Score is the harmonic mean of Efficacy, Specificity, and Generality. REVS is statistically significantly superior to all baselines (Wilcoxon signed-rank, p<0.05).*

### Extraction Attack Resistance (Llama-3-8B, SSN, k=100)

| Method | Resistance Score ↑ | Logit-Lens@100 ↑ | Delta@100 ↑ | Perturb@100 ↑ |
|------|---:|---:|---:|---:|
| FT-L | 82.77 | 63.88 | 98.08 | 98.22 |
| MEMIT | 55.20 | 30.70 | 97.90 | 98.18 |
| NPO-KL | 61.63 | 38.78 | 98.47 | 95.08 |
| **REVS** | **99.27** | **98.88** | **98.92** | **100.00** |

*REVS achieves nearly perfect defense under all three attacks, obtaining a Resistance Score of 99.27, which far outperforms the runner-up FT-L (82.77).*

### Ablation Study

**Neuron Selection Strategy Ablation (GPT-J-6B, Email)**

| Selection Method | Unlearning Score | Resistance Score |
|----------|---:|---:|
| Random | 0 | 0 |
| Rank only | 22.4 | 29.2 |
| Activation only | 68.8 | 81.8 |
| Gradient-based | 71.1 | 78.2 |
| **Rank & Activation (hybrid)** | **75.2** | **84.2** |

**Editing Strategy Ablation: Zeroing vs Rank Editing (GPT-J-6B, Email)**

| Method | Unlearning Score | Efficacy@100 | Specificity | Resistance Score |
|------|---:|---:|---:|---:|
| ZERO (n=5) | 0 | 73.8 | 0 | 79.2 |
| ZERO (n=2) | 14.3 | 26.3 | 9.7 | 47.5 |
| ZERO (n=1) | 11.2 | 6.8 | 31.9 | 17.3 |
| **REVS** | **83.5** | **81.1** | **87.1** | **82.6** |

**Target Token Selection Strategy Ablation (GPT-J-6B, Email)**

| Token Selection | Unlearning Score | Efficacy@100 | Specificity | Resistance Score |
|-----------|---:|---:|---:|---:|
| Most Frequent | 55.6 | 67.7 | 47.22 | 70.6 |
| First | 65.6 | 86.8 | 52.77 | 77.2 |
| Random | 72.0 | 89.0 | 60.4 | 80.5 |
| **Rarest** | **75.2** | **96.1** | **61.8** | **83.9** |

### Key Findings

- **REVS leads comprehensively across all three data categories**: On SSN, it achieves an Unlearning Score of 89.58 (outperforming the nearest baseline FT-L at 36.98 by 2.4x), 62.37 on Email, and 44.25 on URL, all being state-of-the-art.
- **Almost perfect defense against extraction attacks**: It achieves a Resistance Score of 99.27 and a Perturb Attack score of 100.00 on SSN; and 82.80 on URL, significantly outperforming other methods.
- **Almost zero degradation in general capabilities**: MMLU drops slightly from 61.05 to 60.87 (-0.3%). GSM8K drops on the SSN dataset (47.83 $\rightarrow$ 44.20), which the authors attribute to SSN containing mostly numerical data.
- **Necessity of hybrid neuron selection**: Unlearning Score is only 22.4 with pure text ranking and 68.8 with pure activation, while the hybrid strategy reaches 75.2—demonstrating that a single signal is insufficient.
- **Zeroing out is infeasible**: Directly zeroing out neurons causes specificity to collapse (the specificity for ZERO n=5 is 0), showing that neurons are multi-functional and serve multiple tokens.
- **Selecting the rarest tokens is most effective**: Purging the 2 rarest tokens is sufficient to unlearn the entire sequence, which outperforms targeting the most frequent, the first, or random tokens.
- **Consistent trends on GPT-J-6B**: REVS leads on SSN (81.45) and Email (80.65) on GPT-J-6B as well, confirming the generalizability of the method.
- **Pareto dominance**: REVS Pareto-dominates FT-L on the efficacy-specificity trade-off curve, exhibiting almost no trade-off particularly on SSN.

## Highlights & Insights

- **An editing philosophy of "rank over value"**: There is no need to precisely tune the absolute logit values; the target token's rank simply needs to be sufficiently reduced. This bypasses the difficulty of precise logit control and offers greater tolerance to rank changes—as long as $t$ is outside the top-k. This paradigm can be extended to any scenario requiring the suppression of specific outputs (such as toxic content or copyrighted text).
- **Engineering wisdom in pseudo-inverse iteration**: Since the unembedding matrix $U$ is not square and thus non-invertible, projecting back with the pseudo-inverse $U^\dagger$ introduces approximation errors. Instead of searching for a better inverse, the authors employ a simple iterative multiplication factor (1.3/0.8) to achieve convergence, which is highly practical and robust.
- **"Unlearning 2 tokens nullifies the entire sequence"**: The key insight is that information recovery relies on reconstructing the complete sequence; thus, destroying only the 2 hardest-to-guess rare tokens is sufficient. This vastly reduces the extent of model modifications, which is the root cause of the high specificity.
- **Methodology for constructing naturally memorized datasets**: By leveraging the extraction benchmark subset of the Pile, Presidio PII detection, and verification of model-based generation, the authors construct datasets of genuinely memorized Emails/URLs. This pipeline serves as a valuable resource for future research on PII unlearning.

## Limitations & Future Work

- **Limited scale**: The method is only verified on 6B/8B models and has not been tested on 70B+ models, where information representation may be more distributed.
- **Limited data types**: Evaluation mainly focuses on emails, URLs, and SSNs (numbers), while other PII types like phone numbers or home addresses are not validated.
- **GSM8K degradation on SSN**: Editing numerical PII impairs mathematical reasoning (47.83 $\rightarrow$ 44.20), suggesting that editing numerical tokens is harder to achieve with high specificity.
- **English only**: The method has not been evaluated on other languages.
- **Limited attack models**: Only Logit-Lens, Delta, and Perturbation attacks are considered; probe-based or attention head-level extraction attacks are not evaluated.
- **Directions for improvement**: (1) Incorporating attention head analysis for more precise localization; (2) Designing specialized selection strategies for numerical tokens to avoid compromising general capabilities; (3) Exploring whether post-editing lightweight fine-tuning can restore GSM8K without compromising the unlearning efficacy.

## Related Work & Insights

- **vs MEMIT (Meng et al., 2023)**: MEMIT is a knowledge editing method that inserts or modifies facts via constrained optimization of FF2. Adapting it for unlearning yields limited performance (Unlearning Score of 24.72 vs REVS's 89.58), as MEMIT is not meant for rank manipulation. REVS's rank-centric perspective is better suited for unlearning tasks.
- **vs FT-L (Zhu et al., 2020)**: FT-L fine-tunes FF2 via gradient ascent with an $L_\infty$ constraint. It is the strongest baseline among gradient-based methods (Unlearning Score of 36.98), but suffers from poor specificity (24.33) and instability as the number of targets increases. REVS's gradient-free paradigm behaves superiorly in both stability and specificity.
- **vs DEPN (Wu et al., 2023)**: This is the most similar work, which also localizes neurons but directly zeroes them out. Ablation studies demonstrate that zeroing out causes specificity to collapse. REVS's "rank reduction over zeroing" is a crucial improvement.
- **vs RMU (Li et al., 2024)**: RMU applies directional perturbation in the representation space for concept-level unlearning (like WMDP). It performs poorly for PII unlearning (Unlearning Score of 16.42), as PII unlearning is a token-level problem rather than a concept-level one.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Editing token ranks in the vocabulary space is a novel paradigm that framing unlearning as rank manipulation instead of value optimization.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Outstandingly comprehensive coverage with 6 baselines, 3 datasets, 2 models, 3 attacks, and 4 sets of ablation studies.
- Writing Quality: ⭐⭐⭐⭐ The methodology is clearly described with complete algorithmic pseudocode, though the appendix is lengthy and some results are scattered.
- Value: ⭐⭐⭐⭐⭐ Direct practical value for GDPR-compliant PII unlearning, and constructs the first dataset of naturally memorized sensitive information.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] MEGen: Generative Backdoor into Large Language Models via Model Editing](megen_generative_backdoor_into_large_language_models_via_model_editing.md)
- [\[ACL 2025\] ReLearn: Unlearning via Learning for Large Language Models](relearn_unlearning_via_learning_for_large_language_models.md)
- [\[ACL 2025\] Private Memorization Editing: Turning Memorization into a Defense to Strengthen Data Privacy in Large Language Models](private_memorization_editing_turning_memorization_into_a_defense_to_strengthen_d.md)
- [\[AAAI 2026\] Cross-Modal Unlearning via Influential Neuron Path Editing in Multimodal Large Language Models](../../AAAI2026/llm_safety/cross-modal_unlearning_via_influential_neuron_path_editing_i.md)
- [\[ACL 2025\] Mamba Knockout for Unraveling Factual Information Flow](mamba_knockout_for_unraveling_factual_information_flow.md)

</div>

<!-- RELATED:END -->
