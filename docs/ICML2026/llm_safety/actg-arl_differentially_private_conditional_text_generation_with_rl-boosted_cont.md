---
title: >-
  [Paper Note] ACTG-ARL: Differentially Private Conditional Text Generation with RL-Boosted Control
description: >-
  [ICML 2026][LLM Safety][Private Synthetic Data] This paper proposes ACTG, a hierarchical framework that decomposes private text generation into two sub-tasks: feature learning and conditional text generation. It further…
tags:
  - "ICML 2026"
  - "LLM Safety"
  - "Private Synthetic Data"
  - "Conditional Text Generation"
  - "Attribute Control"
  - "Instruction Following"
  - "Reward Hacking"
date: 2026-05-08
content_hash: cdda168ccd318975
---

# ACTG-ARL: Differentially Private Conditional Text Generation with RL-Boosted Control

**Conference**: ICML 2026  
**arXiv**: [2510.18232](https://arxiv.org/abs/2510.18232)  
**Code**: https://github.com/actg-arl/ACTG-ARL  
**Area**: Differential Privacy / Text Generation / RL Alignment  
**Keywords**: Private Synthetic Data, Conditional Text Generation, Attribute Control, Instruction Following, Reward Hacking

## TL;DR
This paper proposes ACTG, a hierarchical framework that decomposes private text generation into two sub-tasks: feature learning and conditional text generation. It further introduces Anchored RL, which enhances the instruction-following capability of the conditional generator by mixing reinforcement learning objectives with SFT anchors based on Best-of-N samples. While maintaining text fidelity, it achieves a 20% MAUVE improvement over prior work on biomedical data.

## Background & Motivation

**Background**
Modern AI applications rely on vast amounts of user data (mobile input, recommendation history, dialogue preferences, etc.), which carry high privacy risks. Generating private synthetic data is a promising paradigm that allows downstream tasks to reuse synthetic data without additional privacy costs. DP synthetic text is a research hotspot, but existing work mainly focuses on generating static datasets, overlooking the practical need for fine-grained control.

**Limitations of Prior Work**
1.  **Limitations of CTCL**: It relies on pre-trained general topic models that may mismatch with private domain data, forcing the classification of nuanced text into coarse-grained topics, leading to inaccurate topic inference. When the dataset is small relative to the number of topics, histograms contain many null values, causing signals to be submerged in noise after denoising.
2.  **Trade-off between Control and Fidelity**: Traditional RL optimization leads to reward hacking, where the model learns to generate outputs that formally satisfy constraints but suffer from degraded text quality (e.g., TL;DR-style summaries).

**Key Challenge**
The distribution matching objective encourages sampling from high-density regions of $P(X,Y)$ (areas where the model is already confident), whereas the value of data augmentation stems from low-density regions (model uncertainty boundaries or under-covered groups)—this leads to a goal misalignment between the generator and the augmentation task.

**Goal**
1.  Construct a modular framework to identify optimal configurations through systematic ablation.
2.  Improve the instruction-following capability of the conditional generator while maintaining privacy.

**Key Insight**
Starting from "attribute conditioning," structured tabular schemas are utilized as features, combined with a DP feature generator and a DP fine-tuned conditional generator. Furthermore, reinforcement learning is integrated with feature constraints to construct verifiable reward signals.

**Core Idea**
Hierarchical decomposition: First, extract modeled features $\mathcal{D}_{\text{priv}}^f$ from private data and use a DP tabular synthesizer to generate private features $\mathcal{D}_{\text{syn}}^{\tilde{f}}$. Second, use DP fine-tuning to learn the conditional mapping from features to text. Finally, use Anchored RL with Best-of-N data as SFT anchors to prevent RL drift, achieving hybrid optimization with $\mathcal{L}=\mathcal{L}_{\text{RL}}+\gamma\cdot\mathcal{L}_{\text{SFT}}$.

## Method

### Overall Architecture
The framework is divided into three stages:

**Phases 0-2 (ACTG)**:
1.  Feature Extraction: Use an Oracle LLM to extract a structured attribute matrix $D_{\text{priv}}^f$ from private text $D_{\text{priv}}^x$, containing $K$ fields with predefined options.
2.  Private Feature Generation (Privacy Budget $\varepsilon_1$): Use AIM (Advanced Information Management) for differentially private synthesis of tabular features to generate $D_{\text{syn}}^{\tilde{f}}$.
3.  Private Conditional Text Generation (Privacy Budget $\varepsilon_2$): Perform DP fine-tuning on (feature, text) pairs to learn $G_{x|f}$ for conditional text generation.

**Phases 3-4 (Anchored RL)**:
4.  Best-of-N Anchor Data: For each feature $f\sim G_f$, generate $N$ text candidates from $G_{x|f}$, select the best one based on Instruction Following Accuracy (IFAcc) to form the SFT anchor $D_{\text{SFT}_N}$ (with no additional privacy cost).
5.  Hybrid Objective Training: Starting from the $G_{x|f}$ checkpoint, jointly optimize using $\mathcal{L}_{\text{RL}}+\gamma\cdot\mathcal{L}_{\text{SFT}}$. A linear decay strategy is applied to $\gamma$ (high initially to maintain fidelity, gradually decreasing to allow instruction-following improvement).

### Key Designs

1.  **Hierarchical Decomposition + Structured Attribute Schema**:
    - **Function**: Decomposes the conditional generation problem into two tractable sub-problems and replaces general topics with domain-specific attribute schemas.
    - **Mechanism**: The first layer learns the feature marginal distribution (in a low-dimensional tabular space, using the mature AIM synthesizer, which is more efficient for the privacy budget); the second layer learns the text distribution conditioned on features (using DP fine-tuning). Attribute schemas are designed by an Oracle LLM or experts on private data to capture key dimensions, avoiding domain mismatch and sparse histogram issues caused by general topics in CTCL.
    - **Design Motivation**: Direct the privacy budget toward critical information and align with the natural structural hierarchy of the data.

2.  **Anchored RL for Reward Hacking Prevention**:
    - **Function**: Simultaneously improves instruction following (IFAcc) and maintains text fidelity (MAUVE), avoiding the reward hacking typical of standard PPO.
    - **Mechanism**: (i) Construct SFT anchors $D_{\text{SFT}_N}$ by sampling from $G_{x|f}$ itself using Best-of-N (no privacy cost since the model is already privately fine-tuned); (ii) Use a hybrid loss $\mathcal{L}=\mathcal{L}_{\text{RL}}+\gamma(t)\mathcal{L}_{\text{SFT}}$ during the RL phase to anchor the model near the reference distribution; (iii) Linear decay of $\gamma(t)$—strong fidelity initially, gradually relaxed for instruction-following gains.
    - **Design Motivation**: Adapt the "reference KL" concept from RLHF to private text generation; using self-sampled SFT anchors maintains quality without leaking additional privacy.

3.  **Instruction Following Accuracy as a Verifiable Reward**:
    - **Function**: Formalizes "compliance with attribute constraints" as an automated reward signal.
    - **Mechanism**: Use an Oracle LLM to back-extract attributes from generated text and calculate $\text{IFAcc}=\mathbb{E}_f[\frac{1}{K}\sum_{k=1}^K\mathbb{I}(f_k=\hat{f}_k)]$. This metric is used as the reward in the RL phase and for Best-of-N filtering.
    - **Design Motivation**: The structured attribute space naturally provides a verifiable, automatically evaluable target signal, serving as a clear reward for RL in generation tasks.

### Loss & Training
**DP Accounting**: The total privacy budget $(\varepsilon,\delta)$ consists of two stages $\varepsilon=\varepsilon_1+\varepsilon_2$. For each total budget $\varepsilon\in\{1,4,\infty\}$, the $(\varepsilon_1, \varepsilon_2)$ split is tuned independently; $\delta=1/(n\log n)$. The RL phase uses a hybrid loss $\mathcal{L}=\mathcal{L}_{\text{RL}}+\gamma(t)\mathcal{L}_{\text{SFT}}$ with linear decay of $\gamma(t)$.

## Key Experimental Results

### Main Results

| Dataset | Method | MAUVE | F1 Class | NTP Acc | IFAcc | $d_{\text{JS}}^f$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| bioRxiv(ε=4) | Aug-PE | 0.68 | 0.72 | - | - | 0.15 |
| | vanilla DP-FT | 0.62 | 0.68 | 0.41 | 0.53 | 0.18 |
| | CTCL | 0.64 | 0.70 | 0.42 | 0.48 | 0.16 |
| | **Ours** (ACTG) | 0.73 | 0.76 | 0.56 | 0.53 | 0.09 |
| | **Ours** (ACTG-ARL) | **0.74** | **0.79** | **0.58** | **0.62** | **0.08** |
| PMC-Patients(ε=4) | CTCL | 0.59 | 0.64 | 0.38 | 0.48 | 0.20 |
| | **Ours** (ACTG) | **0.71** | 0.75 | 0.51 | 0.50 | 0.10 |
| | **Ours** (ACTG-ARL) | 0.70 | **0.77** | **0.53** | **0.58** | **0.09** |

### Ablation Study

| Component | Removal/Replacement | MAUVE | IFAcc | $d_{\text{JS}}^f$ | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Feature Model | Use CTCL general topics | 0.64 | 0.48 | 0.16 | Significant performance drop with general topics |
| Feature Generator | Replace AIM with DP-FT | 0.68 | 0.50 | 0.12 | AIM performs better (less budget waste) |
| Conditional Generator | Replace DP-FT with Direct Prompting | 0.61 | 0.55 | 0.14 | Fine-tuned version is more stable |
| Full ACTG | - | 0.73 | 0.53 | 0.09 | Baseline |
| +Standard PPO | No Anchor | 0.42 | 0.68 | 0.22 | Severe reward hacking; MAUVE collapses |
| +Anchored RL | Full Method | 0.74 | 0.62 | 0.08 | Improved IFAcc while maintaining fidelity |

### Key Findings
- **Feature Design is Pivotal**: Structured attribute schemas significantly outperform general topics, increasing MAUVE from 0.64 to 0.73 (+14%) on bioRxiv.
- **Tabular vs. Text Feature Generation**: AIM (tabular) saves privacy budget compared to DP-FT (text), resulting in smaller error $d_{\text{JS}}^f$ (0.12 vs. 0.14).
- **Severe RL Reward Hacking**: Standard PPO destroys MAUVE from 0.73 to 0.42, whereas Anchored RL recovers it to 0.74 (with IFAcc increasing from 0.53 to 0.62).
- **Best-of-N Effectiveness**: Selecting from $N=5$ or $10$ candidates produces high-quality, diverse SFT datasets without increasing privacy costs.
- **Privacy Budget Splitting**: At $\varepsilon=4$, the optimal split is approximately $(\varepsilon_1,\varepsilon_2)\approx(1.5,2.5)$ or $(2,2)$, indicating that both stages require sufficient budget.

## Highlights & Insights
- **Elegance of Hierarchical Design**: Decomposing complex end-to-end DP text generation into low-dimensional tabular synthesis + conditional text generation improves modularity and allows using optimal tools for each module (AIM vs. LLM fine-tuning).
- **Practical Ingenuity of Anchored RL**: Extracting references from the model itself via Best-of-N avoids private data access and has zero privacy cost, yet effectively prevents reward hacking—a clever adaptation of RLHF for privacy scenarios.
- **Attribute Matching as Reward**: Leveraging the structured attribute space as the basis for the IFAcc metric transforms the text understanding problem into a formalized attribute extraction task, facilitating automation and verification.

## Limitations & Future Work
- **Limited Model and Data Scope**: Experiments were only conducted on gemma-3-1b-pt in the biomedical domain, not covering other fields like law, finance, or dialogue, nor exploring the performance of larger models.
- **A Priori Attribute Space Design**: The paper does not discuss how to automate the design of optimal attribute schemas, currently relying on manual effort or an Oracle LLM, which may be a bottleneck for application.
- **Privacy Budget Split Optimization**: The $(\varepsilon_1,\varepsilon_2)$ split was determined via hyperparameter tuning, lacking theoretical guidance or adaptive strategies.

## Related Work & Insights
- **vs. DP-FT**: Directly applying DP fine-tuning to LLMs without considering conditional control or structured features results in significant quality degradation. This work improves upon it through hierarchy and attribute conditioning.
- **vs. CTCL**: Both use conditioning ideas, but while CTCL uses fixed general topics, this work uses data-specific attribute schemas, significantly improving schema-data alignment.
- **vs. Aug-PE (Private Evolution)**: PE refines through LLM iteration, while this work uses direct fine-tuning + RL. ACTG-ARL is more stable in the bio domain.

## Rating
- Novelty: ⭐⭐⭐⭐ The hierarchical framework and Anchored RL are both new contributions; the zero-cost anchor idea from Best-of-N is ingenious.
- Experimental Thoroughness: ⭐⭐⭐⭐ Two biomedical datasets and multidimensional evaluation with thorough ablation. A slight weakness is the lack of diverse dataset families.
- Writing Quality: ⭐⭐⭐⭐ Clear problem description, complete algorithm pseudocode, and sufficient experimental details.
- Value: ⭐⭐⭐⭐ Practical needs in DP synthetic text are addressed (+20% MAUVE); this is the first systematic exploration of conditional control in privacy applications, offering high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Differentially Private Synthetic Text Generation for Retrieval-Augmented Generation (RAG)](../../ACL2026/llm_safety/differentially_private_synthetic_text_generation_for_retrieval-augmented_generat.md)
- [\[ICML 2026\] Privacy Amplification in Differentially Private Zeroth-Order Optimization with Hidden States](privacy_amplification_in_differentially_private_zeroth-order_optimization_with_h.md)
- [\[ICML 2026\] Differentially Private Preference Data Synthesis for Large Language Model Alignment](differentially_private_preference_data_synthesis_for_large_language_model_alignm.md)
- [\[ICML 2026\] Optimizing Token Choice for Code Watermarking: An RL Approach](optimizing_token_choice_for_code_watermarking_an_rl_approach.md)
- [\[ICML 2026\] AliMark: Enhancing Robustness of Sentence-Level Watermarking Against Text Paraphrasing](alimark_enhancing_robustness_of_sentence-level_watermarking_against_text_paraphr.md)

</div>

<!-- RELATED:END -->
