---
title: >-
  [Paper Note] Mitigating Catastrophic Forgetting in Target Language Adaptation of LLMs via Source-Shielded Updates
description: >-
  [ACL 2026][Multilingual & Machine Translation][Catastrophic forgetting] This paper proposes Source-Shielded Updates (SSU), a column-wise freezing strategy driven by importance scores from source data. During continual pr…
tags:
  - "ACL 2026"
  - "Multilingual & Machine Translation"
  - "Catastrophic forgetting"
  - "language adaptation"
  - "selective parameter updates"
  - "column-wise freezing"
  - "source knowledge protection"
date: 2026-05-08
content_hash: 4ac9065c07a942a0
---

# Mitigating Catastrophic Forgetting in Target Language Adaptation of LLMs via Source-Shielded Updates

**Conference**: ACL 2026  
**arXiv**: [2512.04844](https://arxiv.org/abs/2512.04844)  
**Code**: [GitHub](https://github.com/gucci-j/ssu)  
**Area**: LLM Evaluation  
**Keywords**: Catastrophic forgetting, language adaptation, selective parameter updates, column-wise freezing, source knowledge protection

## TL;DR
This paper proposes Source-Shielded Updates (SSU), a column-wise freezing strategy driven by importance scores from source data. During continual pre-training (CPT) with unlabeled target language data, SSU reduces source language performance degradation from 20.3% in full fine-tuning to 3.4% while maintaining target language performance comparable to or better than full fine-tuning.

## Background & Motivation

**Background**: Expanding the language coverage of LLMs is crucial for global accessibility. The standard practice involves Continual Pre-training (CPT) on target language data, which frequently leads to catastrophic forgetting, particularly harming the core chat and safety capabilities of instruction-tuned models.

**Limitations of Prior Work**: (1) Low-resource languages lack instruction-tuning data, and machine-translated data yields unstable results, forcing reliance on unlabeled text for adaptation; (2) Full Fine-Tuning (FFT) leads to an average performance drop of 20.3% in source languages for 7B models and 22.3% for 13B models; (3) Post-processing methods (e.g., model merging, task vectors) largely fail to effectively mitigate forgetting.

**Key Challenge**: Unlabeled raw text lacks chat templates and is incompatible with the training format of instruction-following models. Existing selective update methods based on target data signals optimize based on this incompatible format, which may instead damage the base capabilities of the model.

**Goal**: To proactively protect source knowledge during the CPT stage, enabling the model to learn the target language while retaining its original instruction-following, chat, and safety capabilities.

**Key Insight**: Shift from target-focused to source-focused parameter selection—identify parameters critical to source knowledge and freeze them before CPT to prevent forgetting at the source.

**Core Idea**: Calculate parameter importance scores using a small amount of source data (500 samples), aggregate them by column, and freeze the top 50% most important columns to ensure that complete feature transformation pathways remain intact.

## Method

### Overall Architecture
SSU consists of three stages: (1) calculating importance scores for each parameter using source data and the Wanda scoring method; (2) aggregating scores column-wise to generate a column-level freezing mask; (3) performing CPT on unlabeled target language data while applying the mask to freeze critical columns during gradient updates.

### Key Designs

1.  **Parameter Importance Scoring**:
    - **Function**: Identifies weight parameters critical for maintaining source language capabilities.
    - **Mechanism**: Adopts the importance score $s_{ij} = |\theta_{ij}| \cdot \|X_j\|_2$ from the Wanda pruning method, where weight magnitude is multiplied by the L2 norm of the corresponding input activation. This requires only 500 source data samples and involves no gradient computation.
    - **Design Motivation**: Source-data-driven scoring directly aligns with the goal of protecting source knowledge; combining weight magnitude and activation frequency is more reliable than using either signal alone (validated by SSU-Mag ablation).

2.  **Column-wise Masking**:
    - **Function**: Converts element-wise importance scores into a structured freezing mask.
    - **Mechanism**: Scores are summed for each column $j$ of the weight matrix $\theta \in \mathbb{R}^{d_{out} \times d_{in}}$ as $S_j = \sum_i s_{ij}$. After sorting by $S_j$, the top-$k$% (default 50%) of columns are frozen. Mask values for frozen columns are set to 0, otherwise 1.
    - **Design Motivation**: Column-wise freezing ensures complete feature transformation pathways—freezing an entire column means the corresponding output dimension remains unchanged, similar to preserving load-bearing columns during a building renovation. Element-wise freezing disrupts feature transformations and leads to catastrophic forgetting (confirmed by experiments).

3.  **Masked Continual Pre-training**:
    - **Function**: Learns the target language while shielding source knowledge.
    - **Mechanism**: Uses the standard causal language modeling objective but applies a static mask during backpropagation: $\theta_{ij} \leftarrow \theta_{ij} - \eta \cdot b_{ij} \cdot \nabla_{\theta_{ij}} L$, where $b_{ij} \in \{0, 1\}$ is the mask value. Embedding layers and LM heads are not constrained by the mask and are fully updated.
    - **Design Motivation**: Static masks are simple to implement and orthogonal to other mitigation methods (regularization, replay); they avoid dynamic mask computation during training, reducing computational overhead.

### Loss & Training
The model is trained on 200M tokens of target language data using the standard causal language modeling loss. Chat templates are removed prior to training to support unlabeled data.

## Key Experimental Results

### Main Results (7B Model, Average Source Language Metrics)

| Method | IFEval | AlpacaEval2 | MT-Bench | GSM8K | Safety | Source Degradation (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Source | 0.675 | 32.6 | 3.98 | 0.796 | 0.851 | 0.0 |
| FFT | 0.456 | 10.4 | 3.48 | 0.608 | 0.797 | -20.3 |
| HFT | 0.621 | 17.6 | 3.83 | 0.677 | 0.826 | -8.0 |
| **Ours (SSU)** | **0.669** | **27.0** | **3.96** | **0.752** | **0.850** | **-3.4** |

### Ablation Study

| Configuration | IFEval | Description |
| :--- | :--- | :--- |
| SSU-Wanda | 0.669 | Full method (source data + Wanda scoring + column-wise freezing) |
| SSU-Rand | 0.608 | Random column freezing without source data guidance |
| SSU-Mag | 0.570 | Weight magnitude scoring only, lacking activation signals |
| Element-wise | ~0.45 | Disrupts feature transformation, leading to performance collapse |

### Key Findings
- SSU consistently outperforms all baselines across core instruction-following and safety tasks, making it the only method to excel in both source language retention and target language improvement.
- Regarding target language performance, SSU exceeds FFT across all benchmarks on the 7B scale and most benchmarks on the 13B scale.
- The significant performance gap between column-wise and element-wise freezing validates the importance of structured protection.
- SSU avoids the code-mixing issues observed in methods like HFT.

## Highlights & Insights
- The "source-focused" rather than "target-focused" parameter selection paradigm is the core insight—protecting existing knowledge is more important than choosing what to update.
- The analogy for column-wise freezing is highly intuitive: "preserving load-bearing columns during renovation"—protecting complete feature pathways rather than isolated parameter points.
- Importance scoring requires only 500 source samples, resulting in minimal overhead and high practical utility.

## Limitations & Future Work
- Currently validated only on the OLMo 2 series; further verification on more architectures is needed.
- A fixed 50% freezing ratio might not be optimal for all languages; adaptive freezing ratios are worth exploring.
- Combination with parameter-efficient methods like LoRA has yet to be investigated.

## Related Work & Insights
- **vs HFT**: HFT randomly freezes sub-layer components and lacks principled protection; SSU uses source-data-driven scoring to precisely locate critical parameters.
- **vs GMT**: GMT selects based on target data gradients, which are unreliable for unlabeled text; the SSU source-focused strategy is more robust.
- **vs AdaLoRA**: AdaLoRA maintains knowledge well but offers limited target improvement; SSU balances both ends effectively.

## Rating
- Novelty: ⭐⭐⭐⭐ The source-focused column-wise freezing strategy is novel and theoretically supported.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluated across 5 languages, 2 model scales, and multiple dimensions.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear logic and rigorous derivation of motivations.
- Value: ⭐⭐⭐⭐⭐ Directly addresses a core pain point in adaptation for low-resource languages.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Efficient Low-Resource Language Adaptation via Multi-Source Dynamic Logit Fusion](efficient_low-resource_language_adaptation_via_multi-source_dynamic_logit_fusion.md)
- [\[ACL 2026\] Exploring Two-Phase Continual Instruction Fine-tuning for Multilingual Adaptation in Large Language Models](exploring_continual_fine-tuning_for_enhancing_language_ability_in_large_language.md)
- [\[ICML 2026\] Toward Robust Multilingual Adaptation of LLMs for Low-Resource Languages](../../ICML2026/multilingual_mt/toward_robust_multilingual_adaptation_of_llms_for_low-resource_languages.md)
- [\[ACL 2026\] TLPO: Token-Level Policy Optimization for Mitigating Language Confusion in Large Language Models](tlpo_token-level_policy_optimization_for_mitigating_language_confusion_in_large_.md)
- [\[ACL 2026\] Mitigating Extrinsic Gender Bias for Bangla Classification Tasks](mitigating_extrinsic_gender_bias_for_bangla_classification_tasks.md)

</div>

<!-- RELATED:END -->
