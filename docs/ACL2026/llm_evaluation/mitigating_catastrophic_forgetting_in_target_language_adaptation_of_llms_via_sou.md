---
title: >-
  [Paper Note] Mitigating Catastrophic Forgetting in Target Language Adaptation of LLMs via Source-Shielded Updates
description: >-
  [ACL 2026][LLM Evaluation][Catastrophic Forgetting] This paper proposes Source-Shielded Updates (SSU), a column-wise freezing strategy driven by source-data parameter importance scoring. During continual pre-training (CPT) using only unlabeled target-language data, SSU reduces source-language performance degradation from 20.3% (full fine-tuning) to 3.4%, while maintaining target-language performance on par with or superior to full fine-tuning.
tags:
  - ACL 2026
  - LLM Evaluation
  - Catastrophic Forgetting
  - Language Adaptation
  - Selective Parameter Update
  - Column-wise Freezing
  - Source Knowledge Preservation
date: 2026-05-08
content_hash: 20611042ab174029
---

# Mitigating Catastrophic Forgetting in Target Language Adaptation of LLMs via Source-Shielded Updates

**Conference**: ACL 2026
**arXiv**: [2512.04844](https://arxiv.org/abs/2512.04844)
**Code**: [GitHub](https://github.com/gucci-j/ssu)
**Area**: LLM Evaluation
**Keywords**: Catastrophic Forgetting, Language Adaptation, Selective Parameter Update, Column-wise Freezing, Source Knowledge Preservation

## TL;DR
This paper proposes Source-Shielded Updates (SSU), a column-wise freezing strategy driven by source-data parameter importance scoring. During continual pre-training (CPT) using only unlabeled target-language data, SSU reduces source-language performance degradation from 20.3% (full fine-tuning) to 3.4%, while maintaining target-language performance on par with or superior to full fine-tuning.

## Background & Motivation

**Background**: Extending the language coverage of LLMs is critical for global accessibility. The standard practice is continual pre-training (CPT) on target-language data, which, however, frequently leads to catastrophic forgetting—particularly degrading the core chat and safety capabilities of instruction-tuned models.

**Limitations of Prior Work**: (1) Low-resource languages lack instruction-tuning data, and machine-translated data yields inconsistent results, leaving unlabeled text as the only viable option for adaptation. (2) Full fine-tuning causes an average source-language performance drop of 20.3% on 7B models and 22.3% on 13B models. (3) Post-hoc methods such as model merging and task vectors largely fail to mitigate forgetting effectively.

**Key Challenge**: Raw unlabeled text lacks chat templates and is thus incompatible with the training format of instruction-tuned models. Selective update methods guided by target-data signals optimize based on this incompatible format, which may in turn undermine the model's foundational capabilities.

**Goal**: Proactively preserve source knowledge during CPT, enabling the model to acquire the target language while retaining its original instruction-following, chat, and safety capabilities.

**Key Insight**: Rather than deriving signals from target data, the method starts from source data—identifying parameters critical to source knowledge and freezing them prior to CPT, thereby preventing forgetting at the root.

**Core Idea**: A small set of source data (500 samples) is used to compute parameter importance scores; scores are aggregated column-wise, and the most important 50% of columns are frozen to ensure that complete feature transformation pathways remain intact.

## Method

### Overall Architecture
SSU proceeds in three stages: (1) compute per-parameter importance scores using source data and the Wanda scoring method; (2) aggregate scores column-wise and generate a column-level freezing mask; (3) perform CPT on unlabeled target-language data, applying the mask during gradient updates to freeze critical columns.

### Key Designs

1. **Parameter Importance Scoring**:

    - Function: Identify weight parameters critical to preserving source-language capabilities.
    - Mechanism: Adopts the importance score from the Wanda pruning method, $s_{ij} = |\theta_{ij}| \cdot \|X_j\|_2$, i.e., the absolute weight value multiplied by the $L_2$ norm of the corresponding input activation. Only 500 source samples are required; no gradient computation is involved.
    - Design Motivation: Source-data-driven scoring directly aligns with the objective of protecting source knowledge. Combining weight magnitude and activation frequency is more reliable than using either signal alone (ablations confirm that SSU-Mag, which uses only weight magnitude, yields noticeably lower performance).

2. **Column-wise Mask Generation**:

    - Function: Convert element-wise importance scores into a structured freezing mask.
    - Mechanism: For each column $j$ of the weight matrix $\theta \in \mathbb{R}^{d_{out} \times d_{in}}$, scores are aggregated as $S_j = \sum_i s_{ij}$; columns are ranked by $S_j$, and the top-$k$% (default 50%) are frozen. Frozen columns receive a mask value of 0; the remainder receive 1.
    - Design Motivation: Column-wise freezing preserves complete feature transformation pathways—freezing an entire column ensures that the corresponding output dimension remains entirely unchanged, analogous to preserving load-bearing columns during building renovation. Element-wise freezing disrupts feature transformations and leads to catastrophic forgetting (confirmed experimentally).

3. **Masked Continual Pre-training**:

    - Function: Learn the target language while protecting source knowledge.
    - Mechanism: Standard causal language modeling objective, with a static mask applied during backpropagation: $\theta_{ij} \leftarrow \theta_{ij} - \eta \cdot b_{ij} \cdot \nabla_{\theta_{ij}} L$, where $b_{ij} \in \{0, 1\}$ is the mask value. Embedding layers and the LM head are not subject to masking and are updated in full.
    - Design Motivation: The static mask is simple to implement and orthogonally composable with other mitigation methods (regularization, replay). It requires no dynamic mask recomputation during training, reducing computational overhead.

### Loss & Training
Standard causal language modeling loss is used for training on 200M tokens of target-language data. Chat templates are removed prior to training to accommodate unlabeled data.

## Key Experimental Results

### Main Results (7B models, averaged over source-language metrics)

| Method | IFEval | AlpacaEval2 | MT-Bench | GSM8K | Safety | Source Degradation (%) |
|--------|--------|-------------|----------|-------|--------|------------------------|
| Source | 0.675 | 32.6 | 3.98 | 0.796 | 0.851 | 0.0 |
| FFT | 0.456 | 10.4 | 3.48 | 0.608 | 0.797 | -20.3 |
| HFT | 0.621 | 17.6 | 3.83 | 0.677 | 0.826 | -8.0 |
| **SSU** | **0.669** | **27.0** | **3.96** | **0.752** | **0.850** | **-3.4** |

### Ablation Study

| Configuration | IFEval | Notes |
|---------------|--------|-------|
| SSU-Wanda | 0.669 | Full method (source data + Wanda scoring + column-wise freezing) |
| SSU-Rand | 0.608 | Random column freezing, no source-data guidance |
| SSU-Mag | 0.570 | Weight magnitude only, no activation signal |
| Element-wise freezing | ~0.45 | Disrupts feature transformations; performance collapses |

### Key Findings
- SSU consistently outperforms all baselines across all core instruction-following and safety tasks, and is the only method that excels simultaneously at source-language preservation and target-language improvement.
- On target-language benchmarks, SSU surpasses FFT on all metrics for 7B models and on most metrics for 13B models.
- The performance gap between column-wise and element-wise freezing is substantial, validating the importance of structured protection.
- SSU avoids the code-mixing problem observed in methods such as HFT.

## Highlights & Insights
- The "source-focused" rather than "target-focused" paradigm for parameter selection is the core insight—protecting existing knowledge is more important than selecting what to update.
- The analogy for column-wise freezing is apt: "preserving load-bearing columns during building renovation"—protecting complete feature pathways rather than isolated parameter points.
- Importance scoring requires only 500 source samples, incurring minimal overhead and offering strong practical utility.

## Limitations & Future Work
- Validation is currently limited to the OLMo 2 model family; broader architectural coverage remains to be explored.
- The fixed 50% freezing ratio may not be optimal for all languages; adaptive freezing ratios warrant investigation.
- Combinations with parameter-efficient methods such as LoRA have not been explored.

## Related Work & Insights
- **vs. HFT**: HFT randomly freezes sub-layer components without principled protection; SSU uses source-data-driven scoring to precisely identify critical parameters.
- **vs. GMT**: GMT dynamically selects parameters based on target-data gradients, which are unreliable for unlabeled text; SSU's source-focused strategy is more robust.
- **vs. AdaLoRA**: AdaLoRA preserves source performance well but yields limited target-language gains; SSU achieves a better balance between the two.

## Rating
- Novelty: ⭐⭐⭐⭐ The source-focused column-wise freezing strategy is novel and theoretically grounded.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Five languages, two model scales, and multi-dimensional evaluation.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear logic and rigorous motivation derivation.
- Value: ⭐⭐⭐⭐⭐ Directly addresses the core challenge of low-resource language adaptation.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] Exploring the Capability Boundaries of LLMs in Mastering of Chinese Chouxiang Language](exploring_the_capability_boundaries_of_llms_in_mastering_of_chinese_chouxiang_la.md)
- [\[ICLR 2026\] Prompt and Parameter Co-Optimization for Large Language Model Task Adaptation](../../ICLR2026/llm_evaluation/prompt_and_parameter_co-optimization_for_large_language_model_task_adaptation.md)
- [\[ACL 2026\] Self-Awareness before Action: Mitigating Logical Inertia via Proactive Cognitive Awareness](self-awareness_before_action_mitigating_logical_inertia_via_proactive_cognitive_.md)
- [\[ACL 2026\] Enhancing Linguistic Competence of Language Models through Pre-training with Language Learning Tasks](enhancing_linguistic_competence_of_language_models_through_pre-training_with_lan.md)
- [\[ACL 2026\] ResearchBench: Benchmarking LLMs in Scientific Discovery via Inspiration-Based Task Decomposition](researchbench_benchmarking_llms_in_scientific_discovery_via_inspiration-based_ta.md)

<!-- RELATED:END -->
