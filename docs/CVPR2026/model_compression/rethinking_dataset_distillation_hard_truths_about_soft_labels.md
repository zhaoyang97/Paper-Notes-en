---
title: >-
  [Paper Note] Rethinking Dataset Distillation: Hard Truths about Soft Labels
description: >-
  [CVPR 2026 (Oral)][Model Compression][Dataset Distillation] This is an analysis paper with a "myth-busting" nature: the authors systematically prove that **the apparent lead of large-scale dataset distillation (DD) methods is primarily sustained by the use of soft labels during downstream training**. Once scalability analysis is performed across different label regimes, the advantage of high-quality subsets over random subsets nearly disappears. Based on this…
tags:
  - "CVPR 2026 (Oral)"
  - "Model Compression"
  - "Dataset Distillation"
  - "Soft Labels"
  - "coreset"
  - "compute budget"
  - "difficulty pruning"
date: 2026-05-08
content_hash: ba5e38b6ae57e974
---

# Rethinking Dataset Distillation: Hard Truths about Soft Labels

**Conference**: CVPR 2026 (Oral)  
**arXiv**: [2604.18811](https://arxiv.org/abs/2604.18811)  
**Code**: None (at the time of writing, not provided in the abstract)  
**Area**: Model Compression / Dataset Distillation  
**Keywords**: Dataset Distillation, Soft Labels, coreset, compute budget, difficulty pruning

> ⚠️ The full HTML version of this arXiv paper is not yet released (`/html/` returns 404); this note is a **best-effort reconstruction** based on the abstract and public information. Please refer to the original paper for specific numbers/implementation details marked with ⚠️.

## TL;DR
This is an analysis paper with a "myth-busting" nature: the authors systematically prove that **the apparent lead of large-scale dataset distillation (DD) methods is primarily sustained by the use of soft labels during downstream training**. Once scalability analysis is performed across different label regimes, the advantage of high-quality subsets over random subsets nearly disappears. Based on this, they propose a compute-aware difficulty pruning metric **CAD-Prune** and a compute-aligned distillation method **CA2D**, which outperform existing DD methods across multiple IPC settings on ImageNet-1K.

## Background & Motivation
**Background**: Dataset Distillation (DD) aims to "condense" a large dataset into a small synthetic set with only a few images per class (IPC), such that models trained on this small set approach the performance of those trained on the full data. Recent large-scale methods (such as SRe²L and its successors) claim great success on ImageNet-1K.

**Limitations of Prior Work**: However, recent evidence suggests that a **simple "random image baseline" can match the performance of SOTA DD methods like SRe²L**. This directly contradicts the experience in coreset (core subset selection) literature—where carefully selected high-quality subsets **consistently** outperform random ones. Why does "selecting good samples" matter for coresets but appear useless for DD?

**Key Challenge**: The authors identify the root cause as the **label regime used during evaluation**. Downstream evaluation in DD commonly uses **soft labels** (probability distributions provided by a teacher network, often generated online with knowledge distillation, KD), whereas classic coreset evaluation uses **hard labels** (one-hot vectors). Once downstream training has a constant supply of soft label supervision, the dimension of subset "quality" is flattened regarding final accuracy—thus, the "superiority" of DD methods might not come from the synthesized images at all, but from the soft label training paradigm.

**Goal**: To quantify the question of "how much does data quality really matter" within a **scalability coordinate system of different label regimes**; and then design a method that truly works under fair conditions (hard labels / fixed compute).

**Key Insight**: Categorize the "abundance" of label supervision into three gradients—from **abundant soft labels (SL+KD)** → **fixed soft labels (SL)** → **hard labels (HL)**—and measure the gain of high-quality coresets relative to random baselines to see where the gain disappears.

**Core Idea**: Summarized in one sentence: **"Under soft label regimes, subset quality hardly affects accuracy (performance saturation), and DD's lead is a credit to soft labels; only under hard labels + aligned compute budgets does selecting samples of the correct difficulty truly matter."** This finding is engineered into CAD-Prune and CA2D.

## Method

### Overall Architecture
This paper is less about "proposing a method" and more about "performing a controlled analysis and then translating the conclusions into a method." It consists of two stages:

1.  **Scalability Analysis (Diagnosis)**: Fix the downstream compute budget and vary subset size and quality (random vs. high-quality coreset vs. existing DD synthetic sets) across three label regimes (SL+KD / SL / HL) to observe accuracy curves and answer "under what conditions quality matters."
2.  **Method Design (Prescription)**: The analysis finds that "optimally difficult samples under hard labels + matched compute budget" are the truly effective signals. Thus, the **CAD-Prune** (Compute-Aware Difficulty pruning) metric is proposed to select such samples, which is then used to construct **CA2D**, a compute-aligned DD method.

Since the core contribution lies in the analytical conclusions, the method itself can be explained simply: "select samples with optimal difficulty according to the compute budget." Therefore, no pipeline diagram is provided. The "Key Designs" follow the sequence of **Analysis Findings 1→2→3 + Implementation**.

### Key Designs

**1. Scalability Analysis of Three Label Regimes: Placing "Quality Importance" in a Controlled Coordinate System**

To address the contradiction of why random baselines match SOTA in DD but not in coresets, the authors designed a controlled experimental framework to measure the gain of "high-quality subsets vs. random subsets" across three levels of label supervision intensity:
-   **SL+KD (abundant soft labels)**: Training is supervised online by a teacher providing soft labels for (augmented) samples, equivalent to an **infinite supply** of soft labels + KD.
-   **SL (fixed soft labels)**: Uses a **fixed** set of pre-computed soft labels; the quantity is limited and does not refresh with augmentation.
-   **HL (hard labels)**: Classic one-hot hard labels without teacher involvement.

The key control variable is a **fixed downstream compute budget**, under which subset scale and quality are swept. This coordinate system allows the decoupling of "label signal abundance" and "data quality." ⚠️ Accurate definitions and teacher configurations follow the original text.

**2. Key Finding: "Performance Saturation" Under Soft Labels Flattening Subset Quality**

The analysis yields the "hard truth": in both **SL and SL+KD regimes, high-quality coresets fail to convincingly outperform random baselines**. Particularly in the **SL+KD** regime, given a compute budget, accuracy **approaches a near-optimal level relative to the full dataset and is almost independent of subset size or quality**—a phenomenon called **performance saturation**.

This finding is impactful because it directly questions the widely adopted practice of "evaluating DD using soft labels." Since subset quality has a negligible impact on final accuracy under soft labels, the "improvements" reported by DD methods in such settings do not prove that the synthetic data itself is better—the observed lead is likely due to the soft label + KD training paradigm rather than the distillation algorithm. In contrast, quality becomes significant again in the **hard label (HL)** regime, explaining the discrepancies between coreset and DD literature.

**3. Systematic Re-evaluation Under Hard Labels: Only RDED Truly Beats Random, but Biases Toward "Easy Sample Patches"**

Moving the evaluation back to the fair HL setting, the authors **systematically evaluated 5 large-scale and 4 small-scale DD methods**. The conclusion is stark: on ImageNet-1K, **only RDED consistently outperforms the random baseline**; other methods are unreliable under hard labels. Even RDED **may fall behind strong coreset methods** because it **over-relies on image patches of easy samples**, leading to an incorrect sample difficulty structure that suffers when soft labels are not present to compensate. This identifies the specific failure mechanism: selecting the wrong difficulty. ⚠️ Refer to the original paper for the list of methods and specific values.

**4. CAD-Prune + CA2D: Selecting "Optimal Difficulty" Based on Compute Budget**

Since the problem lies in the "mismatch between difficulty selection and compute," the authors propose **CAD-Prune (Compute-Aware Difficulty pruning)**. The core idea is that the **"optimal difficulty" of a sample depends on the given compute budget**—more difficult samples are more valuable when compute is high, while overly difficult samples become a burden when compute is tight. This compute-aware metric efficiently screens samples whose difficulty is "just right" for the current budget, avoiding the bias of stacking only easy samples as seen in RDED.

Based on this metric, **CA2D (Compute-Aligned Dataset Distillation)** is constructed by selecting samples aligned with the budget via CAD-Prune. Results show that **CA2D exceeds current DD methods across multiple IPC settings on ImageNet-1K**. ⚠️ Precision scoring formulas for CAD-Prune and the specific IPC values for CA2D follow the original text.

### Loss & Training
The core of the paper is the analysis plus a pruning metric, rather than a new loss function. Downstream evaluation is conducted across three label regimes: HL uses standard cross-entropy with one-hot labels; SL uses fixed soft labels; SL+KD uses teacher-generated online soft labels + KD. CA2D training follows its aligned compute budget constraints. ⚠️ Specific optimizers and hyperparameters follow the original text.

## Key Experimental Results

> ⚠️ The following table is a **qualitative reconstruction** based on the abstract to convey the direction of the conclusions; please refer to the original paper for exact values.

### Main Results

| Setting / Regime | Comparison | Conclusion (Qualitative) | Meaning |
| :--- | :--- | :--- | :--- |
| SL+KD (Abundant Soft Labels) | High-quality coreset vs. Random | Almost a tie, approaching full data optimal → **Performance Saturation** | Subset quality impact is negligible |
| SL (Fixed Soft Labels) | High-quality coreset vs. Random | High-quality subsets **fail to convincingly** win | Soft labels still flatten quality differences |
| HL (Hard Labels) | coreset vs. Random vs. DD | Quality matters again; only **RDED** consistently beats random among 9 DD methods | DD is generally ineffective in fair regimes |
| HL, ImageNet-1K | RDED vs. Strong coreset | RDED may still fall behind (bias toward easy patches) | Difficulty selection is key |
| HL, ImageNet-1K, Multi-IPC | **CA2D** vs. Existing DD | CA2D outperforms existing DD methods | Compute-aligned difficulty pruning is effective |

### Ablation Study

| Configuration | Key Variable | Description (Qualitative) |
| :--- | :--- | :--- |
| Label Regime SL+KD→SL→HL | Soft label abundance | As labels move toward hard, subset quality gain becomes more obvious (saturation vanishes) |
| Subset Size Scan (Fixed Compute) | IPC / Subset Size | Accuracy is insensitive to size/quality under SL+KD (evidence of saturation) |
| RDED Sample Difficulty | Ratio of easy sample patches | Over-reliance on easy samples → falls behind strong coresets |
| CAD-Prune Difficulty vs. Compute | Difficulty selection strategy | Samples with difficulty matched to the budget provide gains |

### Key Findings
- **Most Critical Finding**: Soft labels (especially SL+KD) render the "subset quality" dimension nearly irrelevant to final accuracy (performance saturation). Therefore, **evaluating DD with soft labels overestimates algorithm contribution**—the lead likely comes from the training paradigm, not the synthetic data.
- **Hard Labels are the Touchstone**: Differences in data quality/difficulty only manifest under the HL setting; among 9 DD methods, only RDED consistently beats the random baseline.
- **Difficulty Selection is the Key**: RDED's bias toward easy sample patches is why it lags behind strong coresets; picking "optimally difficult" samples according to the compute budget (CAD-Prune) corrects this, leading CA2D to take the lead in multiple IPC settings.

## Highlights & Insights
- **Methodological contribution outweighs score chasing**: Instead of rushing to propose a new distillation algorithm, the paper quantifies the neglected confounding variable—the "evaluation regime"—proving that many "SOTA" results were likely dividends of soft labels. This "disprove then build" paradigm is a wake-up call for the DD community.
- **"Performance Saturation" is a transferable insight**: Given a compute budget + abundant soft labels, accuracy is insensitive to subset quality. This suggests that any evaluation of data-efficient learning using KD/soft labels should be supplemented with a hard-label control; otherwise, the conclusions may not hold.
- **Binding "Difficulty" to "Compute Budget"**: The core insight of CAD-Prune—that optimal sample difficulty varies with compute—can be transferred to general coreset/data pruning and curriculum learning: don't stack hard samples when compute is tight, and don't feed only easy samples when compute is plentiful.

## Limitations & Future Work
- ⚠️ **Conclusions focus on ImageNet-1K**: Systematic re-evaluation and CA2D's lead are primarily reported for ImageNet-1K; the generalizability across datasets/architectures requires more verification.
-   **Method itself is relatively simple**: CA2D is "difficulty pruning aligned with compute," which is lighter than complex generative DD methods but also means its absolute accuracy ceiling might be limited by "selecting samples" rather than "creating samples."
-   **Soft labels are not useless**: The paper emphasizes that "soft labels mask quality differences and are unsuitable for fair evaluation," not that soft-label training has no value. How to retain soft label benefits while distinguishing data quality remains an open question.
-   **CAD-Prune's difficulty metric depends on proxy models**: Difficulty estimation itself requires compute and a scoring model; how its bias affects sample selection remains to be investigated.

## Related Work & Insights
- **vs. SRe²L / Large-scale DD methods**: They report significant leads under SL+KD; Ours points out that these leads are mostly untenable under hard labels, where random baselines can match them, questioning their evaluation systems fundamentally.
- **vs. RDED**: RDED is the only DD method consistently beating random under hard labels but is biased toward "easy sample patches." CA2D uses compute-aware difficulty pruning to correct the difficulty structure and thus succeeds.
- **vs. Classic Coreset Selection**: Coreset quality advantages are obvious under hard labels. Ours uses this as a reference point for whether quality matters and finds that strong coresets can even outperform some DD methods under HL.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Not just another DD algorithm, but a structural insight revealing that "soft label evaluation systematically overestimates DD," coupled with diagnostic tools.
- Experimental Thoroughness: ⭐⭐⭐⭐ Systematic re-evaluation of 3 label regimes × 9 DD methods is substantial; ⚠️ specific scale/cross-dataset breadth to be confirmed by original text.
- Writing Quality: ⭐⭐⭐⭐ Clear logical chain in the abstract (contradiction → analysis → discovery → method), with a powerful "hard truths" narrative.
- Value: ⭐⭐⭐⭐⭐ Directly impacts evaluation standards for the DD community and provides immediately usable tools in CAD-Prune / CA2D; well-deserved CVPR Oral.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Hard Labels In! Rethinking the Role of Hard Labels in Mitigating Local Semantic Drift](../../ICML2026/model_compression/hard_labels_in_rethinking_the_role_of_hard_labels_in_mitigating_local_semantic_d.md)
- [\[CVPR 2026\] Beyond Soft Label: Dataset Distillation via Orthogonal Gradient Matching](beyond_soft_label_dataset_distillation_via_orthogonal_gradient_matching.md)
- [\[ICML 2026\] The Bridge-Garden Dilemma in LLM Distillation: Why Mixing Hard and Soft Labels Works](../../ICML2026/model_compression/the_bridge-garden_dilemma_in_llm_distillation_why_mixing_hard_and_soft_labels_wo.md)
- [\[CVPR 2026\] Dataset Distillation by Influence Matching](dataset_distillation_by_influence_matching.md)
- [\[ICLR 2026\] ACPBench Hard: Unrestrained Reasoning about Action, Change, and Planning](../../ICLR2026/model_compression/acpbench_hard_unrestrained_reasoning_about_action_change_and_planning.md)

</div>

<!-- RELATED:END -->
