---
title: >-
  [Paper Note] CheXalign: Preference Fine-tuning in Chest X-ray Interpretation Models without Human Feedback
description: >-
  [ACL 2025][Medical LLM][Preference Alignment] CheXalign proposes an automated preference data generation pipeline without radiologist feedback. It leverages reference reports from public datasets and reference-based evaluation metrics (such as GREEN and BERTScore) to construct preference pairs, and performs preference fine-tuning on chest X-ray report generation models using direct alignment algorithms like DPO, achieving SOTA CheXbert scores on MIMIC-CXR.
tags:
  - "ACL 2025"
  - "Medical LLM"
  - "Preference Alignment"
  - "Radiology Report Generation"
  - "DPO"
  - "Chest X-ray"
  - "Length Exploitation"
date: 2026-05-08
content_hash: 03a649aad36efe19
---

# CheXalign: Preference Fine-tuning in Chest X-ray Interpretation Models without Human Feedback

**Conference**: ACL 2025  
**arXiv**: [2410.07025](https://arxiv.org/abs/2410.07025)  
**Code**: [https://github.com/StanfordMIMI/CheXalign](https://github.com/StanfordMIMI/CheXalign)  
**Area**: Medical NLP  
**Keywords**: Preference Alignment, Radiology Report Generation, DPO, Chest X-ray, Length Exploitation  

## TL;DR
CheXalign proposes an automated preference data generation pipeline without radiologist feedback. It leverages reference reports from public datasets and reference-based evaluation metrics (such as GREEN and BERTScore) to construct preference pairs, and performs preference fine-tuning on chest X-ray report generation models using direct alignment algorithms like DPO, achieving SOTA CheXbert scores on MIMIC-CXR.

## Background & Motivation

**Background**: Chest X-ray (CXR) is one of the most common imaging examinations in clinical practice, with approximately 1.4 billion scans globally each year. Vision-Language Models (VLMs) have shown potential in auxiliary radiologist workflows for Radiology Report Generation (RRG), with the mainstream approach being supervised fine-tuning (SFT) of VLMs.

**Limitations of Prior Work**: SFT has a fundamental shortcoming—during training, the log probabilities of "bad responses" inadvertently increase along with "good responses," which causes models to still generate factually inconsistent hallucinations. In high-risk fields like radiology, hallucinations can directly impact diagnostic and treatment decisions. General-domain LLMs have widely adopted preference fine-tuning like RLHF/DPO to resolve this issue, but it remains largely unexplored in medical VLMs.

**Key Challenge**: Preference fine-tuning requires large amounts of high-quality preference data, and acquiring feedback from radiologists in the radiology domain is extremely costly and non-scalable.

**Goal**: (a) How to generate high-quality preference data at scale without human feedback; (b) how to systematically apply DAA algorithms to medical VLMs; (c) how to address the length exploitation problem in reward over-optimization.

**Key Insight**: Public medical datasets (like MIMIC-CXR) already contain reference reports written by radiologists. The authors propose leveraging these reference reports in conjunction with reference-based evaluation metrics (such as GREEN and BERTScore) as automated Judges to replace human feedback.

**Core Idea**: Utilizing reference reports + automatic evaluation metrics to construct preference pairs bypasses the bottleneck of radiologist feedback, enabling low-cost and scalable alignment for medical VLMs.

## Method

### Overall Architecture

The input consists of CXR images + a dataset of reference reports written by radiologists (e.g., MIMIC-CXR), and the output is a preference-fine-tuned VLM. The pipeline consists of three steps: (1) repeatedly sample $N=4$ candidate reports from the baseline SFT model, (2) score each candidate report using an automated Judge (reference-based metrics), and (3) pair the highest and lowest scoring reports to form chosen/rejected preference pairs, which are then fed into DAA algorithms for preference fine-tuning.

### Key Designs

1. **Automated Preference Data Generation**:

    - Function: Automatically constructs preference pairs without extra human annotation.
    - Mechanism: For each training sample, 4 candidate reports are sampled from the SFT model and scored using GREEN or BERTScore against the reference report. The highest-scoring report is selected as *chosen* and the lowest as *rejected*. If all 4 scores are identical, the sample is discarded (GREEN discards about 1.6%, while BERTScore discards only 0.04%).
    - Design Motivation: Utilizing existing reference reports in datasets as factual baselines ensures the preference data has clinically relevant factual grounding without requiring a multimodal Judge.

2. **Length-Controlled GREEN Score (LC-GREEN)**:

    - Function: Mitigates the length exploitation issue in reward over-optimization.
    - Mechanism: $$\text{LC-GREEN} := \text{GREEN} / \max(\text{rel\_verbosity}, 1)$$, where rel_verbosity is the relative redundancy (word count ratio) of the candidate report relative to the reference report.
    - Design Motivation: Experiments revealed that when using GREEN as the Judge, DPO training inflated the average report length from 55.8 words to 140.2 words (a 2.51x redundancy), mostly consisting of semantic or syntactic repetition. LC-GREEN curbs this phenomenon by penalizing overly long reports.

3. **Systematic DAA Algorithm Comparison**:

    - Function: Compares the performance of 5 direct alignment algorithms on the RRG task.
    - Covered Algorithms: DPO (baseline), LC-DPO (explicit length regularization with an additional hyperparameter $\alpha$), IPO (relaxing the Bradley-Terry model assumption), KTO (no preference pairs needed, only binary feedback, doubling the data volume), and ORPO (attaches a negative gradient penalty term to the SFT loss).
    - Design Motivation: Different DAAs have distinct trade-offs in length control, data requirements, and computational efficiency. A systematic comparison helps identify the optimal alignment scheme for medical VLMs.

### Loss & Training

The 5 DAAs employ different objective functions. Taking DPO as an example: $$\mathcal{L}_{\text{DPO}} = -\log\sigma(\beta\log\frac{\pi_\theta(y_c|x)}{\pi_{\text{ref}}(y_c|x)} - \beta\log\frac{\pi_\theta(y_r|x)}{\pi_{\text{ref}}(y_r|x)})$$. LC-DPO adds a length penalty term $\alpha(|y_c| - |y_r|)$ on top of this. KTO does not require preference pairs and uses a loss based on Kahneman-Tversky prospect theory. ORPO directly appends an odds-ratio contrastive term to the NLL loss, requiring no reference model and cutting training time the most (taking only 0.7x of DPO's execution time).

## Key Experimental Results

### Main Results

Baseline models: CheXagent (8B) and CheXagent-2 (3B). Training set: MIMIC-CXR (80k/148k samples). Test set: MIMIC-CXR + CheXpert Plus.

| Model | Judge | GREEN ↑ | LC-GREEN ↑ | BERTScore ↑ |
|------|-------|---------|------------|-------------|
| CheXagent (SFT) | - | 0.249 | 0.218 | 0.856 |
| +KTO | GREEN | **0.328 (+31.9%)** | **0.293 (+34.1%)** | 0.867 |
| +KTO | BERTScore | 0.304 (+21.9%) | 0.279 (+28.2%) | **0.872 (+1.88%)** |
| CheXagent-2 (SFT) | - | 0.326 | 0.297 | 0.888 |
| +DPO | GREEN | **0.387 (+18.9%)** | **0.339 (+14.1%)** | 0.891 |

CheXbert F1 Scores (MIMIC-CXR test set, Avg. F1):

| Model | Judge | Avg. F1 ↑ |
|------|-------|-----------|
| CheXagent (SFT) | - | 47.9 |
| +KTO | BERTScore | **54.6 (+14.0%)** |
| CheXagent-2 (SFT) | - | 55.1 |
| +DPO | GREEN | **56.4 (+2.4%)** |
| MAIRA-2 (baseline comparison) | - | 52.3 |

### Ablation Study

Length exploitation analysis (MIMIC-CXR validation set, GREEN as Judge):

| Configuration | Average Report Length | Relative Redundancy | Description |
|------|-------------|-----------|------|
| CheXagent (SFT) | 55.8 | 1.00 | Baseline |
| +DPO (GREEN Selection) | 140.2 | 2.51 | Severe length inflation |
| +DPO (LC-GREEN Selection) | 68.7 | 1.23 | LC-GREEN effectively controls length |
| +KTO (LC-GREEN Selection) | 55.9 | 1.00 | Almost zero inflation |
| Reference Report | 58.4 | 1.05 | - |

### Key Findings
- **BERTScore as Judge unexpectedly outperforms GREEN on CheXbert scores**: This might be because GREEN is more susceptible to length exploitation, leading to reward over-optimization.
- **KTO achieves the best overall performance** (on CheXagent): It does not require preference pairs, doubles the available training data, and exhibits the least length inflation.
- **The alignment tax is near zero**: The aligned models perform on par with the SFT baseline on 6 auxiliary visual perception/reasoning tasks.
- **Cheap general NLG metrics can effectively improve clinical performance**: Even using a general metric like BERTScore as the Judge yields significant improvements in CheXbert scores.

## Highlights & Insights
- **Cleverly leveraging existing data to bypass human annotation bottlenecks**: Reference reports in public medical datasets are essentially high-quality "human feedback." Equipping them with automated evaluation metrics is sufficient to construct preference pairs. This strategy can be transferred to any generation task that has reference standards (e.g., legal documents, academic writing).
- **LC-GREEN simply and effectively addresses length exploitation**: A simple division-based normalization reduces the report length from 140 words to 69 words, while improving performance. This length-control heuristic can be widely integrated into metric designs across all DAAs.
- **Unexpected findings with general vs. domain-specific metrics**: Using the lower-cost BERTScore as the Judge outperformed the expensive, domain-specific GREEN on CheXbert F1, suggesting that over-optimizing a domain-specific metric may lead to adverse effects.

## Limitations & Future Work
- Only tested on two model families (CheXagent and CheXagent-2); generalizability remains to be verified.
- Lacks large-scale radiologist reader studies; clinical evaluation still heavily relies on automatic metrics.
- Only offline DAAs were utilized, leaving online RL algorithms (such as currently popular methods like RLVR / GRPO) unexplored.
- Did not investigate potential demographic biases (e.g., race, gender) embedded within the preference data.
- Hyperparameter search is not fully exhaustive, and the relative rankings of different methods might shift under finer-grained tuning.

## Related Work & Insights
- **vs. Sun et al. (2024)**: They use reference-free multimodal Judges for preference alignment, whereas this work uses reference-based unimodal Judges. The advantage of this work lies in more reliable factual grounding without requiring high-quality multimodal evaluators.
- **vs. MAIRA-2**: MAIRA-2 achieves high CheXbert scores by using additional context (e.g., prior reports), whereas this work achieves comparable or superior results purely through alignment without using any extra information.
- **vs. DeepSeek-R1**: The authors mention in their limitations that RLVR is a promising direction, where the design of the verification function for the RRG task remains critical.

## Rating
- Novelty: ⭐⭐⭐⭐ The core idea is simple (using reference reports as Judge), but it represents the first systematic application of DAAs to medical VLMs while tackling and analyzing the length exploitation issue.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely comprehensive, covering 5 DAAs, 2 Judges, 2 models, multiple datasets, several metrics (CheXbert/GREEN/BERTScore), alignment tax analysis, and qualitative evaluations.
- Writing Quality: ⭐⭐⭐⭐⭐ Well-structured, detailed experimental setups, and excellent figure/table designs.
- Value: ⭐⭐⭐⭐ Provides a practical and reproducible pipeline for medical VLM alignment, with the analysis of LC-GREEN and length exploitation offering valuable references for the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] The Impact of Auxiliary Patient Data on Automated Chest X-Ray Report Generation and How to Incorporate It](auxiliary_patient_data_xray.md)
- [\[CVPR 2026\] Towards Efficient Medical Reasoning with Minimal Fine-Tuning Data](../../CVPR2026/medical_nlp/towards_efficient_medical_reasoning_with_minimal_fine-tuning_data.md)
- [\[ACL 2025\] Automated Structured Radiology Report Generation](automated_structured_radiology_report_generation.md)
- [\[ACL 2025\] Radar: Enhancing Radiology Report Generation with Supplementary Knowledge Injection](radar_radiology_report_gen.md)
- [\[ACL 2025\] Online Iterative Self-Alignment for Radiology Report Generation](oisa_radiology_report_gen.md)

</div>

<!-- RELATED:END -->
