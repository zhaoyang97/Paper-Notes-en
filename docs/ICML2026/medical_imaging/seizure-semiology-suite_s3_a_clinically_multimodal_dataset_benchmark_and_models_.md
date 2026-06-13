---
title: >-
  [Paper Note] Seizure-Semiology-Suite (S³): A Clinically Multimodal Dataset, Benchmark, and Models for Seizure Semiology Understanding
description: >-
  [ICML 2026][Medical Imaging][Seizure Semiology] This work constructs the first large-scale expert-annotated seizure video dataset, S³ (438 videos, 35,000+ dense labels…
tags:
  - "ICML 2026"
  - "Medical Imaging"
  - "Seizure Semiology"
  - "Clinical Video Understanding"
  - "Multimodal Large Language Models"
  - "Report Quality Evaluation"
  - "Neuro-symbolic Classification"
date: 2026-05-08
content_hash: e0e0a08ffa517628
---

# Seizure-Semiology-Suite (S³): A Clinically Multimodal Dataset, Benchmark, and Models for Seizure Semiology Understanding

**Conference**: ICML 2026 Spotlight  
**arXiv**: [2605.21852](https://arxiv.org/abs/2605.21852)  
**Code**: Yes (GitHub: SeizureSemiologySuite)  
**Area**: Medical Imaging / Multimodal VLM / Video Understanding  
**Keywords**: Seizure Semiology, Clinical Video Understanding, Multimodal Large Language Models, Report Quality Evaluation, Neuro-symbolic Classification

## TL;DR
This work constructs the first large-scale expert-annotated seizure video dataset, S³ (438 videos, 35,000+ dense labels, 20 ILAE semiology features). It introduces a seven-level hierarchical task benchmark and the clinically-aligned Seizure-RQI report quality metric. The study systematically exposes the failure modes of 11 open-source MLLMs in temporal localization, spatial lateralization, and clinical faithfulness, while improving the Seizure vs. Non-Epileptic Seizure (ES vs. NES) classification F1 to 0.96 through domain-specific fine-tuning and a two-stage neuro-symbolic framework.

## Background & Motivation

**Background**: Seizure semiology is the core evidence for clinically diagnosing seizure types, localizing the seizure onset zone (SOZ), and assessing SUDEP risk. Currently, it primarily relies on trained epileptologists manually reviewing long-term Video-EEG Monitoring (EMU) footage frame-by-frame, which is highly subjective, labor-intensive, and nearly impossible to scale in resource-limited areas.

**Limitations of Prior Work**: Automated methods have long been trapped between two extremes. One category consists of narrow-task discriminative pipelines (3D CNNs for tonic-clonic detection, accelerometers, optical flow segmentation, CNN classifiers) that only output coarse "yes/no" labels, lacking descriptive interpretability. The other category involves directly applying MLLMs from general video QA datasets (ActivityNet-QA, MSRVTT-QA, MotionBench) to seizure videos; however, these datasets focus on "purposeful, daily activities" and do not cover "involuntary, pathological movements." Furthermore, representative datasets for medical video understanding (MedVidQA, SV-RCNet) concentrate on procedural scenes like surgery, assuming pre-defined temporal structures that differ fundamentally from seizure events.

**Key Challenge**: For MLLMs to be clinically viable, they must correctly handle "spatial lateralization (patient's left vs. right)," "symptomatic co-occurrence," "temporal evolution (ictal march)," and "narrative report generation" simultaneously. However, there is currently a lack of training/evaluation data with dense expert labels, as well as evaluation metrics capable of distinguishing between "high BLEU but clinically incorrect" and "low BLEU but clinically correct" outputs—traditional N-gram metrics and BERTScore show almost no correlation with clinical factuality.

**Goal**: (i) Create a seizure video dataset with dense ILAE-standardized semiology feature annotations specifically for MLLMs; (ii) Design a hierarchical task system covering the full stack of "Perception → Reasoning → Reporting → Diagnosis"; (iii) Propose a report quality metric aligned with expert judgment; (iv) Provide a domain-adapted training solution that closes the loop on this dataset.

**Key Insight**: The cognitive process of clinical experts interpreting seizure videos is decomposed into seven layers: single-frame/short-window feature recognition → rationale interpretation → lateral/anatomical spatial analysis → temporal boundary localization → ordered symptom sequences → narrative reporting → comprehensive diagnosis. By scoring each layer independently, the "system performance" of MLLMs is decomposed to identify which specific link is the weakest, providing precise diagnostics for model iteration.

**Core Idea**: By using a "four-piece suite" of domain expert annotation, hierarchical clinical tasks, clinically-aligned metrics, and neuro-symbolic decoupling, general MLLMs are transformed into clinically trustworthy seizure semiology interpreters.

## Method

### Overall Architecture
S³ consists of three components: **(1) Seizure-Semiology-Dataset** — 438 continuous videos from 116 adult patients at the UCLA EMU (2019–2023), with expert annotations for 20 ILAE-defined semiology features (e.g., automatisms, tonic, clonic). Each feature includes "presence + start/end timestamps + textual rationale," totaling 35,000+ labels. Data is split 4:1 at the patient level for training/testing, maintaining the ES (epileptic) to NES (non-epileptic) ratio. **(2) Seizure-Semiology-Bench** — A hierarchical evaluation system of 7 tasks, each with its own prompt template, sampling protocol (30s sliding window, event-centered cropping, sparse sampling), and evaluation metrics (F1, MAE, Edit Distance, LCS ratio, Seizure-RQI, etc.). **(3) Improvement Strategies** — SFT and GRPO fine-tuning on Qwen2.5-Omni-7B, alongside a two-stage neuro-symbolic classifier decoupling perception from diagnostic reasoning.

Dataset quality is ensured by a five-stage annotation pipeline: expert training → independent annotation of 75 videos to verify consistency (Kappa = 0.8395) → independent annotation of the remaining 287 videos (difficult cases resolved by a 3-expert consensus) → adaptive sampling to verify feature distribution and expert convergence (ES Pearson 0.893, NES Pearson 0.782) → LLM textual error correction + regularized timestamp validation.

### Key Designs

1. **Seven-Level Hierarchical Task Benchmark (Seizure-Semiology-Bench)**:
    - **Function**: Decomposes "understanding seizure videos" into a gradient of independently diagnosable capabilities—Task 1: Binary Feature Recognition (20 yes/no prompts); Task 2: Textual Rationale Generation (explaining why a feature exists); Task 3: Spatial Lateralization (multiple-choice + explicit anchoring to patient's left/right instead of camera view); Task 4: Temporal Boundary Localization (MM:SS timestamps + MAE evaluation); Task 5: Symptom Temporal Ordering (Edit Distance / Temporal-F1 / LCS ratio); Task 6: Narrative Report Generation; Task 7: Clinical Diagnosis (ES vs. NES, compared across video-only, report-augmented, and two-stage modes).
    - **Mechanism**: Task difficulty increases strictly, and failures in subsequent tasks can be traced back to preceding ones. For instance, a poor Task 5 ordering score can be decomposed into Task 1 recognition errors ("what") and Task 4 temporal errors ("when"). This traceability is unavailable in simple end-to-end "diagnostic accuracy" evaluations.
    - **Design Motivation**: General MLLMs might achieve acceptable average scores while systematically failing in a critical clinical dimension (e.g., lateralization). Without stratification, true failure modes remain hidden. Experiments showed that scaling Qwen2.5-VL from 7B to 72B only resulted in Task 1 F1 fluctuating between 0.42–0.45, proving that "increasing scale" is not the solution and bottlenecks must be identified through layers.

2. **Seizure-RQI Clinically-Aligned Report Evaluation Metric**:
    - **Function**: Replaces surface-level metrics like BLEU/ROUGE/BERTScore, which show nearly zero correlation with clinical judgment (Pearson $r \leq 0.10$), with a score truly aligned with expert assessment.
    - **Mechanism**: Score = Weighted Base Score × Multiple Penalty Terms. The base score consists of four components: Structural completeness $S$ (15%, including onset/propagation/postictal segments), Symptom coverage $C$ (35%, correct features / total ground truth features), Key localization features $L$ (25%, matching ratio of 4 lateralization features), and Temporal fidelity $T$ (25%, Temporal F1 of the chronological feature list). The final metric is $\mathrm{RQI} = (15S + 35C + 25L + 25T) \times P_{\text{hall}} \times P_{\text{off}} \times P_{\text{len}} \times P_{\text{haz}}$, where the four $P$ terms penalize hallucinated features, off-topic content (e.g., nursing interventions), excessive redundancy, and hazardous clinical statements.
    - **Design Motivation**: The value of a clinical report lies in whether it describes key localization cues, follows the correct temporal order, and avoids dangerous misinformation, rather than word-level similarity. Validation shows RQI achieves a Pearson correlation of 0.57 and a pairwise accuracy of 0.74 with expert judgment, significantly higher than the ≈0.54 of general metrics.

3. **Two-Stage Neuro-Symbolic Diagnostic Framework**:
    - **Function**: Addresses the issue where MLLMs suffer from "hallucinatory reasoning" in long-temporal contexts when performing end-to-end diagnosis.
    - **Mechanism**: The first stage treats the MLLM as a pure perceptron, executing Task 1 to output a 20-dimensional binary feature vector $v \in \{0,1\}^{20}$, compressing high-dimensional unstructured video into a structured, interpretable clinical feature representation. The second stage feeds $v$ into a shallow statistical classifier (e.g., Random Forest) for ES vs. NES discrimination. The MLLM is responsible only for "what is seen," while logical classification is outsourced to a reliable symbolic model.
    - **Design Motivation**: It was observed that while MLLMs can approach expert performance in visual recognition (Task 1) after domain fine-tuning, they remain unstable in "rule-based reasoning based on multiple features." Decoupling these tasks allows the Random Forest to output feature importance (e.g., giving high weights to tonicity, head deviation, rapid blinking, or nocturnal onset), providing much higher interpretability than end-to-end MLLMs. Combining this framework with seizure_omni_sft-7B pushed the ES vs. NES F1 from 0.70 (direct diagnosis) to 0.96.

### Loss & Training
Two types of domain-specific fine-tuning were applied to Qwen2.5-Omni-7B: **(i) SFT** using (video, prompt, answer) triples for supervised learning; **(ii) GRPO** using Group Relative Policy Optimization, with task-customized rewards—accuracy rewards for Task 1/3/7, composite BLEU+ROUGE rewards for Task 2/6, temporal proximity rewards for Task 4, and LCS ratio rewards for Task 5. A key lesson from GRPO was that using BLEU/ROUGE as a reward for Task 6 forced the model toward repetitive outputs because these metrics do not reflect clinical relevance—this further validates the necessity of Seizure-RQI.

## Key Experimental Results

### Main Results

| Task / Metric | Best Baseline MLLM | seizure_omni_sft | seizure_omni_grpo | Remarks |
|---|---|---|---|---|
| Task 1 F1 (Feature Recognition) | Qwen2.5-VL-72B ≈ 0.43 | **0.47** | 0.43 | SFT 7B outperforms 72B general model |
| Task 4 Avg MAE (sec) | Qwen2.5-VL-32B 8.19 On / 12.72 Off | 23.02 | **20.02** | GRPO improves 21.5% over baseline 25.50 |
| Task 5 LCS ratio | Qwen3-Omni-30B 0.43 | 0.18 | 0.18 | 50% Gain over Qwen2.5-Omni baseline 0.12 |
| Task 6 Seizure-RQI | Lingshu-32B **39.80** | 31.69 | 36.44 | Small gap between medical pre-train vs general |
| Task 7 ES vs NES F1 (video-only) | Lingshu-32B 0.84 | 0.71 | 0.77 | Limited end-to-end MLLM ceiling |
| Task 7 F1 (two-stage neuro-symbolic) | — | **0.96** | 0.94 | First to reach this on large-scale video only |

### Ablation Study

| Configuration | Avg Task 7 F1 | Key Findings |
|---|---|---|
| Direct MLLM (w/o rpt) | 0.70 | End-to-end diagnosis as baseline |
| Report-augmented (w/ rpt) | 0.79 | Ground truth report provides +0.09 Gain |
| Two-stage neuro-symbolic | **0.86** | Avg +0.16 Gain, exceeding GT report access |
| Seizure-RQI vs BLEU/ROUGE/BERTScore | Pearson 0.57 vs ≤0.10 | Significant clinical alignment |
| FPS 2 → 4 → 10 | Task 1 F1 +0.06 / +0.08 | Sampling rate is not the primary bottleneck |
| Task 4 Full-video sparse 60 frames | Avg MAE +4.91s | Localization is a fundamental MLLM flaw |

### Key Findings
- **Scale is not the solution**: Qwen2.5-VL labels for Task 1 showed almost no improvement from 7B to 72B, indicating a lack of inductive bias for pathological movement at the architectural level that cannot be solved by parameter scaling alone.
- **Domain Tuning vs. Catastrophic Forgetting**: SFT/GRPO brought an average 12% / 15% improvement across six tasks, but Task 3 (lateralization) collapsed to a single output ("left") with F1 = 0.00 due to small sample sizes (527 total, only 98 for head turning). Domain adaptation requires sufficient sub-task samples.
- **Multimodal fusion outperforms unimodal**: Qwen3-Omni-30B outperformed the vision-only Qwen2.5-VL-32B and audio-only Audio-Flamingo-3 on sound-related features, proving semiology requires auditory signals (vocalization, responsiveness).
- **Medical pre-training is a double-edged sword**: Lingshu-32B achieved the best video-only Task 7 F1 (0.84), but performance dropped to 0.60 when adding reports, suggesting its linguistic reasoning was weakened by narrow medical fine-tuning.
- **Spatial lateralization remains an open problem**: Even with explicit prompts for "patient's left vs. camera's left," all models averaged Task 3 F1 < 0.2. Prompt engineering cannot fix this; the root cause is a lack of spatial relationship data in pre-training corpora (e.g., LAION-2B).

## Highlights & Insights
- **Traceability in hierarchical tasks**: Decomposing Task 5 (sequence) errors into Task 1 (recognition) and Task 4 (localization) errors provides a "diagnostic benchmark" applicable to any long-temporal multi-step reasoning task.
- **Multiplicative penalty structure of Seizure-RQI**: Combining additive evidence with multiplicative safety gates (for hallucinations/hazards) is a paradigm that can be directly transferred to radiology, pathology, or surgical reports.
- **Neuro-symbolic decoupling for interpretability**: Treating the MLLM as a "feature engineer" rather than a "diagnostic physician" achieved F1=0.96 while providing feature importance rankings. This engineering paradigm is crucial for moving AI-assisted diagnosis toward clinical deployment by gaining physician trust.
- **GRPO reward failure case**: Using BLEU/ROUGE as rewards encouraged repetitive content, demonstrating that RL phases must use reward functions truly aligned with downstream objectives—a cautionary lesson for RLHF/RLVR in other domains.

## Limitations & Future Work
- Single-source data: Data only from UCLA, adult patients (18–64); lacks pediatric data and cross-institution validation. Small sample sizes for lateralization sub-tasks limited fine-tuning.
- Frame rate constraints: All tasks were evaluated at 2 FPS, leading to systematic information loss for high-frequency events like rapid blinking (10 FPS showed a +0.08 F1 Gain over 2 FPS).
- Fundamental temporal localization weakness: MLLMs cannot iteratively refine start/end times like clinicians; MAE remains at 8–12 seconds. Future work may involve agentic frameworks for tool-augmented refinement.
- Evaluation coverage: No integration with EEG/MRI signals; no prospective clinical deployment. Results remain within a controlled experimental setting.

## Related Work & Insights
- **vs. MedVidQA / SV-RCNet**: Unlike procedural videos with intentional movements, S³ addresses involuntary pathological movements, filling a gap in "involuntary motion + dense feature annotation."
- **vs. MotionBench**: General fine-grained motion benchmarks do not act as proxies for clinical capability, as evidenced by S³'s focus on clinical lateralization and temporal evolution.
- **vs. RadGraph**: While RadGraph uses graph structures for radiology report relations, Seizure-RQI adds narrative structure (onset/propagation/postictal) and temporal consistency, better suited for episode-based events.
- **vs. Traditional seizure methods (3D CNN, accelerometers)**: Discriminate-only methods lack the generative rationale and report capabilities of S³, which aligns more closely with the expert clinical reasoning process.

## Rating
- Novelty: ⭐⭐⭐⭐ The dataset, benchmark, metric, and training suite form a first-of-its-kind closed loop for seizure semiology.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across 11 MLLMs, SFT/GRPO paradigms, two-stage comparisons, and extensive ablations against clinical experts.
- Writing Quality: ⭐⭐⭐⭐ Clear hierarchical narrative with detailed mechanisms for each failure mode.
- Value: ⭐⭐⭐⭐⭐ Sets a high-reference standard for medical multimodal AI and report evaluation; the two-stage F1=0.96 approaches clinical viability thresholds.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] SynerMedGen: Synergizing Medical Multimodal Understanding with Generation via Task Alignment](synermedgen_synergizing_medical_multimodal_understanding_with_generation_via_tas.md)
- [\[NeurIPS 2025\] FAPEX: Fractional Amplitude-Phase Expressor for Robust Cross-Subject Seizure Prediction](../../NeurIPS2025/medical_imaging/fapex_fractional_amplitude-phase_expressor_for_robust_cross-subject_seizure_pred.md)
- [\[NeurIPS 2025\] THUNDER: Tile-level Histopathology image UNDERstanding benchmark](../../NeurIPS2025/medical_imaging/thunder_tile-level_histopathology_image_understanding_benchmark.md)
- [\[CVPR 2026\] GLEAM: A Multimodal Imaging Dataset and HAMM for Glaucoma Classification](../../CVPR2026/medical_imaging/gleam_a_multimodal_imaging_dataset_and_hamm_for_gl.md)
- [\[NeurIPS 2025\] SMMILE: An Expert-Driven Benchmark for Multimodal Medical In-Context Learning](../../NeurIPS2025/medical_imaging/smmile_an_expert-driven_benchmark_for_multimodal_medical_in-context_learning.md)

</div>

<!-- RELATED:END -->
