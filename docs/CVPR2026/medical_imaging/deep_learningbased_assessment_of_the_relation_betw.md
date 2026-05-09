---
title: >-
  [Paper Note] Deep Learning-based Assessment of the Relation Between the Third Molar and Mandibular Canal on Panoramic Radiographs using Local, Centralized, and Federated Learning
description: >-
  [CVPR 2026][Medical Imaging][Federated Learning] This work systematically compares three training paradigms—local learning (LL), federated learning (FL), and centralized learning (CL)—on cropped panoramic dental radiographs partitioned by 8 independent annotators, targeting a binary classification task of third molar–mandibular canal overlap. The study establishes a consistent performance ranking of CL > FL > LL (AUC: 0.831, 0.757, and 0.672, respectively), demonstrating that FL substantially outperforms site-independent training while preserving data privacy.
tags:
  - CVPR 2026
  - Medical Imaging
  - Federated Learning
  - Panoramic Radiograph
  - Third Molar
  - Mandibular Canal
  - Privacy Preservation
date: 2026-05-08
content_hash: d0a73b091a180e56
---

# Deep Learning-based Assessment of the Relation Between the Third Molar and Mandibular Canal on Panoramic Radiographs using Local, Centralized, and Federated Learning

**Conference**: CVPR 2026
**arXiv**: [2603.11850](https://arxiv.org/abs/2603.11850)
**Code**: None
**Area**: Medical Image Analysis / Federated Learning / Dental Imaging
**Keywords**: Federated Learning, Panoramic Radiograph, Third Molar, Mandibular Canal, Privacy Preservation

## TL;DR
This work systematically compares three training paradigms—local learning (LL), federated learning (FL), and centralized learning (CL)—on cropped panoramic dental radiographs partitioned by 8 independent annotators, targeting a binary classification task of third molar–mandibular canal overlap. The study establishes a consistent performance ranking of CL > FL > LL (AUC: 0.831, 0.757, and 0.672, respectively), demonstrating that FL substantially outperforms site-independent training while preserving data privacy.

## Background & Motivation

**Background**: Impacted mandibular third molars (wisdom teeth) represent one of the most frequent procedures in oral surgery. When a wisdom tooth is in close spatial proximity to the mandibular canal—the bony conduit housing the inferior alveolar nerve—extraction carries a risk of nerve injury, potentially causing permanent paresthesia of the lower lip and chin. Orthopantomography (OPG) is the standard preoperative modality for assessing the spatial relationship between the molar and the mandibular canal.

**Limitations of Prior Work**: Determining overlap between the molar and the mandibular canal on panoramic radiographs relies on subjective radiologist expertise, with considerable inter-rater variability. Automated classification could support clinical triage and reduce unnecessary CBCT referrals, which involve higher radiation doses and greater cost. However, dental imaging data are distributed across heterogeneous clinical institutions and annotation teams, and privacy regulations such as GDPR and HIPAA prevent direct data pooling for centralized training.

**Key Challenge**: CL requires data aggregation yet violates privacy constraints, while LL preserves privacy but yields models with poor generalization due to limited per-site data. FL offers a middle ground by enabling multi-center collaboration without sharing raw data, but its performance–privacy trade-off in real-world dental imaging scenarios has not been systematically validated.

**Goal**: To quantify the performance gap among LL, FL, and CL on dental imaging data exhibiting genuine inter-annotator variability (a natural Non-IID characteristic), and to answer whether FL can serve as a privacy-preserving substitute for CL.

**Key Insight**: A semantically clear binary classification task (molar–canal overlap vs. no overlap) is selected, with a pretrained ResNet-34 as the unified backbone. Data from 8 independent annotators are treated as 8 FL clients, and the three training paradigms are compared under controlled conditions.

**Core Idea**: Federated learning can serve as a viable alternative to centralized training in a multi-annotator dental imaging setting, yielding substantially higher performance than site-independent local training.

## Method

### Overall Architecture
The full pipeline comprises four stages: (1) cropping ROIs containing the third molar and mandibular canal from panoramic radiographs; (2) partitioning the dataset by 8 independent annotators to simulate a multi-center setting with 8 FL clients; (3) training a pretrained ResNet-34 for binary classification (overlap vs. no-overlap) under each of the three paradigms (LL, FL, CL); and (4) evaluating model performance via two protocols—per-client (locally optimized threshold) and pooled (global threshold)—supplemented by training dynamics analysis and Grad-CAM visualization.

### Key Designs

1. **Fair Three-Paradigm Comparison Framework**

   - *Function*: Compare LL, FL, and CL under identical data partitioning and backbone settings.
   - *Mechanism*: LL — each annotator's data independently trains a ResNet-34 model, yielding 8 independent models; FL — the FedAvg algorithm aggregates model parameters on the server as $w_{t+1} = \sum_{k=1}^{K} \frac{n_k}{n} w_t^k$, with only model updates (not raw data) leaving each client; CL — all 8 clients' data are centrally pooled to train a single unified model. All three paradigms share the same pretrained ResNet-34 initialization and hyperparameter configuration.
   - *Design Motivation*: Controlling variables is a prerequisite for fair comparison. The natural data partition by 8 annotators more faithfully reflects the data heterogeneity found in real clinical scenarios than artificial random splits.

2. **Dual-Track Evaluation Protocol**

   - *Function*: Evaluate model performance from both local deployment and cross-center generalization perspectives.
   - *Mechanism*: Per-client evaluation — the classification threshold is independently optimized on each client's validation set, and models are assessed on the local test set using the locally optimal threshold, reflecting best-case local deployment; pooled evaluation — a single global threshold is applied across the merged test set, reflecting cross-center generalization. Primary metrics include AUC and threshold-dependent accuracy, sensitivity, and specificity.
   - *Design Motivation*: Real-world FL deployment may follow either a "global model + local threshold" or a "global model + global threshold" strategy; the two evaluation tracks correspond to these distinct deployment scenarios.

3. **Interpretability and Training Dynamics Analysis**

   - *Function*: Analyze behavioral differences among the three paradigms via Grad-CAM and training curves.
   - *Mechanism*: Grad-CAM visualizes attention heatmaps of models trained under each paradigm, examining whether attention is directed toward anatomically relevant regions (third molar root apex and mandibular canal trajectory). Training curves monitor convergence behavior and overfitting across paradigms; server-side aggregation signals (e.g., global validation loss, consistency of client update magnitudes) are additionally monitored in the FL setting to track global training stability.
   - *Design Motivation*: Beyond identifying which paradigm performs better, understanding why is essential. Grad-CAM validates whether models rely on correct anatomical features rather than data artifacts, while training dynamics analysis elucidates the mechanisms underlying LL overfitting.

### Loss & Training
- ImageNet-pretrained ResNet-34 serves as the backbone, with the final fully connected layer replaced by a binary classification head.
- Binary cross-entropy loss is applied, with labels denoting overlap (third molar overlapping the mandibular canal) and no-overlap.
- FL employs standard FedAvg aggregation: each client trains for a fixed number of local epochs before uploading model weights, which the server averages in a weighted fashion and redistributes to all clients.
- Server-side aggregation signals (e.g., global validation loss, consistency of client update magnitudes) are monitored throughout training to assess stability.

## Key Experimental Results

### Main Results

| Training Paradigm | AUC | Accuracy (%) | Notes |
|---|---|---|---|
| CL (Centralized Learning) | **0.831** | **78.2** | Highest performance; all data centrally pooled |
| FL (Federated Learning) | 0.757 | 70.3 | Intermediate; privacy-preserving |
| LL (Local Learning) | 0.619–0.734 (mean 0.672) | — | Lowest and highest variance |

FL achieves a mean AUC gain of +0.085 over LL (0.672 → 0.757); CL further improves over FL by +0.074 (0.757 → 0.831).

### Ablation Study

| Configuration | Key Metric | Description |
|---|---|---|
| LL across 8 clients | AUC 0.619–0.734 | Span >0.1, indicating substantial inter-annotator distribution heterogeneity |
| FL vs. LL best | AUC 0.757 vs. 0.734 | FL surpasses the strongest local model |
| CL vs. FL gap | AUC 0.831 vs. 0.757 | A gap of 0.074; data centralization retains non-negligible advantages |
| Training curve overfitting | LL > FL > CL | LL exhibits the most severe overfitting |

### Key Findings
- The performance ranking CL > FL > LL holds consistently in both AUC and accuracy, confirming the value of data sharing.
- FL is an effective privacy-preserving alternative to CL, though an AUC gap of approximately 0.074 remains.
- LL models overfit most severely—limited per-client data prevents learning generalizable representations.
- Grad-CAM reveals that CL and FL models concentrate attention on anatomically relevant regions (root apex and canal trajectory), whereas LL model attention is more diffuse and may exploit non-anatomical artifacts.
- The LL performance spread across 8 clients exceeds 0.1 AUC, reflecting the substantial impact of genuine inter-annotator variability on model quality.

## Highlights & Insights
- **First systematic FL validation in dental AI**: Applications of federated learning in dental imaging were virtually absent prior to this work; this paper provides the first benchmark.
- **Annotator-based partitioning is more realistic than random splits**: Treating 8 annotators as 8 FL clients naturally introduces Non-IID characteristics driven by differences in annotation standards, more closely mirroring clinical reality than artificially induced heterogeneity.
- **Rigorous dual-track evaluation**: Distinguishing per-client from pooled evaluation corresponds to distinct real-world deployment strategies.
- **Grad-CAM enhances clinical credibility**: The finding that CL and FL models attend to anatomically correct regions increases confidence in their clinical trustworthiness.
- **Unambiguous performance ranking**: The consistent conclusion CL > FL > LL provides a clear reference for clinical decision-makers.

## Limitations & Future Work
- The binary classification task (overlap vs. no-overlap) is overly coarse; clinical practice requires finer-grained risk stratification (e.g., Winter classification, Pell–Gregory classification).
- Only standard FedAvg is employed; advanced algorithms designed for data heterogeneity—such as FedProx, SCAFFOLD, and FedBN—are not explored, despite their potential to narrow the FL–CL performance gap.
- The scale of 8 annotators is limited; validation across larger multi-center settings (e.g., 20+ institutions) would be more convincing.
- Inter-rater agreement metrics (e.g., Cohen's Kappa, Fleiss' Kappa) are not reported, making it impossible to quantify the impact of annotation noise on training.
- ResNet-34 is a relatively dated backbone; more modern architectures such as Vision Transformers are not evaluated.
- Only cropped ROI regions are used; end-to-end detection-plus-classification pipelines on full panoramic radiographs are not explored.

## Related Work & Insights
- **vs. General medical FL studies (e.g., FedAvg on X-ray/CT)**: This work focuses on a dental-specific task, and partitioning data by annotator rather than institution is a distinctive setup that more closely reflects Non-IID characteristics driven by annotation standard heterogeneity.
- **vs. Traditional dental AI**: Most prior work relies on single-center centralized training; this paper is the first to introduce an FL paradigm for molar–mandibular canal relationship assessment.
- **vs. Advanced FL algorithms (FedProx/FedBN/SCAFFOLD)**: FedAvg serves only as a baseline, and the paper explicitly acknowledges room for improvement; nevertheless, the baseline results already establish the feasibility of FL.
- **vs. CBCT-based studies in the same setting**: Panoramic radiographs entail substantially lower acquisition cost and radiation dose than CBCT; automated classification can triage cases that genuinely require CBCT referral.

## Rating
- **Novelty**: ⭐⭐⭐ The first systematic FL validation in dental imaging carries some novelty, but there is no methodological innovation beyond FedAvg + ResNet-34.
- **Experimental Thoroughness**: ⭐⭐⭐ The three-paradigm comparison is well-structured and the dual-track evaluation adds rigor, but FL algorithm exploration is limited, data scale is modest, and inter-rater analysis is absent.
- **Writing Quality**: ⭐⭐⭐⭐ The structure is clear, the abstract is information-dense, and the experimental rationale is coherently articulated.
- **Value**: ⭐⭐⭐ Provides a valuable baseline reference for privacy-preserving training in dental AI, though practical utility is constrained by the coarseness of the binary classification task.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Unlocking Multi-Site Clinical Data: A Federated Approach to Privacy-First Child Autism Behavior Analysis](unlocking_multi-site_clinical_data_a_federated_approach_to_privacy-first_child_a.md)
- [\[CVPR 2026\] OmniFM: Toward Modality-Robust and Task-Agnostic Federated Learning for Heterogeneous Medical Imaging](omnifm_toward_modality-robust_and_task-agnostic_federated_learning_for_heterogen.md)
- [\[CVPR 2026\] Federated Modality-specific Encoders and Partially Personalized Fusion Decoder for Multimodal Brain Tumor Segmentation](federated_modality-specific_encoders_and_partially_personalized_fusion_decoder_f.md)
- [\[CVPR 2026\] Interpretable Cross-Domain Few-Shot Learning with Rectified Target-Domain Local Alignment](interpretable_cross-domain_few-shot_learning_with_rectified_target-domain_local_.md)
- [\[CVPR 2026\] FedVG: Gradient-Guided Aggregation for Enhanced Federated Learning](fedvg_gradient-guided_aggregation_for_enhanced_federated_learning.md)

<!-- RELATED:END -->
