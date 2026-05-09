---
title: >-
  [Paper Note] CARE: Towards Clinical Accountability in Multi-Modal Medical Reasoning with an Evidence-Grounded Agentic Framework
description: >-
  [ICLR 2026][Medical Imaging][Medical VQA] This paper proposes CARE, a framework that decomposes medical VQA into a three-stage expert pipeline—entity proposal → referring segmentation → evidence-grounded QA—with RLVR fine-tuning applied to each VLM and GPT-5 serving as a dynamic coordinator for tool planning and CoT review. CARE achieves an average accuracy of 77.54% across four medical VQA benchmarks using only 10B parameters, surpassing the 32B end-to-end state-of-the-art (72.29%).
tags:
  - ICLR 2026
  - Medical Imaging
  - Medical VQA
  - Evidence-Grounded Reasoning
  - Agentic Framework
  - Referring Segmentation
  - Clinical Accountability
date: 2026-05-08
content_hash: 6162fdea4c1fd4ad
---

# CARE: Towards Clinical Accountability in Multi-Modal Medical Reasoning with an Evidence-Grounded Agentic Framework

**Conference**: ICLR 2026
**arXiv**: [2603.01607](https://arxiv.org/abs/2603.01607)
**Code**: [https://xypb.github.io/CARE-Project-Page/](https://xypb.github.io/CARE-Project-Page/)
**Area**: Medical Imaging / Multimodal VLM / Agent
**Keywords**: Medical VQA, Evidence-Grounded Reasoning, Agentic Framework, Referring Segmentation, Clinical Accountability

## TL;DR
This paper proposes CARE, a framework that decomposes medical VQA into a three-stage expert pipeline—entity proposal → referring segmentation → evidence-grounded QA—with RLVR fine-tuning applied to each VLM and GPT-5 serving as a dynamic coordinator for tool planning and CoT review. CARE achieves an average accuracy of 77.54% across four medical VQA benchmarks using only 10B parameters, surpassing the 32B end-to-end state-of-the-art (72.29%).

## Background & Motivation

**Background**: Multimodal large models (Lingshu, HuatuoGPT-Vision, MedGemma, etc.) continue to push the state of the art on medical VQA, yet nearly all approaches follow an end-to-end, single-pass paradigm—taking an image and question as input and directly producing an answer. This "black-box" mode cannot inform clinicians of *where* the model looked or *what evidence* it relied upon.

**Limitations of Prior Work**: The problem manifests at three levels. (1) **Non-auditability**: The reasoning process of end-to-end VLMs is opaque; clinicians cannot verify whether the model attended to the correct anatomical structures or lesion regions—a critical failure in clinical settings where accountability chains for misdiagnosis must be preserved. (2) **Decoupling of grounding and reasoning**: Although some works (MedPLIB, UniBiomed) equip VLMs with visual grounding heads, the localization results are not fed back into the reasoning process but serve only as auxiliary multi-task outputs, leaving answer quality unimproved by grounding. (3) **Fragility of single-model coupling**: Some general-domain methods (DeepEyes, Fan et al.) attempt to interleave grounding and reasoning within a single generative model, but this demands large-scale paired data and multi-turn RL; moreover, early localization errors directly amplify downstream reasoning hallucinations—once a small but critical lesion is missed, the entire subsequent CoT is built on a faulty premise.

**Key Challenge**: A fundamental tension exists between accuracy and accountability. Larger end-to-end models (e.g., Lingshu-32B, InternVL3-38B) achieve higher accuracy, but every step occurs inside a black box; smaller models can perform interpretable reasoning but lack sufficient capacity. Coupling localization and reasoning within a single VLM is difficult to train and prone to cascading hallucinations from single-point failures.

**Goal**: (1) How can every reasoning step in medical VQA be supported by pixel-level visual evidence? (2) How can such accountability be achieved without sacrificing accuracy? (3) How can a composition of small models outperform a single large model?

**Key Insight**: The authors observe that clinicians' diagnostic workflows are inherently stage-wise—first hypothesizing relevant anatomical structures or lesions (entity candidates), then precisely localizing these regions in the image (visual grounding), and finally integrating local details with global context to reach a conclusion. This human workflow is naturally auditable: each stage has well-defined inputs, outputs, and inspectable intermediate results.

**Core Idea**: Three lightweight expert models (an entity-proposal VLM, a referring segmentation model, and an evidence-grounded VQA VLM) are used to simulate the clinician's staged diagnostic process, coordinated by a powerful VLM that performs dynamic planning and answer review, achieving "small-model toolchain > large-model single-pass inference."

## Method

### Overall Architecture
CARE receives a medical image and a natural-language question and executes three decoupled sub-tasks:

1. **Medical Entity Proposal**: InternVL3-2B (RLVR fine-tuned) proposes candidate medical entities (anatomical structures, lesion names, instruments, etc.) conditioned on the question and image, analogous to a clinician first deciding "which regions deserve attention."
2. **Entity Referring Segmentation**: A segmentation model built on SA-Med-2D generates pixel-level ROI masks for each proposed entity and outputs confidence scores to filter unreliable segmentation results.
3. **Evidence-Grounded VQA (EG-VQA)**: InternVL3-8B (SFT + RLVR fine-tuned) performs reasoning given the original image together with three forms of visual evidence (zoom-in crops, binary masks, and global indicators).

The framework operates in two modes: **CARE-Flow** (a static pipeline that executes all three evidence types and aggregates via majority voting) and **CARE-Coord** (GPT-5 serves as a dynamic coordinator that autonomously selects evidence types, plans tool calls, and reviews CoT–answer consistency).

### Key Designs

1. **RLVR Training of the Entity-Proposal VLM**:

    - *Function*: Models the open-ended task of "which medical entities in the image are relevant to the question" as a trainable generation problem.
    - *Mechanism*: Since no public entity-proposal dataset exists, the authors synthesize (image, question, entity) triplets by randomly sampling segmentation masks and entity names from SA-Med-20M, yielding 10k training and 1k test samples. Training uses the DAPO algorithm with a four-component reward: (a) **Similarity reward** $R_{\text{sim}}$: MiniLM-L6-v2 encodes predicted and GT entity embeddings; a cosine similarity matrix is constructed and the Kuhn–Munkres algorithm finds the optimal bipartite matching, yielding the average similarity of the best-matched pairs; (b) **Count reward** $R_{\text{count}}$: 1 if the number of proposed entities is between 1 and 5, else 0; (c) **Deduplication penalty** $R_{\text{rep}} = 1/(r+1)$, where $r$ is the number of repeated entities; (d) **Format reward** $R_{\text{format}}$ enforcing `<think>`/`<answer>` tags.
    - *Design Motivation*: Compared to binary exact-match rewards, the continuous embedding-similarity reward avoids vanishing gradients and is more robust to the domain gap between synthetic training data and real clinical questions. Kuhn–Munkres matching is more stable than greedy matching—greedy matching grants a reward as soon as any entity pair is matched, which can bias learning. Ablations show that KM + Sim yields an entity accuracy of 85.2%, far exceeding Greedy + Binary (72.8%).

2. **Entity Referring Segmentation Model**:

    - *Function*: Given a text description of a medical entity, produces a pixel-level segmentation mask in the image.
    - *Mechanism*: Text understanding capability is added to SA-Med-2D (a medical-imaging adaptation of SAM, 600M parameters). A frozen Bio-ClinicalBERT encoder converts entity names into token sequences; these are concatenated with image tokens along with binary modality embeddings (image = 0, text = 1) and fed into the SAM encoder. During decoding, only image tokens serve as keys/values, while text tokens—after projection—serve as queries to the SAM mask decoder. Fine-tuning updates only the image projector, encoder, and text projector.
    - *Confidence filtering*: Mask confidence is defined as $C(M_p) = 1 - \text{Entropy}(M_p) / \log(2)$; masks below threshold $\tau_C = 70\%$ are discarded to prevent low-quality segmentations from contaminating downstream reasoning.
    - *Design Motivation*: A dedicated expert segmentation model is preferred over the grounding heads built into VLMs because it achieves superior localization on small but clinically critical lesions. The proposed model achieves an average Dice of 81.9% on the MeCo-G benchmark, surpassing LISA-7B (62.7%) and BiomedParse (30.1%).

3. **Evidence-Grounded VQA (EG-VQA)**:

    - *Function*: Transforms segmented ROIs into three complementary forms of visual evidence to enhance VLM reasoning.
    - *Three evidence types*: (a) **Zoom-in crop**—crops and magnifies the region around the ROI, providing high-resolution local detail suitable for texture/morphology questions; (b) **Binary mask**—feeds the binary mask as an additional image channel, serving as a spatial attention prior suitable for location/shape questions; (c) **Global**—uses an all-ones mask when local localization is unnecessary (e.g., identifying imaging modality or scan orientation), preserving the global view.
    - *Training strategy*: Two-stage fine-tuning. In the first stage, the trained entity-proposal and segmentation models annotate visual cues for existing VQA data. In the second stage, the annotated data first undergoes SFT followed by DAPO-RFT. RFT incorporates a CoT length reward $R_{\text{length}} = 0.25 \cdot \min(1, |\hat{y}|/L)$ to encourage thorough reasoning, in addition to accuracy and format rewards.
    - *Design Motivation*: Masks are not overlaid directly on the original image because pixel values in medical images carry physical meaning (e.g., HU values in CT), and overlay would corrupt the information. Three complementary evidence types are mixed during training, enabling the model to leverage different levels of visual granularity depending on the question type.

### Loss & Training
The framework adopts the DAPO (Decoupled Asymmetric PPO) algorithm for RLVR throughout. For the entity-proposal VLM, the reward is $R_{\text{Entity}} = R_{\text{sim}} + R_{\text{count}} + R_{\text{rep}} + R_{\text{format}}$; for EG-VQA, the reward is $R_{\text{EG-VQA}} = R_{\text{acc}} + R_{\text{format}} + R_{\text{length}}$. Key design choices include:

- **SFT → RFT two-stage pipeline**: SFT injects new knowledge (memorizing medical facts), while RFT optimizes the output distribution to produce well-reasoned CoT; the two stages are complementary.
- The combination of **SFT + DAPO + length reward** is optimal in ablations, outperforming SFT alone by +2.4% and DAPO alone by +3.6%.
- The segmentation model is trained with standard Dice + CE loss; only projection layers are fine-tuned, preserving SAM's pretrained visual features.

## Key Experimental Results

### Main Results (4 Medical VQA Benchmarks, Accuracy %)

| Method | Params | OMVQA-3k | VQA-RAD | SLAKE | VQA-Med-2019 (OOD) | Avg. |
|--------|--------|----------|---------|-------|---------------------|------|
| GPT-4o | — | 64.07 | 58.54 | 63.55 | 59.60 | 61.44 |
| GPT-5 | — | 74.73 | 63.19 | 67.75 | 62.20 | 66.97 |
| InternVL3-8B | 8B | 75.97 | 61.86 | 66.13 | 57.40 | 65.34 |
| HuatuoGPT-Vision-34B | 34B | 76.80 | 60.75 | 64.12 | 60.60 | 65.57 |
| Lingshu-32B | 32B | 83.97 | 64.75 | 82.25 | 58.20 | 72.29 |
| CARE-Flow-S | 4B | 94.53 | 56.32 | 78.44 | 53.60 | 70.72 |
| CARE-Flow-B | 10B | 96.17 | 63.64 | 83.21 | 56.60 | 74.91 |
| **CARE-Coord-B** | **10B** | **97.97** | **68.29** | **83.11** | **60.80** | **77.54** |

CARE-Flow-B (10B) outperforms same-scale baselines by +10.9% and exceeds Lingshu-32B by +2.6%. Adding the coordinator, CARE-Coord-B surpasses Lingshu-32B by +5.2%.

### Ablation Study — Visual Evidence and Coordinator Effectiveness

| Training Evidence | Coordinator | ID Avg. | OOD | Overall | vs. Baseline |
|-------------------|-------------|---------|-----|---------|--------------|
| None | None | 77.9 | 56.0 | 72.4 | +0.0 |
| Mask | None | 79.6 | 54.0 | 73.2 | +0.8 |
| Zoom | None | 79.5 | 56.8 | 73.8 | +1.4 |
| Mask + Zoom | None | 80.2 | 55.6 | 74.1 | +1.7 |
| All three (CARE-Flow) | None | 81.0 | 56.6 | 74.9 | +2.5 |
| All three | Planning | 80.8 | 53.4 | 74.8 | +2.4 |
| All three (CARE-Coord) | Planning + Review | 83.1 | 60.8 | **77.5** | **+5.1** |

### Training Strategy Ablation

| Training Strategy | ID Avg. | OOD | Overall | vs. Baseline |
|-------------------|---------|-----|---------|--------------|
| Baseline (InternVL3-8B) | 67.9 | 57.4 | 65.3 | +0.0 |
| + SFT | 77.8 | 56.6 | 72.5 | +7.2 |
| + GRPO | 75.2 | 54.0 | 69.9 | +4.6 |
| + DAPO | 77.0 | 54.2 | 71.3 | +6.0 |
| + SFT + DAPO | 79.3 | 56.2 | 73.5 | +8.2 |
| + SFT + DAPO + $R_{\text{length}}$ (CARE-Flow) | 81.0 | 56.6 | **74.9** | **+9.6** |

### Key Findings
- **Value of combining visual evidence**: Using all three evidence types outperforms the no-evidence baseline by +2.5%; the three types are mutually complementary—zoom-in alone performs best individually (+1.4%), but the combination is more robust.
- **Coordinator review is the primary source of gain**: Adding planning alone yields no appreciable improvement (+2.4% vs. +2.5%), but incorporating CoT–answer review yields a jump to +5.1%, demonstrating that iterative review—not merely planning—is the coordinator's core contribution.
- **SFT and DAPO are complementary**: SFT alone (+7.2%) outperforms DAPO alone (+6.0%), but their combination (+8.2%) is superior, and adding the length reward achieves the best result (+9.6%), validating the hypothesis that SFT injects knowledge while RFT optimizes reasoning.
- **Coordinator behavior analysis**: The GPT-5 coordinator modifies 7.89% of samples overall; 4.84% are corrections (✗→✓) and 3.05% are erroneous changes (✓→✗), yielding a net gain of +1.79%. The correction rate is higher on OOD data (7.6% ✗→✓ on VQA-Med-2019), indicating that a strong coordinator improves generalization.
- **Expert vs. general-purpose segmentation**: Replacing the proposed SA-Med-2D-based segmentation model with BiomedParse causes a 3.4% drop in VQA accuracy, confirming that expert segmentation quality is critical to the entire pipeline.
- **GPT-5 vs. alternative coordinators**: The GPT-5 coordinator (77.5%) substantially outperforms GPT-4o (73.3%) and InternVL3-38B (74.0%), as weaker coordinators tend to select incorrect evidence types or over-rewrite expert answers.

## Highlights & Insights
- **System design philosophy of "clinical workflow biomimicry"**: Rather than simply scaling up VLMs, the framework formalizes the clinician's "hypothesize → localize → evidence-based diagnosis" workflow into a three-stage pipeline. This design philosophy ensures that every step produces auditable intermediate results (proposed entity lists, segmentation masks, selected evidence types), inherently satisfying clinical requirements for traceability. The same philosophy is transferable to any high-stakes decision-making domain requiring accountability (legal, financial, etc.).
- **Clever RLVR design for open-ended concept proposal**: Entity proposal has no fixed answer space, and binary rewards under standard RL suffer from vanishing gradients in this setting. The authors address this by using embedding similarity with Kuhn–Munkres optimal matching as a continuous reward signal, preserving semantic flexibility while providing stable gradients. This reward-shaping scheme is directly reusable in any RL training scenario that requires soft matching between a generated set and a reference set.
- **Parameter efficiency of the small-model toolchain**: The 10B modular pipeline outperforms a 32B end-to-end model; even the 4B variant matches InternVL3-38B. This demonstrates that, in specialized vertical domains, carefully designed agent + expert-tool strategies are more effective than blindly scaling model size—the key insight being that each module need only solve a relatively simple sub-problem: 2B suffices for entity proposal, and 600M suffices for segmentation.

## Limitations & Future Work
- **Dependence on a strong coordinator**: The performance advantage of CARE-Coord is highly dependent on GPT-5; replacing it with GPT-4o causes a 4.2% drop, and substituting open-source InternVL3-38B causes a 3.5% drop. Although a self-trained InternVL3-8B coordinator outperforms majority voting, it cannot perform CoT review. Training a compact yet reliable coordinator remains an important open problem.
- **Error cascading in the three-stage pipeline**: The sequential dependency of entity proposal → segmentation → VQA means that upstream errors are unrecoverable. While confidence filtering ($\tau_C = 70\%$) discards unreliable segmentations, erroneous entity proposals themselves cannot be detected by downstream components.
- **Limitations of synthetic training data**: The entity-proposal training data is synthesized from segmentation datasets with limited question diversity, potentially failing to cover the full range of real clinical queries.
- **Deployment cost**: CARE-Coord requires GPT-5 API calls for planning and review at each inference step; latency and cost present practical obstacles for clinical deployment.
- **Remaining OOD generalization gap**: CARE-Coord achieves only 60.8% on VQA-Med-2019, a notable gap compared to the in-distribution average of 83.1%. CARE-Flow performs even worse on OOD data (56.6%), indicating that the visual evidence pipeline's adaptability to out-of-distribution data requires further improvement.

## Related Work & Insights
- **vs. Lingshu-32B** (end-to-end medical VLM): Lingshu achieves strong in-domain performance via large-scale medical pretraining of a 32B model, but produces no intermediate evidence. CARE surpasses it with a 10B modular system; the key advantage lies in auditability and parameter efficiency, while the primary drawback is higher latency due to multiple model calls.
- **vs. DeepEyes-7B** (single-model visual reasoning): DeepEyes interleaves grounding and reasoning within a single VLM, requiring multi-turn interaction and extensive training. CARE decouples these two tasks across expert models, avoiding error amplification within a single model and eliminating the need for complex multi-turn RL.
- **vs. MedVLM-R1-2B** (medical reasoning model): MedVLM-R1 employs CoT reasoning with 2B parameters but performs no grounding, achieving an average of only 51.35%. CARE demonstrates that combining reasoning with grounding substantially outperforms reasoning alone.
- **vs. BiomedParse / SA-Med-2D** (segmentation models): The proposed segmentation model achieves a Dice of 81.9% on MeCo-G, substantially outperforming BiomedParse (30.1%); replacing it in the pipeline causes a 3.4% VQA drop, confirming that expert segmentation quality is critical to overall system performance.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — Formalizing the clinical diagnostic workflow as an agentic pipeline is genuinely inspiring, and the RLVR + KM matching reward design is elegant; however, the three-stage decomposition framework itself has precedents in the general domain.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Four benchmarks × multiple baselines + ablations across five dimensions (evidence types, training strategy, coordinator, segmentation model, entity proposal), each yielding clear conclusions.
- **Writing Quality**: ⭐⭐⭐⭐ — The paper is clearly structured with informative figures and tables, though methodological details are distributed across the main text and appendix, requiring frequent cross-referencing.
- **Value**: ⭐⭐⭐⭐⭐ — Proposes a concrete technical pathway to "accountable" medical AI rather than merely advocating for interpretability in the abstract; the story of 10B outperforming 32B has practical significance for resource-constrained healthcare institutions.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] PulseMind: A Multi-Modal Medical Model for Real-World Clinical Diagnosis](../../AAAI2026/medical_imaging/pulsemind_a_multi-modal_medical_model_for_real-world_clinical_diagnosis.md)
- [\[ICLR 2026\] MMedAgent-RL: Optimizing Multi-Agent Collaboration for Multimodal Medical Reasoning](mmedagent-rl_optimizing_multi-agent_collaboration_for_multimodal_medical_reasoni.md)
- [\[CVPR 2026\] EMAD: Evidence-Centric Grounded Multimodal Diagnosis for Alzheimer's Disease](../../CVPR2026/medical_imaging/emad_evidence-centric_grounded_multimodal_diagnosis_for_alzheimers_disease.md)
- [\[ICLR 2026\] MedAgentGym: A Scalable Agentic Training Environment for Code-Centric Reasoning in Biomedical Data Science](medagentgym_agentic_training_biomedical.md)
- [\[NeurIPS 2025\] CureAgent: A Training-Free Executor-Analyst Framework for Clinical Reasoning](../../NeurIPS2025/medical_imaging/cureagent_a_training-free_executor-analyst_framework_for_clinical_reasoning.md)

<!-- RELATED:END -->
