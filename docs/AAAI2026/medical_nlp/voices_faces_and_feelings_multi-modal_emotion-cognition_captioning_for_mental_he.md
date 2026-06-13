---
title: >-
  [Paper Note] Voices, Faces, and Feelings: Multi-modal Emotion-Cognition Captioning for Mental Health Understanding
description: >-
  [AAAI 2026][Medical NLP][Emotion-cognition captioning] This paper proposes the Emotion-Cognition cooperative Multi-modal Captioning (ECMC) task and framework. A dual-stream BridgeNet extracts emotion and cognition featur…
tags:
  - "AAAI 2026"
  - "Medical NLP"
  - "Emotion-cognition captioning"
  - "multimodal"
  - "mental health"
  - "large language models"
  - "depression detection"
date: 2026-05-08
content_hash: 0abbe6e7d24a9ca3
---

# Voices, Faces, and Feelings: Multi-modal Emotion-Cognition Captioning for Mental Health Understanding

**Conference**: AAAI 2026
**arXiv**: [2603.01816](https://arxiv.org/abs/2603.01816)  
**Code**: [github](https://github.com/zhouzyhfut/ECMC)  
**Area**: Medical Imaging
**Keywords**: Emotion-cognition captioning, multimodal, mental health, large language models, depression detection

## TL;DR
This paper proposes the Emotion-Cognition cooperative Multi-modal Captioning (ECMC) task and framework. A dual-stream BridgeNet extracts emotion and cognition features from video, audio, and text, and a LLaMA decoder generates natural language descriptions. The system provides interpretable emotion-cognition profiles for mental health assessment, substantially improving both diagnostic accuracy and explainability.

## Background & Motivation

Mental health disorders are a growing global crisis — over 300 million people suffer from depression, and the WHO projects that untreated mental disorders will account for 13% of the total disease burden by 2030. Existing computer-aided diagnostic methods face three core challenges:

**Challenge 1: Classification paradigms lack interpretability.** Most methods classify multimodal data into a disorder category (e.g., "depressed" / "anxious") without revealing *which cues* are clinically relevant. A single classification label offers little diagnostic value to clinicians.

**Challenge 2: LLM-based methods rely heavily on textual semantics.** While LLMs excel at natural language understanding, their application to mental health analysis largely reduces to detecting symptom-related vocabulary, leaving non-verbal signals such as facial expressions and vocal prosody — critical in clinical observation — uncaptured.

**Challenge 3: Emotion-cognition patterns are underutilized.** Neuroscientific evidence indicates that psychiatric disorders manifest not only in reported symptoms but also in the dynamic mechanisms of affective and cognitive processing. For instance, depression is typically associated with sustained emotional stagnation and inhibited cognitive function, whereas anxiety disorders present as rapid fluctuations in both affect and cognition. Existing methods rarely examine these patterns from a multimodal perspective.

**Core Idea**: Reformulate mental health analysis from a *classification task* to a *captioning task* — rather than outputting a label, the system generates natural language descriptions characterizing a patient's emotional state and cognitive impairments, thereby providing interpretable evidence for clinical diagnosis. This defines the ECMC (Emotion-Cognition cooperative Multi-modal Captioning) task.

## Method

### Overall Architecture

ECMC adopts an encoder–decoder architecture (Figure 2) comprising three core components:

1. **Modality-specific encoders**: Extract initial representations from video, audio, and text independently.
2. **Dual-stream BridgeNet**: Q-Former-based modules for emotion and cognition feature extraction and fusion.
3. **LLaMA decoder**: Converts aligned emotion-cognition features into natural language descriptions.

The overall pipeline is: **multimodal input → initial features → dual-stream BridgeNet compression and fusion → E-embedding + C-embedding → LLaMA caption generation → aggregated user profile → assisted diagnosis**.

### Key Designs

#### 1. Modality-Specific Encoders

Given an utterance sample $\bm{x}_i = \{\bm{X}_v, \bm{X}_a, \bm{X}_t\}$, three pretrained models extract initial representations:

- **Video**: VideoMAE extracts facial expression and body language features.
- **Audio**: HuBERT extracts acoustic features such as vocal pitch and speech rate.
- **Text**: BERT extracts textual semantic features.

Encoder parameters are **frozen** during training due to limited clinical data, which makes end-to-end encoder training infeasible. However, pretrained models focus on frame-level representations and cannot capture emotion- and cognition-relevant semantics.

#### 2. Dual-Stream BridgeNet (Core Contribution)

Inspired by BLIP-2, a Q-Former-based dual-stream BridgeNet is designed to compress and disentangle emotion and cognition representations.

**Emotion BridgeNet**:

Learnable query tokens $\bm{Q}_m$ are introduced for each modality. Self-attention models inter-query dependencies, and cross-attention extracts information from modality features:

$$\bm{Z}_m = \text{softmax}\left(\frac{\bm{Q}'_m \bm{W}^{\prime(c)}_q (\bm{H}'_m \bm{W}^{\prime(c)}_k)^\top}{\sqrt{d_k}}\right) \bm{H}'_m \bm{W}^{\prime(c)}_v$$

Representations from the three modalities are then concatenated, projected, and normalized to produce the E-embedding $\bm{h}_e$.

**Emotion contrastive learning**: Representations are grouped into negative/neutral/positive categories along the valence dimension, and label-guided contrastive learning is applied. The loss jointly optimizes intra-class compactness and inter-class separability:

$$\mathcal{L}_{\text{emo}} = -\frac{1}{N}\sum_{i=1}^{N}\frac{1}{|\{j:\bm{M}_{ij}=1\}|}\sum_{j:\bm{M}_{ij}=1}\log p_{ij} + \frac{1}{N}\sum_{i=1}^{N}\log(1+\sum_{j:\bm{M}_{ij}=0}\exp(\bm{S}_{ij}))$$

**Cognition BridgeNet**:

Structurally analogous to the emotion branch, but dedicated to extracting cognitive impairment representations. Guided by the MMSE clinical cognitive scale, four types of cognitive impairment are addressed: **orientation disorder, attention disorder, memory disorder, and language disorder**.

Since a single sample may exhibit multiple concurrent cognitive impairments (multi-label setting), a **Jaccard similarity**-based multi-label contrastive learning objective is designed:

$$\bm{W}_{ij} = \frac{|\bm{y}_{c,i} \cap \bm{y}_{c,j}|}{|\bm{y}_{c,i} \cup \bm{y}_{c,j}|}$$

Samples with greater label overlap are pulled closer together, enabling soft contrastive learning over multi-label cognitive impairments.

#### 3. LLaMA Decoder

The BOS token, E-embedding, C-embedding, and prompt are concatenated and fed into LLaMA:

$$\bm{u} = \mathcal{F}_{llm}(\text{concat}(\text{<BOS>}, \bm{h}_e, \bm{h}_c, \text{Prompt}))$$

Utterance-level captions are aggregated to generate a user profile $\bm{p}$, which is then used to assist an LLM in mental disorder detection.

### Loss & Training

**Two-stage training**:

- **Stage 1**: Joint training of emotion and cognition representation extraction ($\mathcal{L}_1 = \mathcal{L}_{\text{emo}} + \mathcal{L}_{\text{cog}}$). Modality encoders are frozen; Q-Former is initialized with BERT pretrained weights. Trained for 500 epochs with batch size 64.
- **Stage 2**: Fine-tuning BridgeNet to align features with the LLM input space ($\mathcal{L}_2 = \text{CELoss}(\hat{\bm{u}}, \bm{u})$). Modality encoders and LLM parameters are frozen; micro batch size is 8.

Total parameters: ~7.6B, of which ~605M are trainable. Optimized with DeepSpeed ZeRO stage-2.

## Key Experimental Results

### Main Results

**Emotion captioning performance**:

| Method | Modality | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | F_BERT |
|--------|----------|--------|--------|--------|---------|--------|
| InternVL-2.5-8B | VT | 8.35 | 0.96 | 10.06 | 13.98 | 6.48 |
| Sa2VA-8B | VT | 14.36 | 2.14 | 15.35 | 20.34 | 12.28 |
| Qwen2.5-Omni-7B | AVT | 12.74 | 1.42 | 13.31 | 16.99 | 8.93 |
| CPsyCoun | T | 17.44 | 2.33 | 15.07 | 18.90 | 9.30 |
| **Ours** | **AVT** | **34.76** | **8.28** | **29.47** | **24.91** | **27.13** |

**Cognition captioning performance**:

| Method | Modality | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | F_BERT |
|--------|----------|--------|--------|--------|---------|--------|
| Sa2VA-8B | VT | 15.82 | 1.61 | 13.15 | 22.59 | 18.78 |
| Qwen2.5-Omni-7B | AVT | 12.82 | 1.48 | 12.04 | 19.56 | 20.03 |
| **Ours** | **AVT** | **35.92** | **15.32** | **35.86** | **39.82** | **41.04** |

**Improvement in assisted depression detection** (using emotion-cognition profiles generated by different methods):

| Profile Source | Avg. ACC Gain | Avg. F1 Gain |
|----------------|--------------|-------------|
| InternVL-2.5-8B | +9.48% | +3.56% |
| Sa2VA-8B | +10.40% | +5.30% |
| EmoLLM | +6.42% | +5.39% |
| **Ours** | **+12.54%** | **+9.16%** |

### Ablation Study

| Modality | EmoCL | CogCL | F_BERT |
|----------|-------|-------|--------|
| Audio | ✓ | ✓ | 18.10 |
| Video | ✓ | ✓ | 16.30 |
| Audio+Text | ✓ | ✓ | 22.05 |
| Audio+Video+Text | ✗ | ✗ | 23.89 |
| Audio+Video+Text | ✓ | ✗ | 27.24 |
| Audio+Video+Text | ✗ | ✓ | 26.94 |
| **Audio+Video+Text** | **✓** | **✓** | **34.09** |

### Key Findings

1. **Multimodal fusion is critical**: Audio contributes the most, and the three-modality combination achieves the best performance (34.09 vs. 18.10 for the best single modality).
2. **Both streams are indispensable**: Removing either contrastive learning branch leads to significant performance degradation.
3. **Emotion-cognition profiles substantially improve assisted diagnosis**: Depression detection ACC increases by 12.54% and F1 by 9.16%.
4. **Low-quality captions may degrade detection performance**: Verbose but low-information text increases the difficulty for LLMs to extract relevant clinical indicators.
5. **Distinct emotion-cognition patterns exist between depressed and anxious patients**: Depression is associated with higher frequencies of cognitive impairment.

## Highlights & Insights

1. **Novel task formulation**: Reframing mental health analysis from classification to captioning establishes a new research paradigm.
2. **Elegant dual-stream BridgeNet design**: Disentangled extraction of emotion and cognition, with three-class contrastive learning for affect and multi-label Jaccard contrastive learning for cognition, both grounded in domain-specific characteristics.
3. **Clinically grounded design**: Cognitive impairment categories are defined with reference to the MMSE scale, conferring direct clinical relevance.
4. **Comprehensive evaluation**: Both objective metrics (BLEU, ROUGE, etc.) and subjective assessment (expert psychologist ratings) are employed, strengthening the validity of the findings.
5. **End-to-end applicability**: The complete pipeline from multimodal input to assisted diagnosis demonstrates practical deployment potential.

## Limitations & Future Work

1. Annotation of emotion and cognition captions relies on LLM-generated labels with human correction, which may introduce biases.
2. Evaluation is conducted on a single dataset (MMDA), limiting generalizability.
3. Cognitive accuracy (CAcc) receives relatively low scores in human evaluation (3.9), indicating room for improvement in multimodal cognitive modeling.
4. The total parameter count of 7.6B entails high deployment costs; lighter-weight alternatives warrant exploration.
5. Class imbalance between normal and abnormal samples in the dataset may affect model learning.

## Related Work & Insights

- **BLIP-2** (Li et al., 2023): The architectural inspiration for BridgeNet; a successful application of the Q-Former design.
- **SECap** (Xu et al., 2024): Speech emotion captioning task; this work extends the paradigm to multimodal inputs and the cognitive dimension.
- **Emotion-LLaMA** (Cheng et al., 2024): Used to generate initial emotion annotations; a key component in the data pipeline.
- Insight: Mental health AI should move beyond binary "normal/abnormal" classification toward generating human-interpretable analytical reports; the contribution of audio signals in multimodal fusion may be systematically underestimated.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (First to propose the ECMC task, shifting mental health analysis from classification to captioning)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Comprehensive objective and subjective evaluation, but limited to a single dataset)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure with well-articulated motivation)
- Value: ⭐⭐⭐⭐⭐ (Significant contribution to mental health AI with direct clinical applicability)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Learning Cell-Aware Hierarchical Multi-Modal Representations for Robust Molecular Modeling](learning_cell-aware_hierarchical_multi-modal_representations.md)
- [\[ACL 2026\] Responsible Evaluation of AI for Mental Health](../../ACL2026/medical_nlp/responsible_evaluation_of_ai_for_mental_health.md)
- [\[ACL 2026\] MHGraphBench: Knowledge Graph-Grounded Benchmarking of Mental Health Knowledge in Large Language Models](../../ACL2026/medical_nlp/mhgraphbench_knowledge_graph-grounded_benchmarking_of_mental_health_knowledge_in.md)
- [\[ACL 2026\] Measuring What Matters!! Assessing Therapeutic Principles in Mental-Health Conversation](../../ACL2026/medical_nlp/measuring_what_matters_assessing_therapeutic_principles_in_mental-health_convers.md)
- [\[ICLR 2026\] CounselBench: A Large-Scale Expert Evaluation and Adversarial Benchmarking of LLMs in Mental Health QA](../../ICLR2026/medical_nlp/counselbench_llm_mental_health_qa.md)

</div>

<!-- RELATED:END -->
