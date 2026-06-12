---
title: >-
  [Paper Note] Concept-RuleNet: Grounded Multi-Agent Neurosymbolic Reasoning in Vision Language Models
description: >-
  [AAAI 2026][Multimodal VLM][Neurosymbolic Reasoning] This paper proposes Concept-RuleNet, a three-agent collaborative neurosymbolic reasoning framework that conditions symbol generation and rule construction on visual co…
tags:
  - "AAAI 2026"
  - "Multimodal VLM"
  - "Neurosymbolic Reasoning"
  - "Multi-Agent Systems"
  - "Visual Concept Grounding"
  - "Explainable AI"
  - "Counterfactual Reasoning"
date: 2026-05-08
content_hash: a1aa03e9883d315c
---

# Concept-RuleNet: Grounded Multi-Agent Neurosymbolic Reasoning in Vision Language Models

**Conference**: AAAI 2026
**arXiv**: [2511.11751](https://arxiv.org/abs/2511.11751)  
**Code**: [https://github.com/sanchit97/Concept-RuleNet](https://github.com/sanchit97/Concept-RuleNet)  
**Area**: Multimodal VLM / Agent / Neurosymbolic Reasoning
**Keywords**: Neurosymbolic Reasoning, Multi-Agent Systems, Visual Concept Grounding, Explainable AI, Counterfactual Reasoning

## TL;DR

This paper proposes Concept-RuleNet, a three-agent collaborative neurosymbolic reasoning framework that conditions symbol generation and rule construction on visual concepts extracted from training images. It addresses the symbol hallucination and non-representativeness issues of existing methods (e.g., Symbol-LLM) that rely solely on class labels, achieving an average accuracy improvement of ~5% across 5 OOD benchmarks and reducing hallucinated symbols by up to 50%.

## Background & Motivation

1. **System-1 vs. System-2 Reasoning**: Modern VLMs operate as System-1 reasoners (fast but not interpretable), whereas human cognition also relies on System-2 (slow, logical, and interpretable). Recent work attempts to combine both by augmenting black-box VLM predictions with neurosymbolic rules.
2. **Fatal Flaws of Existing Neurosymbolic Methods**: Methods such as Symbol-LLM condition LLMs solely on task labels to generate symbols and rules, making no use of training image information. This leads to two critical problems:
    - **Insufficient Grounding**: LLMs tend to hallucinate symbols in OOD domains—generating symbols that never appear in the dataset. For example, for a blood cell classification task, a symbol such as "presence of allergic response" may be generated despite being completely absent from the images.
    - **Insufficient Representativeness**: Due to data leakage from LLM pretraining, performance is satisfactory only on common datasets (e.g., HICO/Stanford derived from COCO), while symbols generated for under-represented domains (medical imaging, remote sensing) are irrelevant to the task. For instance, "presence of irrigation systems" may be generated for "agricultural land classification," a concept unverifiable in the images.

## Core Problem

How can visual grounding be incorporated into neurosymbolic reasoning so that automatically generated symbols and rules are faithful to the data distribution and representative of the task, thereby effectively augmenting VLM predictions in OOD and under-represented domains?

## Method

### Overall Architecture

Concept-RuleNet is a three-stage, three-agent collaborative system:

1. **Stage 1 – Visual Concept Extraction Agent ($\mathcal{A}_V$)**: A VLM extracts grounded visual concepts from training images.
2. **Stage 2 – Symbol Exploration and Rule Construction Agent ($\mathcal{A}_L$)**: An LLM generates symbols conditioned on visual concepts and combines them into first-order logic rules.
3. **Stage 3 – Verification Agent ($\mathcal{A}_V$ as verifier)**: At inference time, a VLM quantifies the presence of each symbol, executes the rules, and combines the result with the System-1 prediction via weighted fusion.

Final prediction: $\hat{y} = (1-\lambda) F_{sys1}(x) + \lambda F_{sys2}(x)$

### Key Designs

**1. Image-Conditioned Visual Concept Extraction**
- Low-level visual concepts (color, texture, morphology, etc.) are extracted from a subset of training images per class.
- A VLM (LLaVA-Med for medical data; LLaVA-1.6 for natural images) serves as a "Bag-of-visual-attributes" extractor.
- Prompt format: "In this picture, we see {label}. List {N} visual concepts..."
- 200 images per class for medical data, 50 for remote sensing, up to 100 for iNaturalist.

**2. Context-Dependent Symbol Exploration**
- An initialization function $\mathcal{IS}(y, K)$ generates $K=5$ initial premise symbols.
- An exploration function $\mathcal{ES}(c_y, y)$ is then applied iteratively, expanding the symbol set conditioned on visual concepts.
- Key insight: visual concepts provide a grounded context for the LLM, reducing hallucination.
- MedMNIST uses 10 iterations; other datasets use 7.

**3. Rule Formation**
- Explored symbols are combined into Disjunctive Normal Form (DNF) rules: $l_i = \bigwedge s_i \rightarrow y$
- An LLM computes entailment scores with threshold $\epsilon > 0.7$.
- Rule length is capped at 3 symbols (experiments show diminishing returns for longer rules with exponentially increasing computation cost).
- Entailment prompt: "We know {concepts} is responsible for {y}. Given {rule}, how likely is {y}?"

**4. Symbol Verification at Inference**
- For each symbol in a rule, a VLM binary QA query obtains the presence probability.
- Prompt: "In the image we see a {task}. Does this image show {symbol}?"
- Symbol score = probability of "yes" under softmax(logit["yes"], logit["no"]).
- Rule score = min over symbol scores (conjunction); max across rules (disjunction).

**5. Concept-RuleNet++ Extension**
- Counterfactual symbols are introduced: symbols are selected from rules of other classes, and their "absence" probability serves as evidence.
- Rules are extended from pure DNF to a mixed DNF+CNF form.
- Formal expression: $l = \{\bigwedge\{\bigvee\{\tilde{s}_i, s_i\}\} \rightarrow y\}$

### Loss & Training

- **No training required**: The entire system operates in a zero-shot setting; the System-1 model is not fine-tuned.
- The hyperparameter $\lambda$ controls the System-2 weight: 0.5 for BloodMNIST/DermaMNIST, 0.7 for Satellite/iNaturalist, and 0.5 for WHU.
- Symbol generation uses GPT-4o-mini with temperature 0.7 during exploration and 0 during entailment.
- Visual concept extraction uses temperature 0.2.

## Key Experimental Results

**Datasets**: 5 datasets (BloodMNIST, DermaMNIST, UCMerced-Satellite, WHU, iNaturalist-21), all from OOD or under-represented domains.

**Main Results (Table 1, same System-1 model used as Verifier)**:

| Dataset | Model | S1 | Symbol-LLM | CRN |
|--------|------|-----|------------|-----|
| BloodMNIST | InstructBLIP | 11.55 | 13.56 | **18.09** |
| BloodMNIST | LLaVA-1.5 | 11.55 | 10.05 | **14.57** |
| BloodMNIST | LLaVA-1.6 | 10.05 | 9.67 | **19.35** |
| DermaMNIST | InstructBLIP | 5.05 | 5.05 | **8.54** |
| DermaMNIST | LLaVA-1.5 | 9.54 | 30.15 | **47.73** |
| DermaMNIST | LLaVA-1.6 | 36.68 | 38.19 | **48.74** |
| Satellite | InstructBLIP | 41.33 | 48.00 | **57.33** |
| Satellite | LLaVA-1.5 | 65.33 | 64.00 | **69.33** |
| iNaturalist | InstructBLIP | 52.13 | 52.65 | **53.21** |
| iNaturalist | LLaVA-1.6 | 61.30 | 61.30 | **63.45** |

- CRN outperforms Symbol-LLM by ~**5%** on average on BloodMNIST/DermaMNIST.
- Maximum gain of **9.33%** on UCMerced (InstructBLIP).
- Average gain of 2–4% on WHU.

**Concept-RuleNet++ (Table 3, InstructBLIP)**:

| Dataset | CRN | CRN++ |
|--------|-----|-------|
| BloodMNIST | 18.09 | **21.43** |
| DermaMNIST | 8.54 | **14.23** |
| Satellite | 57.33 | **58.12** |
| WHU | 20.40 | **21.52** |
| iNaturalist | 53.21 | **54.15** |

CRN++ yields an additional average improvement of 1–2%.

**Symbol Grounding Metric**: The average presence probability of CRN-generated symbols in both training and test sets is significantly higher than that of Symbol-LLM. On Satellite and WHU, Symbol-LLM symbol occurrence rates fall below 0.5 (indicating widespread hallucination), while CRN shows marked improvement.

**Representativeness Metric**: GPT-o1 evaluation of symbol representativeness yields an average score of 0.54 for CRN vs. 0.49 for Symbol-LLM.

**Statistical Tests (Table 6)**: CRN vs. Symbol-LLM shows an average improvement of 4.99 pp ($p=0.019$); CRN vs. S1 shows 6.83 pp ($p=0.048$), both statistically significant.

### Ablation Study

**Effect of Visual Context at Each Stage (Table 4, UCMerced/InstructBLIP)**:

| Init | Explore | Entailment | Accuracy |
|--------|------|------|--------|
| ✗ | ✗ | ✗ | 48.00 (= Symbol-LLM) |
| ✔ | ✗ | ✗ | 49.50 |
| ✔ | ✔ | ✗ | 55.10 |
| ✔ | ✔ | ✔ | **57.33** |

Visual concepts contribute gains at every stage, with the exploration stage yielding the largest improvement (+5.6%).

**Sensitivity to $\lambda$ and Image Count (Table 5)**:
- Performance degrades when $\lambda$ is too high or too low.
- Using too many images can introduce irrelevant concepts, leading to overfitting.
- The optimal image count varies by dataset.

**Rule Length**: Accuracy gains diminish for rule lengths exceeding 3, while API call costs grow exponentially.

## Highlights & Insights

1. **Addresses a genuine and overlooked problem** in neurosymbolic reasoning: symbols generated solely from labels have poor grounding, which is especially severe in OOD scenarios.
2. **Elegant and concise method design**: the three agents have clearly defined roles (extract → generate → verify), and each module can be independently replaced.
3. **No training required**: the system operates entirely zero-shot; rules can be precomputed offline, and inference only requires VLM Yes/No queries.
4. **The counterfactual extension in CRN++** is conceptually novel, strengthening decision boundaries by incorporating "symbols that should be absent."
5. **Experiments are conducted on genuinely challenging OOD domains** (medical imaging, remote sensing, biological species) rather than on standard benchmarks.

## Limitations & Future Work

1. **Low absolute accuracy**: BloodMNIST peaks at ~21% (8 classes) and DermaMNIST at ~48% (7 classes)—while these datasets are inherently difficult in zero-shot settings, the gains from System-2 augmentation remain limited.
2. **Heavy reliance on the VLM's binary QA capability**: the verification stage reduces complex visual judgments to Yes/No queries, entailing significant information loss; if the VLM is weak at fine-grained concept judgment, the entire pipeline is constrained.
3. **Symbol and rule quality is bounded by GPT-4o-mini**: stronger reasoning models (e.g., GPT-o1) may yield better symbol exploration at higher cost.
4. **No comparison with fine-tuned methods**: although the paper argues that zero-shot generalization is more practical, not even a few-shot linear probing baseline is provided.
5. **Rule length is fixed at 3**: while ablations support this choice, different tasks may require rules of varying complexity.
6. **$\lambda$ requires validation set tuning**: the optimal $\lambda$ varies across datasets (0.5–0.7), which introduces an additional deployment burden.
7. **Scalability is not discussed**: when the number of classes is large (e.g., ImageNet-1K), what are the API costs for symbol exploration and rule generation?

## Related Work & Insights

| Method | Symbol Source | Uses Image Info | OOD Generalization | Interpretability |
|------|---------|---------------|---------|---------|
| Symbol-LLM (NeurIPS 2024) | LLM conditioned on labels only | ✗ | Poor (strong on HICO/Stanford due to data leakage) | ✔ |
| Concept-RuleNet | LLM + VLM concept conditioning | ✔ | Good | ✔ |
| Concept Bottleneck Models | Predefined concept sets | Partial | Limited by concept definitions | ✔ |
| Neural-Symbolic VQA | Manually designed classifiers | ✔ | Poor (only 4 object categories) | ✔ |

Core advantage: Symbol-LLM performs well on HICO/Stanford because these datasets derive from COCO—a high-frequency distribution in LLM pretraining data. Once shifted to OOD domains (medical/remote sensing), Symbol-LLM's symbol quality degrades sharply. CRN addresses this fundamental issue through visual grounding.

**Broader Implications**:
1. **Visual grounding is key to improving LLM-as-Agent quality**: not only in neurosymbolic reasoning, but in any scenario where LLMs generate visually grounded knowledge, conditioning on actual image information should be considered.
2. **An information-theoretic perspective on grounding**: the paper notes $H(S|x) < H(S)$, i.e., image conditioning reduces symbol uncertainty—a useful theoretical framework.
3. **Potential for integration with Concept Bottleneck Models**: CRN's visual concept extraction could facilitate better concept discovery for CBMs.
4. **Extensibility to video understanding**: temporal logic rules combined with visual concept grounding may be more valuable in action recognition settings.
5. **Counterfactual reasoning potential**: the CRN++ approach can be generalized to other tasks requiring fine-grained discrimination.

## Rating (⭐ 1–5)

⭐⭐⭐⭐ (4/5)

**Strengths**: The problem is clearly defined and important; the method design is elegant; evaluation is conducted on genuinely challenging OOD domains rather than easy benchmarks. The modular three-agent architecture is easy to understand and extend. Statistical testing is thorough.

**Deductions**: Low absolute accuracy limits practical applicability; the absence of fine-tuned baselines weakens the justification for the zero-shot framing; symbol quality evaluation is somewhat qualitative (relying on average presence probabilities and GPT-o1 scores), lacking more rigorous quantitative metrics such as a precise definition and measurement of hallucination rate.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Vision-Language Models Encode Clinical Guidelines for Concept-Based Medical Reasoning](../../CVPR2026/multimodal_vlm/vision-language_models_encode_clinical_guidelines_for_concept-based_medical_reas.md)
- [\[AAAI 2026\] RMAdapter: Reconstruction-based Multi-Modal Adapter for Vision-Language Models (Oral)](rmadapter_reconstructionbased_multimodal_adapter_for_visionlanguage.md)
- [\[AAAI 2026\] Multi-Agent VLMs Guided Self-Training with PNU Loss for Low-Resource Offensive Content Detection](multi-agent_vlms_guided_self-training_with_pnu_loss_for_low-resource_offensive_c.md)
- [\[ACL 2026\] OMHBench: Benchmarking Balanced and Grounded Omni-Modal Multi-Hop Reasoning](../../ACL2026/multimodal_vlm/omhbench_benchmarking_balanced_and_grounded_omni-modal_multi-hop_reasoning.md)
- [\[ACL 2026\] VIGNETTE: Socially Grounded Bias Evaluation for Vision-Language Models](../../ACL2026/multimodal_vlm/vignette_socially_grounded_bias_evaluation_for_vision-language_models.md)

</div>

<!-- RELATED:END -->
