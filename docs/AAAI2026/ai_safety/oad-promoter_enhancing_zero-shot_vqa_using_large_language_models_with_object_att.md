---
title: >-
  [Paper Note] OAD-Promoter: Enhancing Zero-shot VQA using Large Language Models with Object Attribute Description
description: >-
  [AAAI 2026][AI Safety][Visual Question Answering] This paper proposes OAD-Promoter, which cooperatively utilizes three modules—Object-concentrated Example Generation (OEG), Memory Knowledge Assistance (MKA), and OAD Prompt—to mitigate language bias inherited by LLMs and enhance domain adaptation capabilities under the zero-shot setting, achieving SOTA performance on multiple benchmarks such as VQAv2.
tags:
  - "AAAI 2026"
  - "AI Safety"
  - "Visual Question Answering"
  - "Zero-shot VQA"
  - "Language Bias"
  - "Domain Adaptation"
  - "Object Attribute Description"
date: 2026-05-08
content_hash: 520637379a918165
---

# OAD-Promoter: Enhancing Zero-shot VQA using Large Language Models with Object Attribute Description

**Conference**: AAAI 2026  
**arXiv**: [2511.12131](https://arxiv.org/abs/2511.12131)  
**Code**: None  
**Area**: Information Retrieval  
**Keywords**: Visual Question Answering, Zero-shot VQA, Language Bias, Domain Adaptation, Object Attribute Description

## TL;DR
This paper proposes OAD-Promoter, which cooperatively utilizes three modules—Object-concentrated Example Generation (OEG), Memory Knowledge Assistance (MKA), and OAD Prompt—to mitigate language bias inherited by LLMs and enhance domain adaptation capabilities under the zero-shot setting, achieving SOTA performance on multiple benchmarks such as VQAv2.

## Background & Motivation

**Background**: LLMs have become key tools for processing knowledge-intensive questions in VQA tasks. Existing LLM-based KBVQA methods (such as PICa, Prophet, Img2LLM, etc.) have achieved remarkable results in few-shot and zero-shot scenarios.

**Limitations of Prior Work—Language Bias**: Language bias is a persistent issue in the VQA domain. For example, in the training data, the dominant answer to "What color... bananas?" is "yellow," causing the model to exploit this superficial correlation instead of truly understanding the image. This problem is not only present in traditional VQA models but is **equally severe in LLM-based methods**, as LLMs inevitably learn shortcut correlations (shortcut learning) during pre-training on large-scale datasets.

**Two Major Negative Impacts** (Figure 1):

**Unreliable Predictions**: LLMs use inherited language bias for reasoning, leading to biased answers.

**Poor OOD Generalization**: Although LLMs possess strong knowledge reasoning capabilities, language bias exacerbates the difficulties of domain adaptation.

**Blind Spots of Existing Methods**:
- Existing LLM-based KBVQA methods neglect the integration of global and regional visual information.
- There is no auxiliary memory module to help LLMs handle scenarios with distribution shifts.
- Directly integrating debiasing methods (e.g., LMH, CSS) into the LLM pipeline degrades performance (as verified by experiments).

**Mechanism**:
1. Finer-grained visual information can mitigate language bias (allowing LLMs to "see" more, reducing dependence on language priors).
2. Assistance from memory examples can enhance reasoning reliability, especially in domain adaptation scenarios.
3. Prompts integrating the above two points can continuously enhance domain adaptation capabilities.

## Method

### Overall Architecture
OAD-Promoter contains three synergistic modules (Figure 2):
1. **OEG Module** (green box): Generates global captions and object-concentrated examples.
2. **MKA Module** (blue box): Uses stored examples to assist the LLM in processing new inputs.
3. **OAD Prompt** (red box): Integrates outputs of the first two modules to guide LLM reasoning.

The entire pipeline does not rely on any external knowledge sources or data that needs retrieval, making it a purely zero-shot method.

### Key Designs

#### 1. **OEG Module (Object-concentrated Example Generation)**

Contains two generation processes:

**Multi-level Caption Generation**:
- A pre-trained BLIP2 is used to generate a **global caption**, capturing the overall semantics of the image.
- A VinVL detector is used to generate **object-concentrated captions** (regional captions), focusing on individual object attributes.

**Synthetic Question Generation**:
- A caption parser is used to extract potential answers (noun phrases, verb phrases, adjectives, numbers, boolean words) from object captions.
- A T5-large model fine-tuned on SQuAD2.0, MultiRC, BookQA, CommonsenseQA, and Social IQa is used to generate corresponding questions.
- A complete object-concentrated example $E_i = (C, Q, A)$ is formed.

Design Motivation: The global caption provides macro-level understanding, while the regional captions supplement fine-grained information. Combining both grants the LLM more complete visual information, reducing the likelihood of relying on language priors. These generated examples serve as both the memory bank for MKA and components of the Prompt.

#### 2. **MKA Module (Memory Knowledge Assistance)**

Contains two processes:

**Answer Estimation**:
- **Vanilla VQA Model** (UpDn): Outputs a vanilla answer $A_O$ (containing visual information).
- **Biased QA Model** (the off-shift model in LMH): Outputs a biased answer $A_B$ (lacking visual information, pure language bias).

Selection Mode Decision:
$$M = \begin{cases} Positive, & \text{if } A_O \neq A_B \\ Negative, & \text{if } A_O = A_B \end{cases}$$

Key Insight: If $A_B = A_O$, it indicates that the same answer can be obtained even without looking at the image—implying that the vanilla model is utilizing language bias. Since the training scale of LLMs is far larger than that of vanilla VQA models, the probability of LLMs exploiting this bias is even higher.

**Similarity Calculation**:
$$E_S = \begin{cases} \text{argTopN} \frac{f^T f_j}{\|f\|_2 \|f_j\|_2}, & \text{if } M = Positive \\ \text{argBottomN} \frac{f^T f_j}{\|f\|_2 \|f_j\|_2}, & \text{if } M = Negative \end{cases}$$

- Positive Mode: Selects the most similar stored examples (supporting normal reasoning).
- **Negative Mode**: Selects the **least similar** stored examples (countering language bias).

Design Motivation: By pre-registering bias signals and actively choosing auxiliary examples opposite to the bias direction, the LLM is guided to bypass language bias. As inference proceeds, the memory bank continuously grows, and the domain adaptation capability is steadily enhanced.

#### 3. **OAD Prompt**

A structured prompt that integrates outputs from the first two modules:
$$[\text{Instruction } I \;/\; \text{Global Caption } C_G \;/\; \text{Object Examples } E_O \;/\; \text{Memory Examples } E_S \;/\; \text{Question } Q_O]$$

Initially, the MKA memory is empty, and the prompt is $[I / C_G / E_O / Q_O]$. From the second sample onward, it transitions to the full form.

Key Difference from Existing Methods: It simultaneously considers both global descriptions and object attribute descriptions, rather than relying solely on global captions. Ablation studies demonstrate that CQA-CQA-CQA (keeping the complete triple for each example) outperforms CCC-QAQAQA (separated arrangement).

### Loss & Training
- OAD-Promoter itself is not trained; it is an inference-time framework.
- The UpDn model is first pre-trained on VQAv2+Visual Genome, and then fine-tuned on the OKVQA training set.
- Main experiments use GPT-3 and OPT as frozen LLMs.
- Avoiding Data Contamination: Images that appear in the OKVQA test set are removed from the pre-training data.

## Key Experimental Results

### Main Results—Performance under Zero-shot Settings

| Method | VQAv2 test | A-OKVQA test | OKVQA test |
|------|-----------|-------------|------------|
| Flamingo-80B | 56.21 | - | 50.57 |
| Img2LLM w/ GPT-3 | 59.22 | 43.39 | 42.80 |
| Img2LLM w/ OPT | 61.83 | 40.69 | 45.58 |
| **OAD-Promoter w/ OPT** | **61.93** | **40.68** | **45.58** |
| **OAD-Promoter w/ GPT-3** | **61.98** | **41.71** | **45.61** |

### Generalization Verification on Different LLMs (OKVQA Zero-shot)

| LLM | Parameters | OKVQA |
|-----|--------|-------|
| GPT-Neo | 2.7B | 33.41 |
| GPT-J | 6B | 38.89 |
| BLOOM | 7.1B | 33.77 |
| OPT | 6.7B | 36.18 |
| OPT | 30B | 40.46 |
| OPT | 175B | 45.58 |
| GPT-3 | 175B | 45.61 |

### Ablation Study

| Configuration | OKVQA (Few-shot) | OKVQA (Zero-shot) | Note |
|------|-----------------|-------------------|------|
| No OEG + No MKA | 47.33 | 42.50 | Baseline |
| With OEG + No MKA | 54.68 | 44.26 | OEG contributes the most |
| No OEG + With MKA | 48.95 | 43.64 | MKA is also helpful independently |
| **With OEG + With MKA** | **60.04** | **45.61** | Synergy of both yields the best results |

| Number of MKA Memory Examples K | OKVQA | Note |
|---------------|-------|------|
| 0 | 43.64 | No memory |
| 60 | 43.65 | Few examples |
| 200 | 43.92 | Medium examples |
| 400 | 44.15 | More examples yield better performance |

### Domain Adaptation Experiments (Few-shot, Different LLMs)

| LLM | VQA-CP | GQA-OOD |
|-----|--------|---------|
| GPT-4 (GRACE) | 57.61 | 50.19 |
| **GPT-4 (OAD-Promoter)** | 55.93 | **50.21** |

### Key Findings
1. A new SOTA (61.98) is achieved under the VQAv2 zero-shot setting, outperforming all large-scale multimodal pre-training and frozen LLM methods.
2. The OEG module contributes the most (+7.35) under the few-shot setting, proving that fine-grained visual information is key to mitigating language bias.
3. Directly integrating traditional debiasing methods (LMH, CSS) into the LLM pipeline degrades OKVQA performance (Table 4), indicating that language bias in LLMs requires different handling strategies.
4. With GPT-4, OAD-Promoter achieves the best performance on GQA-OOD (50.21), showing that stronger LLMs can better leverage domain adaptation capabilities.
5. The mechanism where the memory bank grows during inference leads to continuous performance improvements (K=400 > K=200 > K=60).
6. Changing the input order has no impact on OAD-Promoter (100% robustness), whereas Img2LLM is sensitive to order.

## Highlights & Insights
1. **Reveals the Severity of Language Bias in LLMs**: Not only do traditional VQA models suffer from bias issues, but LLM-based methods do too, and traditional debiasing methods prove ineffective.
2. **Innovative Bias Countering Strategy**: Actively choosing the least similar examples via Negative Mode to counter bias is a novel and effective idea.
3. **Zero-shot Outperforming Few-shot**: Thanks to the memory growth mechanism of MKA, the reasoning capability under the zero-shot setting is continuously strengthened.
4. **Plug-and-play**: The framework can be combined with various LLMs (e.g., GPT-3, OPT, BLOOM, GPT-Neo, GPT-J, GPT-4).
5. **Self-growing Memory Bank**: The MKA module naturally accumulates knowledge as inference proceeds, representing an elegant form of continual learning.

## Limitations & Future Work
1. Dependency on the quality of pre-trained models like VinVL and BLIP2; failure of these models degrades the output of the OEG module.
2. Bias detection in the MKA module relies on the QA modules of UpDn and LMH, which have limited capabilities themselves.
3. The performance gain under the zero-shot setting on A-OKVQA and OKVQA is limited (<1%), showing the ceiling of this method on more difficult knowledge-reasoning problems.
4. Infinite growth of the memory bank may pose storage and retrieval efficiency challenges.
5. The selection mode between Positive and Negative is a hard switch; soft interpolation could be explored.

## Related Work & Insights
- The mechanism of bias detection (comparing predictions with vs. without visual cues) can be extended to other scenarios requiring shortcut learning detection.
- The memory-augmented reasoning framework serves as a valuable reference for other reasoning tasks that require continuous improvement.
- The integration strategy of multi-granularity visual information (global + regional) is worth borrowing for other VLM works.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Hierarchically Robust Zero-shot Vision-language Models](../../CVPR2026/ai_safety/hierarchically_robust_zero-shot_vision-language_models.md)
- [\[ICML 2026\] Calibrating Uncertainty for Zero-Shot Adversarial CLIP](../../ICML2026/ai_safety/calibrating_uncertainty_for_zero-shot_adversarial_clip.md)
- [\[CVPR 2026\] Towards Robust Multimodal Large Language Models Against Jailbreak Attacks](../../CVPR2026/ai_safety/towards_robust_multimodal_large_language_models_against_jailbreak_attacks.md)
- [\[CVPR 2026\] SIF: Semantically In-Distribution Fingerprints for Large Vision-Language Models](../../CVPR2026/ai_safety/sif_semantically_in-distribution_fingerprints_for_large_vision-language_models.md)
- [\[CVPR 2026\] GenBreak: Red Teaming Text-to-Image Generation Using Large Language Models](../../CVPR2026/ai_safety/genbreak_red_teaming_text-to-image_generation_using_large_language_models.md)

</div>

<!-- RELATED:END -->
