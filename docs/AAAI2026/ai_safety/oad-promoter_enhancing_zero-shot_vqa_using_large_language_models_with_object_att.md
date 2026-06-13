---
title: >-
  [Paper Note] OAD-Promoter: Enhancing Zero-shot VQA using Large Language Models with Object Attribute Description
description: >-
  [AAAI 2026][AI Safety][Visual Question Answering] This paper proposes OAD-Promoter, a framework comprising three collaborative modules—Object-concentrated Example Generation (OEG), Memory Knowledge Assistance (MKA)…
tags:
  - "AAAI 2026"
  - "AI Safety"
  - "Visual Question Answering"
  - "Zero-shot VQA"
  - "Language Bias"
  - "Domain Transfer"
  - "Object Attribute Description"
date: 2026-05-08
content_hash: e2f86866c420f267
---

# OAD-Promoter: Enhancing Zero-shot VQA using Large Language Models with Object Attribute Description

**Conference**: AAAI 2026
**arXiv**: [2511.12131](https://arxiv.org/abs/2511.12131)  
**Code**: None  
**Area**: Information Retrieval
**Keywords**: Visual Question Answering, Zero-shot VQA, Language Bias, Domain Transfer, Object Attribute Description

## TL;DR
This paper proposes OAD-Promoter, a framework comprising three collaborative modules—Object-concentrated Example Generation (OEG), Memory Knowledge Assistance (MKA), and OAD Prompt—to mitigate language bias inherited by LLMs and improve domain transfer under zero-shot settings, achieving state-of-the-art performance on VQAv2 and multiple other benchmarks.

## Background & Motivation

**Background**: LLMs have become a critical tool for handling knowledge-intensive questions in VQA tasks. Existing LLM-based KBVQA methods (PICa, Prophet, Img2LLM, etc.) have achieved notable results in both few-shot and zero-shot settings.

**Core Limitation — Language Bias**: Language bias is a persistent problem in VQA. For instance, the dominant answer to "What color...bananas?" in training data is "yellow," and models tend to exploit such superficial correlations rather than genuinely understanding the image. This issue exists not only in conventional VQA models but is **equally severe in LLM-based approaches**, as LLMs inevitably learn spurious correlations (shortcut learning) during large-scale pretraining.

**Two Major Negative Effects** (Figure 1):

**Unreliable Predictions**: LLMs exploit inherited language biases during inference, leading to biased answers.

**Poor OOD Generalization**: Despite strong knowledge reasoning capabilities, language bias exacerbates difficulties in domain transfer.

**Blind Spots of Existing Methods**:
- Existing LLM-based KBVQA methods overlook the integration of global and regional visual information.
- No auxiliary memory module exists to help LLMs handle distribution-shifted scenarios.
- Debiasing methods (e.g., LMH, CSS) directly integrated into the LLM pipeline actually degrade performance (verified experimentally).

**Core Mechanism**:
1. Richer visual information can alleviate language bias by enabling LLMs to "see" more, thereby reducing reliance on linguistic priors.
2. Memory-augmented examples improve inference reliability, particularly in domain transfer scenarios.
3. A prompt that integrates the above two components continuously enhances domain adaptation.

## Method

### Overall Architecture
OAD-Promoter consists of three collaborative modules (Figure 2):
1. **OEG Module** (green box): Generates global captions and object-concentrated examples.
2. **MKA Module** (blue box): Leverages stored examples to assist LLMs in processing new inputs.
3. **OAD Prompt** (red box): Integrates outputs from the preceding two modules to guide LLM reasoning.

The entire pipeline relies on no external knowledge sources or retrieval corpora, constituting a purely zero-shot approach.

### Key Designs

#### 1. **OEG Module (Object-concentrated Example Generation)**

Contains two generation processes:

**Multi-level Caption Generation**:
- A pretrained BLIP2 generates **global captions** ($C_G$) to capture overall image semantics.
- A VinVL detector generates **region-level captions** (object-concentrated captions) focusing on individual object attributes.

**Synthetic Question Generation**:
- A caption evaluation tool extracts candidate answers from object captions (noun phrases, verb phrases, adjectives, numbers, Boolean words).
- A T5-large model fine-tuned on SQuAD2.0, MultiRC, BookQA, CommonsenseQA, and Social IQa generates corresponding questions.
- Complete object-concentrated examples are formed as $E_i = (C, Q, A)$.

**Design Motivation**: Global captions provide macro-level understanding, while region-level captions supply fine-grained information. Their combination gives LLMs more complete visual information, reducing opportunities to rely on linguistic priors. These generated examples serve simultaneously as the MKA memory bank and as components of the prompt.

#### 2. **MKA Module (Memory Knowledge Assistance)**

Contains two processes:

**Answer Estimation**:
- **Standard VQA model** (UpDn): Outputs a vision-aware answer $A_O$.
- **Bias-only QA model** (off-shift model from LMH): Outputs a biased answer $A_B$ without visual input (pure language bias).

Mode determination:
$$M = \begin{cases} Positive, & \text{if } A_O \neq A_B \\ Negative, & \text{if } A_O = A_B \end{cases}$$

**Key Insight**: If $A_B = A_O$, the same answer is obtainable without viewing the image, suggesting the standard model is exploiting language bias. Since LLMs are trained on far larger corpora than standard VQA models, they are even more prone to such bias.

**Similarity Computation**:
$$E_S = \begin{cases} \text{argTopN} \frac{f^T f_j}{\|f\|_2 \|f_j\|_2}, & \text{if } M = Positive \\ \text{argBottomN} \frac{f^T f_j}{\|f\|_2 \|f_j\|_2}, & \text{if } M = Negative \end{cases}$$

- Positive mode: Selects the most similar stored examples (supporting normal reasoning).
- **Negative mode**: Selects the **least similar** stored examples (counteracting language bias).

**Design Motivation**: By anticipating bias signals, the module proactively selects auxiliary examples in the direction opposing the bias, thereby guiding LLMs away from language shortcuts. As inference proceeds, the memory bank grows continuously, progressively enhancing domain adaptation.

#### 3. **OAD Prompt**

A structured prompt integrating outputs from the preceding two modules:
$$[\text{Instruction } I \;/\; \text{Global Caption } C_G \;/\; \text{Object Examples } E_O \;/\; \text{Memory Examples } E_S \;/\; \text{Question } Q_O]$$

At initialization, the MKA memory is empty and the prompt reduces to $[I / C_G / E_O / Q_O]$; from the second sample onward, the full form is used.

**Key Distinction from Prior Methods**: The prompt simultaneously incorporates global descriptions and object attribute descriptions, rather than relying solely on global captions. Ablation experiments confirm that the CQA-CQA-CQA arrangement (each example as a complete triple) outperforms the CCC-QAQAQA arrangement (separated layout).

### Loss & Training
- OAD-Promoter itself requires no training; it operates as an inference-time framework.
- The UpDn model is first pretrained on VQAv2 + Visual Genome, then fine-tuned on the OKVQA training set.
- Main experiments use GPT-3 and OPT as frozen LLMs.
- To avoid data contamination, images appearing in the OKVQA test set are removed from pretraining data.

## Key Experimental Results

### Main Results — Zero-shot Performance

| Method | VQAv2 test | A-OKVQA test | OKVQA test |
|--------|-----------|-------------|------------|
| Flamingo-80B | 56.21 | - | 50.57 |
| Img2LLM w/ GPT-3 | 59.22 | 43.39 | 42.80 |
| Img2LLM w/ OPT | 61.83 | 40.69 | 45.58 |
| **OAD-Promoter w/ OPT** | **61.93** | **40.68** | **45.58** |
| **OAD-Promoter w/ GPT-3** | **61.98** | **41.71** | **45.61** |

### Generalization Across Different LLMs (OKVQA Zero-shot)

| LLM | Parameters | OKVQA |
|-----|-----------|-------|
| GPT-Neo | 2.7B | 33.41 |
| GPT-J | 6B | 38.89 |
| BLOOM | 7.1B | 33.77 |
| OPT | 6.7B | 36.18 |
| OPT | 30B | 40.46 |
| OPT | 175B | 45.58 |
| GPT-3 | 175B | 45.61 |

### Ablation Study

| Configuration | OKVQA (Few-shot) | OKVQA (Zero-shot) | Note |
|---------------|-----------------|-------------------|------|
| w/o OEG + w/o MKA | 47.33 | 42.50 | Baseline |
| w/ OEG + w/o MKA | 54.68 | 44.26 | OEG contributes most |
| w/o OEG + w/ MKA | 48.95 | 43.64 | MKA alone also helps |
| **w/ OEG + w/ MKA** | **60.04** | **45.61** | Best synergy |

| MKA Memory Size K | OKVQA | Note |
|-------------------|-------|------|
| 0 | 43.64 | No memory |
| 60 | 43.65 | Few examples |
| 200 | 43.92 | Moderate examples |
| 400 | 44.15 | More examples is better |

### Domain Transfer (Few-shot, Different LLMs)

| LLM | VQA-CP | GQA-OOD |
|-----|--------|---------|
| GPT-4 (GRACE) | 57.61 | 50.19 |
| **GPT-4 (OAD-Promoter)** | 55.93 | **50.21** |

### Key Findings
1. A new state-of-the-art is achieved on VQAv2 zero-shot (61.98), surpassing all large-scale multimodal pretraining methods and frozen LLM methods.
2. The OEG module contributes the most in the few-shot setting (+7.35), confirming that fine-grained visual information is critical for mitigating language bias.
3. Directly integrating conventional debiasing methods (LMH, CSS) into the LLM pipeline degrades OKVQA performance (Table 4), indicating that language bias in LLMs requires fundamentally different remediation strategies.
4. With GPT-4, OAD-Promoter achieves the best result on GQA-OOD (50.21), suggesting that stronger LLMs can better leverage domain transfer capabilities.
5. The growing memory bank mechanism yields continuous performance improvement (K=400 > K=200 > K=60).
6. OAD-Promoter is invariant to input ordering (100% correct rate), whereas Img2LLM is sensitive to ordering.

## Highlights & Insights
1. **Exposes the severity of language bias in LLMs**: Language bias is not limited to conventional VQA models; LLM-based methods suffer equally, and conventional debiasing approaches are ineffective in this context.
2. **Novel strategy for counteracting bias**: Selecting the least similar examples in Negative mode to counteract bias is both innovative and empirically effective.
3. **Zero-shot surpasses few-shot**: Thanks to the growing memory mechanism in MKA, reasoning capability continuously improves in the zero-shot setting.
4. **Plug-and-play**: The framework is compatible with various LLMs (GPT-3, OPT, BLOOM, GPT-Neo, GPT-J, GPT-4, etc.).
5. **Self-growing memory bank**: The MKA module naturally accumulates knowledge throughout inference, representing an elegant form of continual learning.

## Limitations & Future Work
1. The framework depends on the quality of pretrained models such as VinVL and BLIP2; failures in these components degrade OEG output.
2. Bias detection in MKA relies on the UpDn and LMH QA modules, whose individual capabilities are limited.
3. Performance gains on A-OKVQA and OKVQA under the zero-shot setting are modest (<1%), indicating a ceiling for more challenging knowledge-intensive reasoning tasks.
4. Unbounded growth of the memory bank may introduce storage and retrieval efficiency concerns.
5. The Positive/Negative mode selection is a hard switch; soft interpolation could be considered.

## Related Work & Insights
- The bias detection strategy (comparing predictions with vs. without visual input) is generalizable to other settings where shortcut learning detection is needed.
- The memory-augmented inference framework offers a useful reference for other reasoning tasks requiring continuous improvement.
- The multi-granularity visual information integration strategy (global + regional) is worth adopting in other vision-language model research.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Calibrating Uncertainty for Zero-Shot Adversarial CLIP](../../ICML2026/ai_safety/calibrating_uncertainty_for_zero-shot_adversarial_clip.md)
- [\[CVPR 2026\] $\varphi$-DPO: Fairness Direct Preference Optimization Approach to Continual Learning in Large Multimodal Models](../../CVPR2026/ai_safety/φ-dpo_fairness_direct_preference_optimization_approach_to_continual_learning_in_.md)
- [\[AAAI 2026\] TopoReformer: Mitigating Adversarial Attacks Using Topological Purification in OCR Models](toporeformer_mitigating_adversarial_attacks_using_topological_purification_in_oc.md)
- [\[AAAI 2026\] Transferable Backdoor Attacks for Code Models via Sharpness-Aware Adversarial Perturbation](transferable_backdoor_attacks_for_code_models_via_sharpness-aware_adversarial_pe.md)
- [\[CVPR 2026\] When Robots Obey the Patch: Universal Transferable Patch Attacks on Vision-Language-Action Models](../../CVPR2026/ai_safety/when_robots_obey_the_patch_universal_transferable_patch_attacks_on_vision-langua.md)

</div>

<!-- RELATED:END -->
