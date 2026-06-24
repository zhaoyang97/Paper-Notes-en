---
title: >-
  [Paper Note] Cross-Lingual Optimization for Language Transfer in Large Language Models
description: >-
  [ACL 2025][Multilingual & Machine Translation][cross-lingual transfer] This paper proposes Cross-Lingual Optimization (CLO), which modifies the DPO loss function to achieve cross-lingual preference optimization—preferring target language responses given target language inputs, and English responses given English inputs. It consistently outperforms SFT across 5 models × 6 languages; in low-resource languages, CLO requiring only 3,200 samples outperforms SFT trained on 6…
tags:
  - "ACL 2025"
  - "Multilingual & Machine Translation"
  - "cross-lingual transfer"
  - "DPO"
  - "multilingual"
  - "low-resource language"
  - "language adaptation"
  - "attention-only tuning"
date: 2026-05-08
content_hash: 69d9457dd3d82fec
---

# Cross-Lingual Optimization for Language Transfer in Large Language Models

**Conference**: ACL 2025  
**arXiv**: [2505.14297](https://arxiv.org/abs/2505.14297)  
**Code**: None  
**Area**: Multilingual Translation  
**Keywords**: cross-lingual transfer, DPO, multilingual, low-resource language, language adaptation, attention-only tuning

## TL;DR

This paper proposes Cross-Lingual Optimization (CLO), which modifies the DPO loss function to achieve cross-lingual preference optimization—preferring target language responses given target language inputs, and English responses given English inputs. It consistently outperforms SFT across 5 models × 6 languages; in low-resource languages, CLO requiring only 3,200 samples outperforms SFT trained on 6,400 samples.

## Background & Motivation

**Background**: English-centric LLMs (Llama, Mistral, Qwen, etc.) perform significantly worse in other languages. The standard practice is to perform language transfer via SFT using instruction data in the target language.

**Limitations of Prior Work**: SFT suffers from a severe "English bias" problem in data-scarce scenarios—the model may understand the target language input but still defaults to replying in English (e.g., Llama3 Chat can understand Swahili but answers in English). For low-to-medium resource languages, the stability of SFT is particularly poor.

**Key Challenge**: How to effectively transfer to the target language while retaining the model's English capabilities? SFT leads to a trade-off dilemma: too much English data results in insufficient target language transfer, whereas too much target language data degrades English performance, and low-resource languages often lack sufficient high-quality instruction data.

**Goal**: To achieve efficient cross-lingual transfer under data-constrained environments using accessible English SFT data + translation models, while maintaining English performance.

**Key Insight**: Convert the problem from "learning target language knowledge" to "learning the correspondence between input and output languages"—suppressing "target language input $\rightarrow$ English response" and enhancing "target language input $\rightarrow$ target language response", and vice versa.

**Core Idea**: Synthesize cross-lingual preference pairs (where the target language response to the same question is "chosen" and the English response is "rejected") combined with a modified DPO to train the model to "respond in the correct language".

## Method

### Overall Architecture

English SFT data $(x_{en}, y_{en})$ $\rightarrow$ translation model (M2M100-1.2B) generates target language data $(x_\ell, y_\ell)$ $\rightarrow$ construct cross-lingual preference pairs $\rightarrow$ CLO Loss = $\lambda \cdot \mathcal{L}_{SFT} + (1-\lambda) \cdot \mathcal{L}_{CL}$ $\rightarrow$ fine-tune only attention layer parameters.

### Key Designs

#### 1. Cross-Lingual Dataset Preparation

- **Function**: Translate English SFT data into target languages to construct two sets of preference pairs.
- **Why**: To make the model explicitly learn the correspondence of "input language-output language", rather than just learning the content.
- **How**:
    - English input $x_{en}$: chosen = $y_{en}$ (English response), rejected = $y_\ell$ (target language response) $\rightarrow$ retain English capabilities.
    - Target language input $x_\ell$: chosen = $y_\ell$ (target language response), rejected = $y_{en}$ (English response) $\rightarrow$ transfer target language capabilities.
    - Use the M2M100-1.2B translation model to generate a total of 12,800 cross-lingual pairs from 6,400 English data entries.

#### 2. Cross-Lingual Optimization Loss

- **Function**: A joint optimization objective combining NLL and a modified DPO.
- **Why**: The base model itself cannot answer queries; NLL teaches the model to generate responses, while the modified DPO teaches the model to select the correct language.
- **How**:
    - $\mathcal{L}_{CLO} = \lambda \cdot \mathcal{L}_{SFT} + (1-\lambda) \cdot \mathcal{L}_{CL}$
    - $\mathcal{L}_{SFT}$: Compute NLL only on the **target language output** (key design: exclude English NLL to mitigate English bias).
    - $\mathcal{L}_{CL}$: Cross-lingual DPO loss, consisting of two parts: preferring English responses to suppress target language responses for English inputs + preferring target language responses to suppress English responses for target language inputs.
    - Ablation study verification: Adding English NLL causes the model to bias toward English output (Swahili Win Rate drops from 83.0 to 74.5).

#### 3. Attention-Only Fine-Tuning

- **Function**: Only update attention layer parameters during CLO training while freezing other layers.
- **Why**: Based on the findings of Zeping & Sophia (2024), language-related capabilities are primarily stored in attention layers, with important neurons concentrated in deeper layers.
- **How**: Perform gradient updates only on the Q/K/V/O projection matrices.
- **Effect**: Comparable to full-parameter fine-tuning (Chinese/Korean Win Rate around 50:50), but training GPU memory is about 30% lower than DPO, and only about 55% higher than SFT.
- **Exception**: In extremely low-resource languages like Swahili, attention-only training performance is significantly lower than full-parameter training (Win 29.7% vs 70.3%) because the model's embedded language knowledge is insufficient.

### Loss & Training

- English seed data: Top 6,400 highly-ranked single-turn data entries from OpenAssistant.
- Translation model: M2M100-1.2B.
- Training objective: $\mathcal{L}_{CLO}$ joint loss.
- Reference model: The base model itself (consistent with DPO).
- Trainable parameters: Attention layer only.

## Key Experimental Results

### Main Results 1: AlpacaEval Instruction-Following Ability (Win Rate %, vs SFT Baseline)

| Model | Chinese (High) | German (High) | Korean (Medium) | Indonesian (Medium) | Swahili (Low) | Yoruba (Low) |
|------|---------|---------|---------|-----------|-------------|-------------|
| Llama3-8B CLO $\Delta$ | +11.3 | +2.9 | +15.3 | +1.5 | **+17.6** | +11.6 |
| Llama2-7B CLO $\Delta$ | +2.0 | +9.2 | +1.3 | +5.0 | +0.9 | **+23.6** |
| Llama2-13B CLO $\Delta$ | +5.9 | +3.2 | +2.3 | +9.3 | **+17.0** | **+23.8** |
| Mistral-7B CLO $\Delta$ | +0.6 | +0.9 | +2.0 | +1.1 | **+15.9** | +0.2 |
| Qwen2.5-3B CLO $\Delta$ | +5.7 | +1.0 | +10.1 | +1.1 | **+23.0** | **+23.0** |

**Key Findings**: CLO outperforms SFT+DPO in all 30 (model, language) combinations, with the most significant improvements in low-resource languages (max $\Delta$ up to +23.8). English performance is simultaneously maintained or even improved.

### Main Results 2: BELEBELE Reading Comprehension Accuracy

| Model | Method | Swahili | Yoruba | Korean |
|------|------|-----------|---------|------|
| Llama3-8B | SFT | 42.0 | 29.6 | 46.7 |
| Llama3-8B | CLO | 42.6 | 29.8 | **57.7** |
| Qwen2.5-3B | SFT | 51.7 | 45.9 | 52.3 |
| Qwen2.5-3B | CLO | **74.7** | **68.9** | **62.4** |

### Main Results 3: MMMLU Reasoning Ability

| Model | Method | Chinese | Korean | Swahili | English |
|------|------|------|------|-----------|------|
| Llama3-8B | SFT | 39.36 | 25.31 | 27.59 | 53.00 |
| Llama3-8B | CLO | **41.99** | **32.73** | **33.38** | **57.55** |
| Qwen2.5-3B | SFT | 46.34 | 35.90 | 26.22 | 55.70 |
| Qwen2.5-3B | CLO | **52.10** | **41.94** | **29.80** | **60.52** |

### Ablation Study

| Ablation Configuration | Swahili Win Rate | English Win Rate |
|---------|-------------------|--------------|
| CLO (NLL target language only, ours) | **83.0** | 65.4 |
| CLO (NLL target + English) | 74.5 | 67.8 |

| Attention-only vs Full (Llama2) | Target Language Win% | English Win% |
|------|---------|---------|
| Chinese | 50.7 vs 49.1 | 54.4 vs 45.6 |
| Korean | 52.7 vs 46.4 | 52.7 vs 47.3 |
| Swahili | 29.7 vs **69.6** | 50.1 vs 49.9 |

### Data Efficiency Experiments

| Training Data Size | SFT Swahili | CLO Swahili |
|-----------|--------------|--------------|
| 1,600 pairs | Far below 6,400 SFT | $\approx$ 6,400 SFT |
| 3,200 pairs | Still below 6,400 SFT | **> 6,400 SFT** |
| 6,400 pairs | Baseline | Far exceeds baseline |

**Key Conclusion**: CLO requires only 1,600 pairs of data in Swahili to match the performance of SFT using 6,400 pairs, representing a 4x increase in data efficiency.

### Summary of Key Findings

- **CLO consistently outperforms SFT across all languages and models**, with the largest advantage in low-resource languages (max $\Delta$ +23.8%).
- **Data Efficiency**: CLO 3,200 > SFT 6,400 (low-resource); CLO 400 $\approx$ SFT 6,400 (Llama3 Swahili).
- **SFT is highly sensitive to the volume of data in low-resource languages** (showing a huge jump from 3,200 $\rightarrow$ 6,400), whereas CLO improves smoothly.
- **CLO maintains or even improves English capabilities simultaneously**, whereas SFT's English capability often degrades after adding target language data.
- **Computing NLL only on the target language** is a key design; including English NLL reintroduces English bias.

## Highlights & Insights

- **The cross-lingual preference pair design is highly elegant**—it transforms the "language selection" problem into a classic preference optimization problem, utilizing translated data to build "language correctness" preference signals without requiring any human annotation.
- **Fine-tuning only the attention layers** is a practical finding, validating the hypothesis that language capabilities are mainly stored in the attention layers, which substantially reduces training costs.
- **The data efficiency improvement for low-resource languages** holds direct value for practical multilingual deployment—many languages struggle to obtain even 10,000 high-quality instruction data entries.
- **Systematic demonstration of SFT's "English bias"**: Through the comparison of three variants (SFT-eng, SFT-tgt, and SFT), it clearly illustrates the failure modes of SFT in medium-to-low resource languages.
- **Qwen2.5-3B, despite being the smallest model, achieves the best multilingual transfer results**, indicating that multilingual data coverage in pre-training is more important than model size.

## Limitations & Future Work

- **Only supports single-language transfer**: It can only transfer to one target language at a time and does not support simultaneous multilingual transfer.
- **Dependency on translation model quality**: The translation quality of M2M100 might be poor for low-resource languages, but the paper argues that the comparison is fair since SFT and CLO share the same translation data.
- **Insufficient language-specific evaluation**: The evaluation uses translated AlpacaEval/MMMLU, failing to capture language-specific cultural contexts.
- **Validated only on DPO**: It does not explore the compatibility of the CLO framework with other preference optimization algorithms (e.g., KTO, SimPO).
- **Attention-only fails for extremely low-resource languages**: In Swahili, attention-only training lags significantly behind full-parameter training, requiring adaptive strategies.
- **Directions for improvement**: Simultaneous multilingual transfer, integration with KTO/SimPO, adaptive layer selection strategies, and stronger translation models.

## Related Work & Insights

- **vs Standard SFT (Lee et al. 2023; Shaham et al. 2024a)**: SFT simply mimics the target language output via maximum likelihood and cannot explicitly learn language selection strategies; CLO teaches the model to "choose the right language" through preference optimization, which shows a significant advantage in low-resource scenarios.
- **vs Continued Pre-training + SFT (Cui et al. 2023; Zhao et al. 2024a)**: Continued pre-training requires a large corpus in the target language (usually millions of tokens), whereas CLO only requires 6,400 translated data entries, presenting a massive cost difference.
- **vs InstructionCP (Chen & Lee 2024)**: InstructionCP requires large-scale target language instruction data and complex architectural analysis, whereas CLO only requires English data + a translation model, offering a much simpler approach.

## Rating

- Novelty: ⭐⭐⭐⭐ Innovatively applies DPO to cross-lingual transfer, presenting a novel perspective of "language selection preference"; however, the core technology (modified DPO + NLL) is not overly complex.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 5 models × 6 languages × 3 evaluation benchmarks, with comprehensive ablations (NLL variants, attention-only, data volume curves). The experimental scale is top-tier in this field.
- Writing Quality: ⭐⭐⭐⭐ Clear hypotheses, rigorous methodology derivation, and intuitive illustrations; however, the Limitations section is slightly verbose.
- Value: ⭐⭐⭐⭐⭐ Direct practical value for deploying LLMs in low-resource languages; the method is simple and easy to reproduce, requiring only public English data + a translation model.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Semantic Aware Linear Transfer by Recycling Pre-trained Language Models for Cross-Lingual Transfer](semantic_aware_linear_transfer_by_recycling_pre-trained_language_models_for_cros.md)
- [\[ACL 2025\] Bridging the Language Gaps in Large Language Models with Inference-Time Cross-Lingual Intervention](bridging_the_language_gaps_in_large_language_models_with_inference-time_cross-li.md)
- [\[ACL 2025\] Language Fusion for Parameter-Efficient Cross-lingual Transfer (FLARE)](flare_crosslingual_lora.md)
- [\[ACL 2025\] Cross-Lingual Transfer of Cultural Knowledge: An Asymmetric Phenomenon](cross-lingual_transfer_of_cultural_knowledge_an_asymmetric_phenomenon.md)
- [\[ACL 2025\] Cross-Lingual Pitfalls: Automatic Probing Cross-Lingual Weakness of Multilingual Large Language Models](crosslingual_pitfalls.md)

</div>

<!-- RELATED:END -->
