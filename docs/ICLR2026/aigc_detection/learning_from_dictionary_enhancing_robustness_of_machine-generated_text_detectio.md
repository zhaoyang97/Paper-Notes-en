---
title: >-
  [Paper Note] Learning From Dictionary: Enhancing Robustness of Machine-Generated Text Detection in Zero-Shot Language via Adversarial Training
description: >-
  [ICLR 2026][AIGC Detection][Machine-Generated Text Detection] To address the sharp drop in robustness of Machine-Generated Text (MGT) detectors on unseen languages, this paper proposes the TASTE framework: it uses translation dictionaries to perform "code-switching" on MGTs to generate multilingual adversarial samples. Combined with a gradient-reversal language discriminator (LAAL loss), it forces the detector to learn language-agnostic features. Using only single-language an…
tags:
  - "ICLR 2026"
  - "AIGC Detection"
  - "Machine-Generated Text Detection"
  - "Zero-Shot Language"
  - "Adversarial Training"
  - "Translation Dictionary"
  - "Language-Agnostic Features"
date: 2026-05-08
content_hash: 78b494aad234d985
---

# Learning From Dictionary: Enhancing Robustness of Machine-Generated Text Detection in Zero-Shot Language via Adversarial Training

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=bTcFHJo1Zk](https://openreview.net/forum?id=bTcFHJo1Zk)  
**Code**: https://github.com/Liyuuuu111/MGT-Eval (Available)  
**Area**: AIGC Detection / Machine-Generated Text Detection / Adversarial Training / Multilingual  
**Keywords**: Machine-Generated Text Detection, Zero-Shot Language, Adversarial Training, Translation Dictionary, Language-Agnostic Features

## TL;DR
To address the sharp drop in robustness of Machine-Generated Text (MGT) detectors on unseen languages, this paper proposes the TASTE framework: it uses translation dictionaries to perform "code-switching" on MGTs to generate multilingual adversarial samples. Combined with a gradient-reversal language discriminator (LAAL loss), it forces the detector to learn language-agnostic features. Using only single-language annotations and translation dictionaries, it improves the average F1 on zero-shot languages to 0.773 and suppresses the average Attack Success Rate to 18.0%.

## Background & Motivation
**Background**: As LLMs make high-quality text generation effortless, machine-generated text (MGT) detection has become a critical defense for maintaining the credibility of web content. Mainstream detectors are categorized into two types: metric-based (e.g., Fast-DetectGPT, Binoculars, LRR, using statistics like perplexity or log-likelihood ratio) and model-based (fine-tuning encoders like mBERT or XLM-R on annotated corpora for binary classification).

**Limitations of Prior Work**: These detectors achieve high accuracy in monolingual (especially English) scenarios, but a recent benchmark reveals a concerning reality—all detectors suffer a significant performance collapse on "zero-shot languages" (languages unseen during training). Worse still, robustness collapses: in zero-shot languages, **modifying just two words** can drop detection accuracy by 20–40%, allowing attackers to bypass them with minimal effort. Currently, most detectors are developed and evaluated only in English, leaving billions of non-English readers unprotected.

**Key Challenge**: To perform adversarial training in multilingual scenarios for enhanced robustness, existing methods require **large-scale multilingual corpora**. However, mid-resource languages often only have translation dictionaries and lack annotated corpora, while low-resource languages lack even basic lexical resources. It is challenging to collect sufficient data for multilingual adversarial training. There is a direct conflict between the "desire for multilingual robustness" and the "unavailability of multilingual data."

**Goal**: To train a robust multilingual detector that can generalize to zero-shot languages and resist various attacks, using only single high-resource language (English) data/labels plus **translation dictionaries** for mid-resource languages.

**Key Insight**: The authors note that translation dictionaries are a much cheaper and overlooked source of cross-lingual supervision compared to annotated corpora. If an "attacker" uses a dictionary to replace keywords in English MGT with other languages (i.e., code-switching), the generated adversarial samples naturally carry multilingual perturbations. By forcing the detector to provide consistent predictions for the original and code-switched samples, it is compelled to learn semantic features independent of specific languages.

**Core Idea**: Replace "large-scale multilingual corpora" with a "translation dictionary-driven code-switching attacker" to create multilingual adversarial samples, and use a gradient-reversal language discriminator to erase language-specific cues from features, thereby transferring English supervision to zero-shot languages.

## Method

### Overall Architecture
TASTE (Translation-based Attacker Strengthens MulTilingual DefEnder) is an **adversarial training framework where the attacker and detector are updated synchronously**. The input consists of Human-Written Text (HWT) and Machine-Generated Text (MGT) from a single English corpus; the output is a target detector $D_{tar}$ robust to zero-shot languages and unseen attacks.

The process involves a three-stage cycle within each training step: (A) The **attacker** first uses a proxy model $D_{sur}$ to estimate the importance of each token in the MGT, selects the top-$k$ key tokens, and uses translation dictionaries for code-switching into multiple languages to create an adversarial sample $\tilde{X}$; (B) The **detector** optimizes two losses simultaneously on "clean + adversarial" samples—the Robustness Preservation Loss $L_{RPL}$ (predicting correctly for both $X$ and $\tilde{X}$) and the Language-Agnostic Adversarial Loss $L_{LAAL}$ (erasing language cues via gradient reversal); (C) An **attack strength scheduler** gradually increases the size of the code-switched token set $|I|$ from 1, making the adversarial samples progressively harder. The attacker assumes a black-box setting—it only sees the labels output by the target detector—and thus uses the proxy model to obtain gradients.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["Input: English HWT / MGT"] --> B["Key Token Identification<br/>Proxy model D_sur estimates token importance<br/>Select top-k key tokens"]
    B --> C["Translation Dictionary Code-switching<br/>Replace key tokens with multilingual equivalents<br/>Generate adversarial sample X̃"]
    C --> D["Detector Dual-Loss Training<br/>RPL: Correct prediction for X and X̃<br/>LAAL: Erase language cues via gradient reversal"]
    D -->|Attack intensity scheduler: increase |I| from 1| B
    D --> E["Output: Robust Multilingual Detector D_tar"]
```

### Key Designs

**1. Translation Dictionary-Driven Code-Switching Attacker: Replacing expensive multilingual corpora with cheap dictionaries**

This design directly addresses the lack of data for multilingual adversarial training. The attacker generates strong adversarial samples under a black-box setting in two steps. First: **Key Token Identification**. Since the weights of the target detector $D_{tar}$ are invisible, the authors introduce an open-source proxy model $D_{sur}$ to estimate token importance. Given MGT $X=[x_1,\dots,x_T]$ and label $l$, the last layer hidden states $H=D_{sur}(X)=[h_1,\dots,h_T]$ are obtained. The classification loss $L_{cls}=-\log P_{sur}(l\mid X)$ is used to calculate the gradient norm for each hidden state as an importance score $g_t=\left\|\frac{\partial L_{cls}}{\partial h_t}\right\|_2$. The top-$k$ tokens form the key set $I=\text{top-}k([(x_t,g_t)])$. Second: **Code-switching**. Each English token in $I$ is looked up in translation dictionary $T$ and replaced with the corresponding word in target languages to obtain the multilingual adversarial sample $\tilde{X}$ (e.g., mixing "launched→released, new→brand-new, Archon→archon" in the same sentence).

This is effective because perturbing only the "few most critical tokens for detection" can efficiently flip predictions, and translation dictionaries are readily available for mid-resource languages, bypassing dependency on large-scale annotated corpora.

**2. Language-Agnostic Adversarial Loss (LAAL): Erasing "Language" from features via gradient reversal**

Creating code-switched samples is not enough—the detector might learn language-specific shortcuts like "non-English characters imply machine-written text," failing to generalize. LAAL solves this using a domain-adversarial approach: the $[\text{CLS}]$ vector $h_{cls}^{(i)}$ from the encoder is fed into a lightweight **language discriminator** $f_{lang}(\cdot)$ to determine if the input is code-switched ($y_{lang}=1$) or clean ($y_{lang}=0$). The discriminator minimizes cross-entropy:

$$L_{LAAL}=-\frac{1}{N}\sum_{i=1}^{N}\left[y_{lang}^{(i)}\log p_{lang}^{(i)}+(1-y_{lang}^{(i)})\log(1-p_{lang}^{(i)})\right]$$

Crucially, during backpropagation, the gradient flowing from the discriminator back to the encoder is multiplied by a negative coefficient $-\lambda_{lang}$. This **gradient reversal** forces the encoder to "deceive" the discriminator—maximizing the discriminator's loss—thereby erasing language-specific cues and retaining only cross-lingual semantic signals. Combined with the Robustness Preservation Loss $L_{RPL}$ (requiring correct labels $l_i$ for both $X$ and $\tilde{X}$):

$$L_{RPL}=\frac{1}{N}\sum_{i=1}^{N}\left[-\log P_{tar}(l_i\mid X_i)-\log P_{tar}(l_i\mid \tilde{X}_i)\right]$$

The total detector loss is $L_{det}=L_{RPL}+\lambda_{lang}L_{LAAL}$. This combination transfers the English decision boundary to unseen languages.

**3. Synchronous Attacker-Detector Update + Dynamic Attack Intensity: Progressing from easy to hard to prevent underfitting**

Unlike traditional adversarial training using fixed intensity or static samples, TASTE updates the attacker and detector alternately within the **same training step** to achieve co-evolution. In each step, the detector is frozen, and the attacker loss $L_{att}=\frac{1}{N}\sum_i[-\log P_{sur}(\hat{l}_i\mid X_i)]$ updates the proxy model ($\hat{l}_i=\arg\max D_{tar}(X_i)$ is the pseudo-label from the target detector). This step distills the evolving knowledge of the target detector into the proxy model, making its gradients more reliable for token importance estimation. Then, the attacker is frozen, and $L_{det}$ updates the detector and language discriminator.

Dynamic intensity is reflected in the code-switch set size $|I|$, which starts at 1 and increases to a maximum $\max|I|$, allowing the detector to learn from simple to complex adversarial samples, mitigating underfitting. Experiments show intensity has a "sweet spot": Acc is highest and ASR is lowest at a code-switch ratio of 0.05; at 0.25, performance degrades due to excessive noise causing overfitting to extreme samples.

### Loss & Training
- **Attacker Loss** $L_{att}$: Distills target detector knowledge into proxy model using pseudo-labels; detector is frozen during $D_{sur}$ update.
- **Detector Loss** $L_{det}=L_{RPL}+\lambda_{lang}L_{LAAL}$: Attacker is frozen during $D_{tar}$ and $f_{lang}$ update.
- **Intensity Scheduling**: $s\leftarrow\min(s+\Delta,\,s_{max})$, gradually increasing code-switched tokens.
- Training uses only a single English annotated corpus + translation dictionaries for Ar/Zh/De/Ru; the proxy model is a GPT-2 fine-tuned on corresponding data; the target detector uses mBERT as the backbone for fair comparison with all baselines.

## Key Experimental Results

### Main Results
Evaluated zero-shot cross-lingual detection on 9 languages in the M4GT dataset (where bg/id/it/ur are zero-shot languages), comparing 8 SOTA detectors:

| Setting | Metric | TASTE | Next Best SOTA | Gain |
|------|------|-------|-----------|------|
| Avg. across 9 languages | Acc | 0.762 | 0.691 | +0.071 (+10.3%) |
| Avg. across 9 languages | F1 | 0.773 | 0.709 | +0.064 (+9.0%) |
| Avg. across 4 Zero-Shot | Acc | 0.762 | 0.741 (mBERT) | +2.1 pp |
| Avg. across 4 Zero-Shot | F1 | 0.786 | 0.766 (mBERT) | +2.0 pp |
| English | F1 | 0.971 | 0.893 (GREATER-D) | +7.8 pp |

In terms of robustness (average Attack Success Rate ASR across 8 attacks, lower is better):

| Detector | Avg. ASR ↓ | vs TASTE |
|--------|-----------|-----------|
| TASTE (Ours) | 18.0% | — |
| GREATER-D (Strongest SOTA Defense) | 21.8% | TASTE relative ↓ 17.4% |
| Fast-DetectGPT (Best Metric Baseline) | 41.7% | TASTE relative ↓ 56.8% |

TASTE outperforms in 8 out of 9 languages, with an average absolute reduction of 3.82 ASR points (e.g., ar 17.1→10.0, ur 20.5→12.0, zh 23.2→16.3), significant at $p=0.039$. By attack type: ASR dropped by 38.9% for DELETE, 32.8% for CODE-SWITCHING, 59.7% for HUMAN-OBF., and achieved near-zero vulnerability under SWAP attacks.

### Ablation Study
The paper analyzes intensity and dictionary quality:

| Configuration | Key Metric | Description |
|------|---------|------|
| Attack Intensity 0.05 | Acc 76.2% / ASR 18.0% | Sweet spot: highest accuracy, lowest ASR |
| Attack Intensity 0.25 | Acc 73.1% / ASR 34.2% | Excessive perturbation → overfitting, performance drop |
| Dictionary Swap Rate 0% (Clean) | Avg. Acc 76.2% | Dictionary provides effective cross-lingual signals |
| Dictionary Swap Rate 90% (High Noise) | Avg. Acc 68.0% | Degraded but still above best baseline (0.691) |
| Dictionary Coverage 10% | Still > Best baseline 0.691 | Small dictionaries suffice; full coverage not required |

### Key Findings
- **Intensity Sweet Spot**: A code-switching ratio of ~0.05 balances accuracy, robustness, and training time. Training time per step surged from 150s to 9468s as intensity rose from 0.00 to 0.30; excessive intensity is both expensive and leads to overfitting.
- **Resilience to Dictionary Noise/Incompleteness**: While performance generally drops as noise increases, results remain above the best baseline even at 90% error. Languages relying more on cross-lingual transfer (Chinese, Indonesian) are more vulnerable, whereas moderate noise sometimes helps remove pseudo-lexical cues (Italian 69.4%→89.0% @70% noise).
- **Zero-Shot Generalization via Adversarial Training**: Trained only on English, TASTE maintains high accuracy and F1 across all languages, verifying that language-agnostic features effectively transfer to unseen languages.

## Highlights & Insights
- **Translation Dictionary as an Adversarial Weapon**: The most ingenious part is recognizing that mid-resource languages lack annotated corpora but have dictionaries. Using dictionary code-switching instead of expensive multilingual corpora reduces costs while naturally introducing multilingual perturbations.
- **Coupling GRL with Code-Switched Samples**: Code-switching alone could lead the model to learn "non-English = machine-written," and GRL alone would have no multilingual signal to erase. Together, LAAL squeezes out language-specific cues, which is the key to generalization.
- **Distillation via Proxy Model for Black-Box Gradients**: In a black-box setting where target gradients are unavailable, the authors use pseudo-labels to continuously distill target detector knowledge into a proxy model, making its importance estimation increasingly accurate.
- **Curriculum-like Intensity Scheduling**: Gradually increasing code-switched tokens acts as a curriculum for the detector. The discovery of the 0.05 sweet spot provides valuable reference for curriculum-based adversarial training.

## Limitations & Future Work
- **Dependency on Dictionary Quality/Coverage**: While resilient to noise, performance drops significantly for languages like Chinese or Indonesian at 90% swap rates. For true low-resource languages lacking even dictionaries, the foundation of the method is weakened.
- **Static Word-Level Substitution**: Code-switching is word-for-word and cannot handle language-specific syntax or morphology, potentially generating samples with semantic drift.
- **Limited Multilingual Validation**: Training dictionaries only covered Ar/Zh/De/Ru. Whether the method remains effective for languages with more complex morphology or vastly different script systems requires broader validation.
- **Improvement Ideas**: Introduce context-aware translation (instead of static dictionaries) to reduce semantic drift; use multi-pivot language dictionary combinations for low-resource languages.

## Related Work & Insights
- **vs RADAR / GREATER-D (Monolingual Adversarial Training)**: These also use synchronous attacker-detector updates (GREATER-D is from Li et al. 2025), but attacks are limited to monolingual paraphrasing/perturbation and fail to generalize to multilingual settings. TASTE replaces the attack with dictionary code-switching + LAAL for zero-shot languages, achieving 0.064 higher F1 and 3.8 points lower ASR than GREATER-D.
- **vs Fast-DetectGPT / Binoculars / LRR (Metric-based)**: These rely on statistical properties like perplexity. Their ASR reaches 41.7% in zero-shot languages, showing high vulnerability to word-level perturbations. TASTE, as an adversarial model-based method, reduces ASR to 18.0%.
- **vs Traditional Multilingual Adversarial Training**: Previous work generally assumed multilingual robustness requires massive multilingual data. This paper demonstrates cross-lingual generalization via cheap dictionaries + English annotations, challenging the "scale for robustness" assumption.

## Rating
- Novelty: ⭐⭐⭐⭐ Practical and novel use of dictionaries + GRL for zero-shot multilingual MGT detection.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 9 languages and 8 attacks against 8 SOTA baselines. Detailed analysis of noise/intensity, though mapping language coverage was limited to 4.
- Writing Quality: ⭐⭐⭐⭐ Clear chain from motivation to method and experiments. Complete pseudocode and loss formulas.
- Value: ⭐⭐⭐⭐ Addresses protection for non-English readers with a low-cost solution, highly valuable for practical multilingual deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Iron Sharpens Iron: Defending Against Attacks in Machine-Generated Text Detection with Adversarial Training](../../ACL2025/aigc_detection/greater_adversarial_mgt_detection.md)
- [\[ACL 2025\] Reliably Bounding False Positives: A Zero-Shot Machine-Generated Text Detection Framework via Multiscaled Conformal Prediction](../../ACL2025/aigc_detection/reliably_bounding_false_positives_a_zero-shot_machine-generated_text_detection_f.md)
- [\[CVPR 2025\] Enhancing Few-Shot Class-Incremental Learning via Training-Free Bi-Level Modality Calibration](../../CVPR2025/aigc_detection/enhancing_few-shot_class-incremental_learning_via_training-free_bi-level_modalit.md)
- [\[ICML 2026\] Dissect and Prune: Enhancing Robustness in AI-Generated Image Detection](../../ICML2026/aigc_detection/dissect_and_prune_enhancing_robustness_in_ai-generated_image_detection.md)
- [\[ICLR 2026\] Learn-to-Distance: Distance Learning for Detecting LLM-Generated Text](learn-to-distance_distance_learning_for_detecting_llm-generated_text.md)

</div>

<!-- RELATED:END -->
