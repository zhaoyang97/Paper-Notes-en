---
title: >-
  [Paper Note] Data-efficient Targeted Token-level Preference Optimization for LLM-based Text-to-Speech
description: >-
  [ACL 2026][Audio & Speech][TKTO] To address the alignment challenge of ambiguous pronunciations in LLM-based TTS (e.g., the Japanese word "辛い" can be read as either *karai* or *tsurai*)…
tags:
  - "ACL 2026"
  - "Audio & Speech"
  - "TKTO"
  - "KTO"
  - "Unpaired Preferences"
  - "Token-level Reward"
  - "Japanese Ambiguous Pronunciation"
date: 2026-05-08
content_hash: bb3bed6abe33677a
---

# Data-efficient Targeted Token-level Preference Optimization for LLM-based Text-to-Speech

**Conference**: ACL 2026  
**arXiv**: [2510.05799](https://arxiv.org/abs/2510.05799)  
**Code**: Not publicly disclosed by the authors (no repository link in cache)  
**Area**: LLM Alignment / TTS / Preference Optimization  
**Keywords**: TKTO, KTO, Unpaired Preferences, Token-level Reward, Japanese Ambiguous Pronunciation

## TL;DR
To address the alignment challenge of ambiguous pronunciations in LLM-based TTS (e.g., the Japanese word "辛い" can be read as either *karai* or *tsurai*), the authors propose TKTO. This method first estimates importance weights $w_t$ for each token using two contrasting KTO models trained with swapped labels. It then decomposes the utterance-level value function of KTO into token-level components with weighted aggregation. This achieves a dual upgrade: "no paired data required + automatic localization of target tokens," increasing Japanese pronunciation accuracy from 0.668 to 0.958 (+39%) and reducing CER by 54%.

## Background & Motivation
**Background**: LLM-based TTS (such as CosyVoice2 and F5-TTS) has widely adopted DPO-series preference optimization (Zhang et al. 2025, Tian et al. 2025) to improve intelligibility and speaker similarity, bypassing the hard rules of traditional G2P converters.

**Limitations of Prior Work**: Two major bottlenecks exist for ambiguous pronunciation scenarios (e.g., Japanese "辛い," heteronyms in Chinese, personal/place names): (i) **Mandatory paired data**: DPO requires samples containing both "correct" and "incorrect" pronunciations for the same sentence. However, actual TTS outputs are often "all correct" or "all incorrect"; the paper notes that 89.5% of sentences have only single-sided samples, while only 10.5% have complete pairs. (ii) **Utterance-level optimization**: Pronunciation issues are inherently char/token-level, but DPO treats the entire utterance as a single label, diluting the target signal across hundreds of tokens.

**Key Challenge**: Fine-grained signals (which should be at the token level and usable with unpaired data) vs. existing preference optimization (utterance-level + mandatory pairing). The latter directly limits sample efficiency and alignment precision.

**Goal**: (i) Eliminate the requirement for paired data to utilize more datasets; (ii) automatically identify tokens that "truly determine pronunciation correctness" without token-level annotations and increase their weights.

**Key Insight**: The authors leverage the inherent support for unpaired data in KTO (Kahneman-Tversky Optimization). They then differ the implicit rewards of tokens by training two KTO models with inverted labels—a classic approach for token-level reward estimation using DPO/KTO formulas (Rafailov et al. 2024)—but apply it to "estimate weights first, then perform weighted KTO."

**Core Idea**: Use a KTO contrastive pair ($\pi^+ / \pi^-$) for token-level reward estimation → apply exp to this reward to serve as token weights → integrate these into a weighted token-level KTO loss, concentratrating alignment pressure on key tokens in an end-to-end manner.

## Method

### Overall Architecture
A two-step process (see Figure 2 in the original paper). **Step 1: Targeted Token Weight Estimation** — Train two contrasting LLMs with non-shared parameters: $\pi^+$ (using original desirable/undesirable labels) and $\pi^-$ (with flipped labels: desirable ↔ undesirable). The token-level weight $w_t$ is estimated using their log-ratio. **Step 2: Token-level KTO** — Decompose the utterance-level sigmoid value of KTO into each token position to obtain $v_t$, then use $w_t$ for a weighted sum as the final loss. The backbone is CosyVoice2 (0.5B), where only this PO layer is trained while the vocoder remains frozen.

### Key Designs

1.  **KTO Contrastive LLM + Token-level Importance Estimation**:
    - **Function**: Automatically identifies "which token is key to determining desirable/undesirable labels" without token-level labels.
    - **Mechanism**: After training $\pi^+$ and $\pi^-$, calculate $w_t = \exp\left(\mu\cdot \text{clamp}\left(\log\frac{\pi^+(y_t\mid x, y_{<t})}{\pi^-(y_t\mid x, y_{<t})}, L, U\right)\right)$ for each generated token, where $\mu>0$ for desirable samples and $\mu<0$ for undesirable samples, and $[L,U]$ is the clipping range. Intuitively, if a token has a high probability under the "positive model" and low under the "negative model," it is crucial for distinguishing quality, thus its weight is increased. If the probabilities are similar, the token is deemed preference-neutral, and the weight nears 1.
    - **Design Motivation**: Manual annotation of token-level preferences is extremely costly, and paired data is scarce. Using contrastive KTO with "same data + flipped labels" allows distilling a token-level reward signal in an unpaired setting without extra annotation. Empirically, for the target character "辛," the reward for the desirable token was 0.22 (higher than the average 0.12), while the undesirable token was -1.54, resulting in a 12.8× automatic amplification of the target token weight.

2.  **Token-level KTO Value Function**:
    - **Function**: Decomposes the KTO sigmoid value function from the utterance level to the token level.
    - **Mechanism**: The reward for each token $y_t$ is defined as $r_{\theta,t}(x,y)=\log\frac{\pi_\theta(y_t\mid x, y_{<t})}{\pi_{\text{ref}}(y_t\mid x, y_{<t})}$, with a reference baseline $z_{0,t}=\mathrm{KL}(\pi_\theta(\cdot\mid x,y_{<t})\|\pi_{\text{ref}}(\cdot\mid x,y_{<t}))$ (microbatch estimation, no gradient backpropagation). Token-level value: $v_t = \lambda_D\sigma(\beta(r_{\theta,t}-z_{0,t}))$ if $y$ is desirable; $v_t = \lambda_U\sigma(\beta(z_{0,t}-r_{\theta,t}))$ if undesirable.
    - **Design Motivation**: Original KTO sums the entire reward at the utterance level before applying the sigmoid, which is equivalent to "averaging before non-linearity," drowning out the contribution of key tokens. In token-level decomposition, the sigmoid saturates independently at each position, preventing strong signal tokens from being flattened by weak signal tokens.

3.  **Weighted Token KTO Objective**:
    - **Function**: Combines $w_t$ and $v_t$ into a simple summation loss for end-to-end training.
    - **Mechanism**: $\mathcal{L}_{\text{TKTO}} = \mathbb{E}_{(x,y)}\left[-\sum_{t=1}^{|y|} w_t \cdot v_t(x,y)\right]$. During the forward pass, the two contrastive LLMs are frozen and used only to calculate $w_t$; the backward pass updates only the policy $\pi_\theta$.
    - **Design Motivation**: Completely decouples "importance estimation" from "preference optimization." The former is a one-time pre-computation (10 minutes on 8×A100 in the paper), while the latter is a standard KTO with an additional positional weight, resulting in minimal implementation overhead with fine-grained benefits.

### Loss & Training
Contrastive LLMs are trained using standard KTO; the TKTO stage freezes contrastive models and uses pre-computed $w_t$ caches. The base model is CosyVoice2 (0.5B), fine-tuned on 20K hours of Japanese TTS. For data construction, 5 male and 5 female candidates are generated per text; the best (desirable) and worst (undesirable) are selected based on pronunciation accuracy and CER (calculated via whisper-v3-large).

## Key Experimental Results

### Main Results
On a Japanese test set of 5,000 sentences containing "辛い," Acc (accuracy of the target character), CER, and Bad (ratio of CER > 0.3) were reported for female/male speakers. Selected data from Table 1:

| Model / Data Type | Female Acc ↑ | Female CER ↓ | Male Acc ↑ | Male CER ↓ |
|:---|---:|---:|---:|---:|
| Base CosyVoice2 (Du et al. 2024) | 0.683 | 0.128 | 0.668 | 0.138 |
| SFT (desirable only) | 0.674 | 0.119 | 0.654 | 0.130 |
| DPO (paired) | 0.706 | 0.120 | 0.693 | 0.130 |
| KTO (paired) | 0.654 | 0.066 | 0.651 | 0.074 |
| KTO (unpaired) | 0.933 | 0.079 | 0.952 | 0.087 |
| **TKTO (paired)** | 0.681 | 0.059 | 0.701 | 0.066 |
| **TKTO (unpaired)** | **0.949** | **0.075** | **0.958** | 0.085 |
| F5-TTS / F5-TTS+G2P (Ref.) | 0.498–0.500 | 0.136–0.177 | 0.500 | 0.146–0.177 |
| gpt-4o-mini-tts (Industry Baseline) | 0.900 | 0.109 | 0.939 | 0.111 |

TKTO (unpaired) achieved the highest Acc and lowest CER, outperforming closed-source industrial models like gpt-4o-mini-tts and gemini-2.5-pro-preview-tts on both metrics.

### Ablation Study

| Configuration | Acc / CER Trend | Description |
|:---|:---|:---|
| TKTO (unpaired) | 0.949 / 0.075 (F), 0.958 / 0.085 (M) | Full Method |
| TKTO (paired, 10.5% data) | 0.681 / 0.059, 0.701 / 0.066 | Matches DPO Acc with 6× less data |
| KTO (unpaired) | 0.933 / 0.079, 0.952 / 0.087 | Removing token weighting drops Acc by 0.6-1.6 pt |
| KTO (paired) | 0.654 / 0.066, 0.651 / 0.074 | Scarcity of paired data leads to Acc lower than base |
| SFT desirable-only | 0.674 / 0.119, 0.654 / 0.130 | No contrastive signal; CER shows no improvement |

NMOS subjective scores (Table 2): Base 4.09 < KTO 4.17 < TKTO 4.21. ABX preference tests (Figure 6) also favored TKTO.

### Key Findings
- **Token-level weighting is more valuable than data volume**: Adding token weights to the same unpaired data yields a 0.6–1.6 pt Acc gain over vanilla KTO while simultaneously reducing CER. This proves that directing weights to pronunciation-critical tokens provides stable improvements even without additional data.
- **Training dynamics**: TKTO only increases the log-likelihood of desirable tokens (Figure 3), whereas SFT simultaneously increases the likelihood of undesirable tokens. This indicates that TKTO gradients are more focused and safer.
- **Automatic target character localization**: Without explicitly being told that "辛" is the key character, the model assigned an average weight 12.8× higher than the sentence average to that position. Case studies show weights for other kanji remain near 1, proving the $\pi^+/\pi^-$ differencing effectively performs implicit token attribution.
- **Paired data as a bottleneck**: DPO is limited to 1.5K pairs, whereas unpaired KTO/TKTO can utilize 9K. This 6× data dividend directly translates to a leap from 0.668 to 0.958 in Acc.

## Highlights & Insights
- Training a second model with flipped labels and using the log-ratio as a token reward is an elegantly simple trick. It eliminates the need for manual token preference labeling and external reward models, maximizing the utility of KTO's implicit reward properties.
- Decoupling "weight estimation" from "preference optimization" makes the method orthogonal to downstream PO algorithms. $w_t$ can potentially be applied to DPO, IPO, or ORPO for similar token-level extensions.
- In TTS, a field long dominated by G2P rules, this method allows the LLM to learn which characters are most prone to mispronunciation and emphasizes them during training. This self-attributed curriculum idea has transfer potential to any task where local output determines overall quality (e.g., speech, OCR, code).

## Limitations & Future Work
- Evaluation is limited to Japanese, a single target character ("辛"), 5,000 sentences, and three annotators. It lacks validation for cross-lingual generalization or multi-character ambiguities.
- The contrastive LLM training assumes a "same data, flipped labels" hypothesis. If the boundary between desirable/undesirable is fuzzy, $\pi^-$ may fail to learn stable signals. clamping values $[L,U]$ and $\mu$ rely on manual tuning.
- Only the TTS decoder is tuned; vocoder and text G2P are not part of the end-to-end optimization, potentially leaving residual biases from the upstream G2P.
- NMOS and ABX only compared Base, KTO, and TKTO, without subjective blind tests against closed-source models like gpt-4o-mini-tts.

## Related Work & Insights
- **vs. DPO (Tian et al. 2025)**: Also performs TTS preference optimization but requires paired data, resulting in 1/6 the data efficiency of TKTO. TKTO matches/slightly exceeds DPO in paired settings and significantly leads in unpaired settings.
- **vs. vanilla KTO (Ethayarajh et al. 2024)**: Only calculates values at the utterance level. TKTO improves Acc by 1–2 pts on the same unpaired data by decomposing to the token level and learning token weights.
- **vs. G2P-based Hard Rules (Oura et al. 2010 + F5-TTS)**: Even with G2P assistance, F5-TTS Acc remains at 0.500, indicating that dictionary-based approaches cannot resolve polysemic kanji disambiguation. TKTO learns disambiguation within the LLM's context.
- **vs. Liu et al. 2025's token-level importance sampling**: Shares the idea of using log-ratios for token rewards, but this work constructs $\pi^\pm$ using the KTO framework, which is better suited for the unpaired realities of TTS.

## Rating
- Novelty: ⭐⭐⭐⭐ The two-stage design using contrastive KTO for token attribution and weighted KTO is simple yet represents a systematic application for TTS.
- Experimental Thoroughness: ⭐⭐⭐ Includes both objective and subjective evaluations, though lacks multi-lingual expansion. Ablations cover key variables.
- Writing Quality: ⭐⭐⭐⭐ Mathematical derivations are clear; case studies are intuitive.
- Value: ⭐⭐⭐⭐ 6× data dividend + 39% accuracy improvement offers direct reference value for industrial TTS alignment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Semi-Supervised Diseased Detection from Speech Dialogues with Multi-Level Data Modeling](semi-supervised_diseased_detection_from_speech_dialogues_with_multi-level_data_m.md)
- [\[ICLR 2026\] AVERE: Improving Audiovisual Emotion Reasoning with Preference Optimization](../../ICLR2026/audio_speech/avere_improving_audiovisual_emotion_reasoning_with_preference_optimization.md)
- [\[ICML 2026\] Sparse Tokens Suffice: Jailbreaking Audio Language Models via Token-Aware Gradient Optimization](../../ICML2026/audio_speech/sparse_tokens_suffice_jailbreaking_audio_language_models_via_token-aware_gradien.md)
- [\[ACL 2026\] Do We Need Distinct Representations for Every Speech Token? Unveiling and Exploiting Redundancy in Large Speech Language Models](do_we_need_distinct_representations_for_every_speech_token_unveiling_and_exploit.md)
- [\[ICLR 2026\] VowelPrompt: Hearing Speech Emotions from Text via Vowel-level Prosodic Augmentation](../../ICLR2026/audio_speech/vowelprompt_hearing_speech_emotions_from_text_via_vowel-level_prosodic_augmentat.md)

</div>

<!-- RELATED:END -->
