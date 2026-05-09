---
title: >-
  [Paper Note] Adversarial Paraphrasing: A Universal Attack for Humanizing AI-Generated Text
description: >-
  [NeurIPS 2025][LLM Safety][AI text detection] This paper proposes Adversarial Paraphrasing — a training-free universal attack framework that selects the most "human-like" token at each decoding step by leveraging feedback signals from AI text detectors during token-by-token paraphrasing. The approach achieves an average T@1%F reduction of 87.88% across 8 detectors and exhibits strong cross-detector transferability.
tags:
  - NeurIPS 2025
  - LLM Safety
  - AI text detection
  - adversarial attack
  - paraphrasing
  - watermark evasion
  - controllable text generation
date: 2026-05-08
content_hash: 5ebcc4559c54dd36
---

# Adversarial Paraphrasing: A Universal Attack for Humanizing AI-Generated Text

**Conference**: NeurIPS 2025
**arXiv**: [2506.07001](https://arxiv.org/abs/2506.07001)
**Code**: [chengez/Adversarial-Paraphrasing](https://github.com/chengez/Adversarial-Paraphrasing)
**Authors**: Yize Cheng, Vinu Sankar Sadasivan, Mehrdad Saberi, Shoumik Saha, Soheil Feizi (UMD)
**Area**: AI Security
**Keywords**: AI text detection, adversarial attack, paraphrasing, watermark evasion, controllable text generation

## TL;DR

This paper proposes Adversarial Paraphrasing — a training-free universal attack framework that selects the most "human-like" token at each decoding step by leveraging feedback signals from AI text detectors during token-by-token paraphrasing. The approach achieves an average T@1%F reduction of 87.88% across 8 detectors and exhibits strong cross-detector transferability.

## Background & Motivation

**Background**: Large models such as GPT, Gemini, and LLaMA can generate fluent and coherent text, introducing security risks including plagiarism and social engineering attacks.

**Limitations of Prior Work**: Three major categories of AI text detectors have emerged: neural-network-based trained classifiers (OpenAI-RoBERTa, RADAR, MAGE), zero-shot detectors (Fast-DetectGPT, GLTR), and watermarking schemes (KGW, Unigram, SynthID). Although early detectors could be defeated by single-pass or recursive paraphrasing, adversarially trained detectors such as RADAR have become robust against naive paraphrasing — simple paraphrasing even increases RADAR's detection rate (T@1%F +8.57%).

**Key Challenge**: Prior attacks lack generality: DIPPER requires a dedicated T5 paraphrasing model, while Watermark Stealing targets only watermark-based detectors and does not transfer to other detection paradigms.

**Goal**: To design a universal, training-free attack framework capable of reliably evading diverse AI text detectors without prior knowledge of the deployed detection scheme.

**Key Insight**: Works such as PPLM and BEAST demonstrate that incorporating external classifier guidance signals at decoding time can control text attributes; this paper applies the same principle to "humanizing" paraphrasing.

## Method

### Overall Architecture

Adversarial Paraphrasing consists of two off-the-shelf modules and requires no additional training:

- **Paraphraser**: LLaMA-3-8B-Instruct, prompted via a custom system prompt to function as a paraphrasing model.
- **Guidance Detector**: Any trained AI text classifier (e.g., OpenAI-RoBERTa-Large), used to provide an "AI score" at each decoding step.

**Workflow**: The original AI-generated text is fed to the paraphraser, which autoregressively generates a paraphrased output token by token. At each step, instead of sampling directly, candidate tokens are scored by the guidance detector, and the token that yields the lowest AI score for the generated sequence so far is selected.

### Key Designs

1. **Top-p/Top-k Candidate Filtering**: At each decoding step, candidate tokens are first filtered from the paraphraser's logit distribution using top-p ($p=0.99$) and top-k ($k=50$) to ensure semantic and grammatical plausibility.
2. **Detector-Guided Greedy Selection**: Each candidate token is appended to the already-generated sequence and passed through the guidance detector to obtain an AI score; the candidate with the lowest score (i.e., most human-like) is selected. This is equivalent to a depth-1 beam search optimizing the detector score.
3. **System Prompt Engineering**: A carefully designed system prompt converts a general instruction-following LLM into a paraphraser, enabling any instruction-tuned LLM to serve as the paraphrasing module without fine-tuning.

### Loss & Training

The method involves no explicit loss function. Instead, it performs objective-guided decoding at inference time:

$$y^* = \arg\min_{c \in \text{candidates}} \mathcal{D}(y_{:m} \oplus c)$$

where $\mathcal{D}$ is the AI score output of the guidance detector, $y_{:m}$ is the already-generated sequence, and $c$ is a candidate token. By minimizing the AI score at each step, the overall paraphrased text is driven toward the statistical distribution of human-written text.

## Key Experimental Results

### Experimental Setup

- **Dataset**: MAGE dataset (2,000 AI-generated texts + 2,000 human-written texts, 100–500 tokens); watermark experiments use LLaMA-3.1-8B to generate watermarked text.
- **Detectors**: 8 in total — 4 trained classifiers (OpenAI-RoBERTa-Base/Large, MAGE, RADAR), 2 watermarking schemes (KGW, Unigram), 2 zero-shot detectors (Fast-DetectGPT, GLTR).
- **Baselines**: Simple paraphrasing, recursive paraphrasing (2 and 3 rounds).

### Main Results (Table 2 Summary)

| Deployed Detector | No Attack T@1%F | Simple Para. T@1%F | AdvPara (RoBERTa-L) T@1%F | Relative Drop |
|---|---|---|---|---|
| RoBERTa-Large | 0.163 | 0.096 | **0.000** | −100% |
| RoBERTa-Base | 0.182 | 0.119 | **0.000** | −100% |
| MAGE | 0.768 | 0.616 | **0.142** | −81.5% |
| RADAR | 0.124 | 0.140 ↑ | **0.044** | −64.5% |
| KGW Watermark | 1.000 | 0.295 | **0.131** | −86.9% |
| Unigram Watermark | 0.999 | 0.609 | **0.294** | −70.6% |
| Fast-DetectGPT | 0.323 | 0.326 ↑ | **0.003** | −99.0% |
| GLTR | 0.174 | 0.049 | **0.001** | −99.4% |

### Key Findings

1. **Universal Evasion**: Using OpenAI-RoBERTa-Large as the guidance detector, the average T@1%F across all 8 deployed detectors drops by **87.88%** relative to no attack, far surpassing simple paraphrasing (30.27%).
2. **Transfer Matrix**: Any trained classifier used as the guidance detector effectively evades all 7 other detectors (including watermarking and zero-shot methods), confirming the universal transferability of the attack.
3. **Counterproductive Effect of Simple Paraphrasing**: Detection rates for RADAR and Fast-DetectGPT actually increase after simple paraphrasing (+8.57% and +15.03%, respectively), demonstrating that adversarial training renders these detectors immune to naive paraphrasing.
4. **Text Quality Preservation**: GPT-4o automatic evaluation shows that adversarial paraphrasing quality (4.48±0.77) is close to simple paraphrasing (4.75±0.54), with only a minor degradation, and substantially outperforms 3-round recursive paraphrasing (4.26±0.74).

## Highlights & Insights

- **Training-Free and Universal**: Only an off-the-shelf LLM and an off-the-shelf detector are required — no model fine-tuning is needed — yet the attack is effective against all three categories of detectors (8 in total).
- **Elegant Design**: The method elegantly combines controllable text generation with adversarial attack by using detector signals to guide token selection at the decoding level; a depth-1 beam search already proves sufficient.
- **Convincing Intuition for Transferability**: Different detectors tend to converge on the same boundary of the "human text distribution," so a paraphrase that evades one detector naturally evades others.
- **Comprehensive Evaluation**: Experiments cover trained classifiers, watermarking schemes, and zero-shot detectors, with a complete transfer matrix and text quality analysis.

## Limitations & Future Work

- **Inference Efficiency**: Each token decoding step requires up to $k=50$ detector forward passes, making generation far slower than standard decoding and costly for practical deployment.
- **Reliance on Trained Guidance Detectors**: The current work only validates the use of trained classifiers as guidance signals; the feasibility of using zero-shot detectors or watermark detection signals as guidance has not been explored.
- **Quality–Attack Effectiveness Trade-off**: Although overall quality degradation is modest, adversarial paraphrasing may introduce unnatural expressions in some samples; quality degradation on long-form text has not been thoroughly analyzed.
- **Limited Evaluation Text Length**: Experiments are primarily conducted on short texts of 100–500 tokens; attack effectiveness on long documents (papers, reports) remains unclear.
- **Insufficient Defense Perspective**: The paper focuses mainly on attack capability and provides limited discussion on how to build more robust detectors (e.g., the arms race between adversarial training and adversarial paraphrasing).

## Related Work & Insights

- **AI Text Detection**: OpenAI-RoBERTa (Solaiman et al.); RADAR (Hu et al.), which enhances robustness via adversarial training; MAGE (Li et al.), which improves generalization with diverse datasets; DetectGPT/Fast-DetectGPT, which exploit log-probability curvature for zero-shot detection; KGW/Unigram/SynthID watermarking schemes.
- **Detector Attacks**: Recursive paraphrasing attacks and theoretical impossibility analysis by Sadasivan et al.; DIPPER (Krishna et al.), a T5-based paraphrasing model; Watermark Stealing (Jovanovic et al.), targeting watermark-based detectors.
- **Controllable Text Generation**: PPLM (Dathathri et al.), which uses attribute classifier gradients to guide decoding; BEAST, which uses beam search to generate adversarial prompts; InstructCTG, which controls generation via natural language instructions. The proposed method is gradient-free and thus simpler and more efficient.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The idea of incorporating detector feedback into paraphrasing decoding is original; the training-free, gradient-free implementation is elegantly simple.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers 8 detectors across 3 major categories, with a complete transfer matrix and text quality analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — Structure is clear; algorithm pseudocode and visualizations are well-presented; the intuition for transferability is accessible.
- **Value**: ⭐⭐⭐⭐ — Raises an important security alarm for the AI text detection field; the universality and transferability of the attack provide valuable reference for the defense side.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] DNA-DetectLLM: Unveiling AI-Generated Text via a DNA-Inspired Mutation-Repair Paradigm](dna-detectllm_unveiling_ai-generated_text_via_a_dna-inspired_mutation-repair_par.md)
- [\[NeurIPS 2025\] Stop DDoS Attacking the Research Community with AI-Generated Survey Papers](stop_ddos_attacking_the_research_community_with_ai-generated_survey_papers.md)
- [\[NeurIPS 2025\] AgentStealth: Reinforcing Large Language Model for Anonymizing User-generated Text](agentstealth_reinforcing_large_language_model_for_anonymizing_user-generated_tex.md)
- [\[CVPR 2026\] V-Attack: Targeting Disentangled Value Features for Controllable Adversarial Attacks on LVLMs](../../CVPR2026/llm_safety/v-attack_targeting_disentangled_value_features_for_controllable_adversarial_atta.md)
- [\[NeurIPS 2025\] Virus Infection Attack on LLMs: Your Poisoning Can Spread "VIA" Synthetic Data](virus_infection_attack_on_llms_your_poisoning_can_spread_via_synthetic_data.md)

<!-- RELATED:END -->
