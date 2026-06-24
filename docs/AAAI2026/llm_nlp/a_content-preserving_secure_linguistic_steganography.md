---
title: >-
  [Paper Note] A Content-Preserving Secure Linguistic Steganography
description: >-
  [AAAI 2026][LLM (Other)][linguistic steganography] This paper proposes CLstega, the first content-preserving linguistic steganography paradigm, which embeds secret information into an unmodified cover text by fine-tuning a masked language model (MLM) to controllably transform its prediction distribution. The approach achieves a 100% extraction success rate and near-perfect security, with steganalysis detection accuracy approaching the random-guess baseline of 0.5.
tags:
  - "AAAI 2026"
  - "LLM (Other)"
  - "linguistic steganography"
  - "content preservation"
  - "masked language model"
  - "distribution transformation"
  - "secure communication"
date: 2026-05-08
content_hash: 96ce0af08f27372c
---

# A Content-Preserving Secure Linguistic Steganography

**Conference**: AAAI 2026
**arXiv**: [2511.12565](https://arxiv.org/abs/2511.12565)  
**Code**: None  
**Area**: LLM/NLP
**Keywords**: linguistic steganography, content preservation, masked language model, distribution transformation, secure communication

## TL;DR
This paper proposes CLstega, the first content-preserving linguistic steganography paradigm, which embeds secret information into an unmodified cover text by fine-tuning a masked language model (MLM) to controllably transform its prediction distribution. The approach achieves a 100% extraction success rate and near-perfect security, with steganalysis detection accuracy approaching the random-guess baseline of 0.5.

## Background & Motivation
Linguistic steganography (LS) conceals secret messages within natural language text, exploiting everyday communication as cover for covert transmission. Existing methods fall into two categories: **modification-based (MLS)**, which alters the original text via synonym substitution or syntactic transformation, and **generative (GLS)**, which embeds information by controlling word selection during automatic text generation. Neither category can fully eliminate statistical, semantic, or perceptual discrepancies between stego text and normal text. Even distribution-preserving methods such as Discop produce text whose distribution diverges measurably from natural text, rendering it detectable by advanced steganalysis tools.

The key insight is that any alteration to textual content—whether by modification or generation—inevitably introduces detectable artifacts. This motivates the question: can secret information be embedded **without modifying the cover text at all**?

## Core Problem
How can different secret messages be embedded in the same text without changing a single token, while still enabling reliable extraction? Under conventional thinking this appears impossible—an unchanged text seemingly cannot carry different information. The central breakthrough of this paper is: rather than altering the text itself, the paper alters the encoding/decoding function used to *interpret* the text.

## Method

### Overall Architecture
CLstega comprises three core modules: **Augmented Masking**, **Dynamic Distribution Steganographic Coding (DDSC)**, and **Controllable Distribution Transformation**.

The overall pipeline proceeds as follows. Given a cover text, part-of-speech tagging first identifies suitable embedding positions (non-function words). An MLM then obtains the original prediction distribution at these positions. Target distributions are constructed according to the secret bit sequence to be embedded, and carefully selected label words are used to build training samples for fine-tuning the MLM, so that its predictions at embedding positions conform to the target distributions. The fine-tuned MLM serves as the shared secret key—both sender and receiver hold the same fine-tuned model, and the receiver extracts the secret message by running inference on the original unmodified text.

### Key Designs

1. **Augmented Masking Strategy**:
   The strategy involves two stages: *localization* and *masking*. During localization, a POS tagger identifies non-function words (nouns, verbs, etc.) in the sentence, as these positions exhibit higher entropy and more flexible distributions amenable to manipulation. During masking, the paper proposes the **Single-Position Augmented Masking (SPAM)** strategy: rather than masking all embedding positions simultaneously (Full-Position Masking, FPM), SPAM creates an independent masked copy for each embedding position, masking only one position per copy. Each copy thus retains $l-1$ contextual tokens, enabling more accurate MLM predictions and substantially improving distribution transformation success.

2. **Dynamic Distribution Steganographic Coding (DDSC)**:
   This is the core mechanism for mapping the *same token* to *different secret bits*. The encoding rule is elegantly simple: for each embedding position, the MLM produces a probability distribution $P$ over the vocabulary. If the original word $w$ is the highest-probability token (rank 1), the position is encoded as `0`; otherwise it is encoded as `1`. By controlling whether $w$ ranks first or not in the distribution, one bit of information is encoded per position. Critically, the text tokens remain unchanged; only the MLM's prediction distribution changes.

3. **Controllable Distribution Transformation**:
   To make the MLM's prediction distribution conform to the target encoding, fine-tuning is applied to "transform the distribution." Specifically: (1) if encoding `0` is required (original word must rank first), the original word itself is used as the label; (2) if encoding `1` is required but the original word currently ranks first, the second-ranked candidate is used as the label to displace the original word; (3) if encoding `1` is required and the original word already does not rank first, the current top-ranked word is used as the label to reinforce its position. All label–masked-sentence pairs constitute the training set, and the MLM is fine-tuned with cross-entropy loss.

### Loss & Training
- **Loss function**: Standard cross-entropy loss measuring the discrepancy between the MLM's predicted distribution at masked positions and the target word label
- **Optimizer**: AdamW, weight decay 0.01, initial learning rate $5 \times 10^{-5}$
- **Training configuration**: batch size 32, FP16 mixed-precision training
- **MLM backbone**: bert-base-cased (HuggingFace pretrained version)
- Notably, each communication session requires a dedicated fine-tuning pass for the specific cover text and secret message; the resulting fine-tuned MLM serves as the shared key

## Key Experimental Results

| Dataset/Setting | Metric | Ours (CLstega) | Prev. SOTA | Gain |
|---|---|---|---|---|
| CC-100 (BiLSTM-Dense detector) | Acc/F1 | 0.4955/0.5070 | CPGLS: 0.5130/0.5375 | Closer to 0.5 (perfect security) |
| CC-100 (SeSy detector) | Acc/F1 | 0.5038/0.4968 | Discop: 0.5032/0.5095 | On par / superior |
| CC-100 (HiDuNet detector) | Acc/F1 | 0.5012/0.4924 | CPGLS: 0.5390/0.5035 | Significantly more secure |
| CC-100 | PPL (perplexity) | **70.16** | CPGLS: 82.55 | −15% |
| CC-100 | ER (embedding rate) | 0.4204 | ARLS: 0.2542 | +65% vs. MLS methods |
| CC-100 (AW, $k$=all) | ER | **0.9538** | — | Near theoretical maximum 1.0 |
| All settings | ESR (extraction success rate) | **100%** | — | Perfect extraction |

### Ablation Study
- **Masking strategy**: SPAM converges faster than FPM and achieves higher ESR under the same number of epochs. SPAM reaches 100% ESR more quickly across $k=2, 4, 8,$ all
- **Number of embedding positions $k$**: larger $k$ increases embedding capacity but requires more fine-tuning epochs to converge to 100% ESR
- **Localization strategy**: non-function words (NFW), function words (FW), and arbitrary words (AW) yield similar embedding rates at fixed $k$; AW + $k$=all achieves the highest ER of 0.9538
- **Extraction efficiency**: FPM extraction has time complexity $O(N)$, independent of $k$; SPAM requires $O(kN)$, with efficiency degrading at large $k$

## Highlights & Insights
- **Paradigm-level innovation**: The paper is the first to propose a content-preserving linguistic steganography paradigm, fundamentally eliminating the discrepancy between stego text and cover text, since they are identical
- **Theoretical guarantee of perfect security**: Because stego text $=$ cover text, the KL divergence is strictly zero, making the two indistinguishable across statistical, semantic, syntactic, and perceptual dimensions
- **Elegant encoding idea**: Rather than modifying text, the method modifies the *interpretation*—encoding and decoding are realized by controlling the MLM's prediction distribution, a conceptually clean design
- **100% extraction success rate**: Perfect extraction is achieved across all experimental settings
- In the case study, CLstega is the only method whose stego text is identical to the cover text, with fully matching perplexity (58.70)

## Limitations & Future Work
- **High computational cost**: Each communication session requires fine-tuning the MLM for the specific text and message, making the embedding process non-real-time
- **Limited embedding capacity**: The current encoding rule embeds only 1 bit per position (rank-1 vs. non-rank-1), with a theoretical maximum of 1 bpw (bit per word)
- **Key distribution challenge**: The fine-tuned MLM itself constitutes the secret key; securely and efficiently distributing this "large key" poses a practical deployment challenge
- **Robustness not validated**: The paper does not investigate extraction robustness under minor text perturbations (e.g., OCR noise, formatting changes)
- **Extension to multi-bit encoding**: More fine-grained encoding rules (e.g., assigning multiple bits based on rank intervals) could be designed to increase embedding capacity

## Related Work & Insights
- **vs. CPGLS** (modification-based SOTA): CPGLS employs a CNN-based causality-aware network to select safe embedding positions and performs word substitution, achieving reasonable security (Acc ≈ 0.51). CLstega surpasses it (Acc ≈ 0.50) by leaving the text entirely intact. CPGLS's embedding rate of 0.108 is far below CLstega's 0.42.
- **vs. Discop** (generative SOTA): Discop maintains distributional consistency in generated text by replicating probability distributions, achieving a substantially higher embedding rate of 5.53, but at the cost of PPL = 86.33 and F1 = 0.548 under HiDuNet detection, indicating inferior security compared to CLstega.
- **vs. traditional steganography (image/audio)**: The unique challenges of linguistic steganography stem from the discrete nature of text and its semantic sensitivity. CLstega elegantly circumvents these challenges by operating in *model space* rather than *text space*.

**Broader implications**:
- The **model-as-key** paradigm warrants deeper exploration: differences in fine-tuned model parameters can serve as information carriers, potentially inspiring new forms of covert communication.
- There are potential connections to **model watermarking**: both embed information without altering the surface-level output.
- From an adversarial perspective, this approach poses new challenges for steganalysis—when stego text and cover text are identical, text-level detection methods are entirely ineffective, and future detection may need to focus on the behavioral patterns of communicating parties.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — The paper is the first to propose the content-preserving paradigm, fundamentally shifting the approach from "modifying text" to "modifying interpretation"
- Experimental Thoroughness: ⭐⭐⭐⭐ — Security, embedding rate, PPL, extraction success rate, and ablation studies are all covered, but large-scale scenario testing and practical deployment evaluation are absent
- Writing Quality: ⭐⭐⭐⭐ — The logic is clear and the presentation progresses from intuition to technical detail, though the notation is dense
- Value: ⭐⭐⭐⭐ — The paradigm innovation is significant, but computational overhead and limited embedding capacity constrain practical applicability

## Supplementary Notes
- The methodology and experimental design of this work offer reference value for related areas.
- Future work may validate generalizability and scalability across broader scenarios and larger scales.
- Potential research value exists in combining this approach with recent advances (e.g., RL/MCTS or multimodal methods).
- Deployment feasibility and computational efficiency should be assessed against practical application requirements.
- The choice of datasets and evaluation metrics may affect the generality of conclusions; cross-validation on additional benchmarks is recommended.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] MOSAIC: Multiple Observers Spotting AI Content](../../ACL2025/llm_nlp/mosaic_multiple_observers_spotting_ai_content.md)
- [\[ACL 2025\] Robust Utility-Preserving Text Anonymization Based on Large Language Models](../../ACL2025/llm_nlp/robust_utility-preserving_text_anonymization_based_on_large_language_models.md)
- [\[ACL 2025\] Comparing Linguistic Acceptability Judgments of Autoregressive Language Models](../../ACL2025/llm_nlp/comparing_linguistic_acceptability_judgments_of_autoregressive_language_models.md)
- [\[ACL 2025\] Aligning Large Language Models with Implicit Preferences from User-Generated Content](../../ACL2025/llm_nlp/pugc_align_implicit_pref_ugc.md)
- [\[ACL 2025\] Can LLMs Interpret and Leverage Structured Linguistic Representations? A Case Study with AMRs](../../ACL2025/llm_nlp/can_llms_interpret_and_leverage_structured_linguistic_representations_a_case_stu.md)

</div>

<!-- RELATED:END -->
