---
title: >-
  [Paper Note] DualGuard: Dual-stream Large Language Model Watermarking Defense against Paraphrase and Spoofing Attack
description: >-
  [ACL 2026][LLM Safety][Dual-stream Watermarking] DualGuard introduces the first **dual-stream watermarking** mechanism: it adaptively injects different watermarks using two complementary standard/adversarial heads based…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "Dual-stream Watermarking"
  - "Spoofing Attack"
  - "Paraphrase Attack"
  - "Adversarial Attribution"
  - "Semantic-invariant Watermarking"
date: 2026-05-08
content_hash: 5aa778f5fa99aa8d
---

# DualGuard: Dual-stream Large Language Model Watermarking Defense against Paraphrase and Spoofing Attack

**Conference**: ACL 2026  
**arXiv**: [2512.16182](https://arxiv.org/abs/2512.16182)  
**Code**: https://github.com/hlee-top/DualGuard  
**Area**: LLM Security / Watermarking  
**Keywords**: Dual-stream Watermarking, Spoofing Attack, Paraphrase Attack, Adversarial Attribution, Semantic-invariant Watermarking

## TL;DR
DualGuard introduces the first **dual-stream watermarking** mechanism: it adaptively injects different watermarks using two complementary standard/adversarial heads based on whether content is "benign" or "malicious." This makes benign text consistent across both streams while malicious text diverges, maintaining robustness against paraphrasing while **enabling detection and attribution** of malicious segments injected via piggyback spoofing.

## Background & Motivation

**Background**: Mainstream LLM watermarking methods typically add a "greenlist/redlist" bias to logits (e.g., KGW, SIR, XSIR, EWD, SWEET) or introduce pseudo-random numbers in the sampling layer (e.g., AAR, SynthID, DIPmark) for post-hoc statistical traceability. Most existing works optimize for **paraphrase robustness**—ensuring watermarks remain identifiable even after word-level rewriting.

**Limitations of Prior Work**: The authors identify that "over-pursuing robustness" creates vulnerabilities to piggyback spoofing attacks. In such attacks, an adversary takes watermarked LLM output and **injects malicious/harmful content** (e.g., hate speech, misinformation). If the watermark survives, the harmful content is falsely attributed to the model provider, turning the watermark from a "shield" into "incriminating evidence." The only existing spoofing defense, An et al. (2025), can only mark contaminated text as "unwatermarked" without identifying which segment is malicious or tracing its source.

**Key Challenge**: To resist paraphrasing, watermarks must be insensitive to local edits; however, this insensitivity allows malicious spoofing injections to be preserved. In other words, **paraphrase robustness and spoofing detectability are naturally in conflict**.

**Goal**: To achieve four objectives within a single watermarking framework: (i) paraphrase robustness, (ii) spoofing detectability, (iii) attribution of malicious segments, and (iv) no degradation in text quality.

**Key Insight**: The authors observe that benign and malicious content are separable in embedding space. By maintaining "two complementary watermarking heads," consistency can be enforced in benign regions while divergence is induced in malicious regions—where the degree of divergence serves as the spoofing signal.

**Core Idea**: A standard head and an adversarial head are trained using **contrastive loss** to be cosine-similar on benign text and cosine-opposite on malicious text. During generation, the injection head is dynamically switched per window. During detection, dual-stream consistency is used to simultaneously determine the watermark, detect spoofing, and trace malicious sources.

## Method

### Overall Architecture
The input is the LLM token sequence $y_{:t}$ to be generated, and the output is the dual-stream watermarked text along with three scores during detection: watermark detection score $\text{Score}_{wd}$, spoofing detection score $\text{Score}_{sd}$, and spoofing attribution score $\text{Score}_{st}$. The process consists of three stages: (1) Offline training of a mapping model $\mathcal{G}$ to obtain a shared backbone and two watermarking heads $\Theta_s, \Theta_a$; (2) Decoding where the dual-stream cosine distance $\text{dist}(y_{:t})$ of the current prefix is calculated per window $k$. If distance $<\alpha$, $\Theta_s$ is used; otherwise, it switches to $\Theta_a$. The output is mapped through tanh + random projection to $|\mathcal{V}|$ dimensions and injected via $P_{\mathcal{M}'}^t = P_\mathcal{M}^t + \delta\cdot P_\mathcal{M}^t P_\Theta^t$; (3) Detection where head selection is replayed per window, and watermark detection, spoofing detection, and attribution are completed using average watermark logit, average dual-stream distance, and adversarial head hit rate, respectively.

### Key Designs

1.  **Dual-head Mapping Network $\mathcal{G}$ (Standard + Adversarial Heads)**:
    - **Function**: Maps the current token prefix embedding $e_t=\mathcal{E}(y_{t-\rho:t})$ simultaneously to two sets of watermark logits $\Theta_s(e_t), \Theta_a(e_t)$, serving as "dual-stream signals."
    - **Mechanism**: A shared multi-layer FFN with residual connections branches into two independent heads. To ensure "semantic invariance," each head minimizes a semantic loss $\mathcal{L}_{sem}(\Theta)$ consisting of three terms: ensuring similar embeddings produce similar watermarks (scaling cosine via $\phi(x)=\tanh(\tau(x-\bar{x}))$), balancing positive and negative values in single-sample watermarks across the vocabulary, and ensuring zero-mean watermark expectation across the dataset.
    - **Design Motivation**: Single-stream watermarks face a binary choice between detection and bypass. Dual heads allow each to maintain semantic invariance while carrying **complementary** information, providing a "reference frame" for spoofing detection.

2.  **Content-Sensitive Contrastive Loss $\mathcal{L}_{con}$**:
    - **Function**: Forces the two heads towards a cosine similarity of $\rightarrow +1$ on the benign subset $\mathcal{D}_s$ and $\le -\eta$ on the malicious subset $\mathcal{D}_a$ ("consistent on benign, divergent on malicious").
    - **Mechanism**: Minimizes $-\cos(\Theta_s(e_i),\Theta_a(e_i))$ for benign samples; uses a hinge loss $\max(0,\cos(\Theta_s,\Theta_a)+\eta)$ for malicious samples to push similarity below the separation margin $\eta$. The total loss $\mathcal{L}=\mathcal{L}_{sem}+\lambda\mathcal{L}_{con}$ balances semantic invariance and content sensitivity.
    - **Design Motivation**: Since spoofing involves inserting malicious segments into benign text, if the heads naturally diverge in malicious regions, the **dual-head distance itself serves as a training-free statistic for spoofing** without requiring an external classifier.

3.  **Window-level Adaptive Injection + Three-way Detection**:
    - **Function**: Performs head selection every $k$ tokens during decoding, encoding the "switching trajectory" into the text so the detector can reproduce the path and generate three scores.
    - **Mechanism**: The injector selects $\Theta=\Theta_s$ if $\text{dist}(y_{:t})<\alpha$ else $\Theta_a$ (window boundary decision). It projects $P_\Theta^t = F(\tanh(\gamma\Theta(e_t)))$ to the vocabulary and injects it multiplicatively $\delta\cdot P_\mathcal{M}^t P_\Theta^t$ to minimize distribution shift. The detector recovers the head sequence using the same rules, calculates $\text{Score}_{wd}=\text{mean}\,P_\Theta^t[y_t]$ for watermark detection and $\text{Score}_{sd}=\text{mean}\,\text{dist}(y_{:t})$ for spoofing detection. Finally, it calculates the hit rate for tokens covered by the adversarial head $\text{Score}_{st}=\frac{1}{N}\sum\mathbb{1}(P_\Theta^t[y_t]>0)$ for **attribution**: model-generated malicious content yields a high hit rate, while externally spoofed malicious content yields a low hit rate.
    - **Design Motivation**: Making the switch a function of the token sequence allows reproduction without external keys. Leveraging the natural difference in head matching between "self-generated" and "externally spoofed" malicious content converts attribution into a statistical threshold problem.

### Loss & Training
Two heads are trained jointly: $\mathcal{L}=\mathcal{L}_{sem}(\Theta_s)+\mathcal{L}_{sem}(\Theta_a)+\lambda\mathcal{L}_{con}$. $\mathcal{L}_{sem}$ includes cosine fitting, single-sample balancing, and dataset-level unbiasedness terms. $\mathcal{L}_{con}$ uses contrastive loss with margin $\eta$ to push the malicious subset away. Key injection hyperparameters: window $k$, threshold $\alpha$, scaling $\gamma$, strength $\delta$, and prefix length $\rho$. The mapping network is trained once and serves multiple backbones (OPT-1.3B, Llama-3.1-8B-Instruct, etc.) via random projection $F(\cdot)$.

## Key Experimental Results

### Main Results
Paraphrase robustness ($\text{Robustness}_{para}$) and spoofing robustness ($\text{Robustness}_{spoof}$) were evaluated on RealNewsLike (C4) and BookSum. The table below shows Overall AUC on OPT-1.3B.

| Dataset | Metric | KGW | SWEET | SIR | XSIR | **Ours** |
|--------|------|------|-------|------|------|---------------|
| RealNewsLike | Para AUC | 0.9871 | 0.9731 | 0.9235 | 0.9224 | 0.9680 |
| RealNewsLike | Spoof AUC | 0.5141 | 0.5730 | 0.4190 | 0.4300 | **0.9284** |
| RealNewsLike | Overall AUC | 0.7506 | 0.7730 | 0.6713 | 0.6762 | **0.9482** |
| BookSum | Para AUC | 0.9777 | 0.9849 | 0.9306 | 0.9601 | 0.9760 |
| BookSum | Spoof AUC | 0.4613 | 0.5136 | 0.4190 | 0.3882 | **0.9552** |
| BookSum | Overall AUC | 0.7195 | 0.7492 | 0.6748 | 0.6741 | **0.9656** |

On Llama-3.1-8B-Instruct, Spoof AUC jumped from $\le 0.57$ for all baselines to **0.9159 / 0.9354**, with Overall AUC $\ge 0.92$. All baselines degraded to near-random (AUC ≈ 0.5) under spoofing attacks, confirming the fragility of over-pursuing paraphrase robustness.

### Ablation Study

| Configuration | Para AUC | Spoof AUC | Description |
|------|---------|-----------|------|
| Full DualGuard | 0.9680 | **0.9284** | Dual-head + contrastive + adaptive |
| Single-stream (Only $\Theta_s$) | 0.9235 | 0.4190 | Degrades to traditional semantic-invariant; Spoof AUC becomes random |
| Single-stream (Only $\Theta_a$) | 0.9224 | 0.4300 | Same as above; adversarial head cannot detect spoof without comparison |
| w/o Contrastive Loss $\mathcal{L}_{con}$ | ≈Para Unchanged | Near 0.5 | Heads do not diverge; spoofing signal disappears |

### Key Findings
- **Dual heads are critical**: Removing either head drops Spoof AUC from 0.92+ to 0.4–0.5, showing spoofing detection stems from "dual-stream difference" rather than single-watermark properties.
- **Negligible paraphrase performance loss**: Para AUC remains competitive with strong baselines like SWEET/SIR, indicating the dual-stream mechanism does not sacrifice existing robustness.
- **Cross-model transferability**: The same $\mathcal{G}$ serves multiple models without retraining, thanks to $F(\cdot)$ adapting to any vocabulary.
- **High attribution accuracy**: $\text{Score}_{st}$ shows a bimodal distribution between "self-generated" and "externally spoofed" malicious content, enabling the first attribution of malicious segments.

## Highlights & Insights
- **Re-framing "Defense" as a "Contrastive Signal"**: While previous works only asked if a watermark is detectable, this work asks "Are two watermarks consistent?", converting adversarial robustness into a computable statistic.
- **Symmetric Window Switching**: Decoder head selection follows the same rules as injection, removing the need for extra keys while supporting streaming generation.
- **First Distinction between "Self-generated vs. Externally Injected" Malice**: By using head hit rate asymmetry, it provides attribution evidence, allowing providers to prove "I did not generate this malicious content."

## Limitations & Future Work
- **Reliance on Benign/Malicious Partition**: Contrastive loss requires pre-labeled $\mathcal{D}_s, \mathcal{D}_a$. If malicious distributions shift (e.g., new hate speech, cross-lingual attacks), divergence may weaken.
- **Window-switching Vulnerability**: An adversary knowing $k$ and $\alpha$ could theoretically construct inputs where both heads are near the boundary, weakening the spoof signal.
- **Narrow Quality Assessment**: Only perplexity and task metrics were used; human evaluations are missing, and multiplicative injection impacts on low-probability tokens could accumulate.

## Related Work & Insights
- **vs KGW/SWEET/EWD**: These rely on "greenlist + entropy control." They show Para AUC near 1.0 but Spoof AUC near 0.5. DualGuard raises Spoof AUC to 0.93+, proving the robustness-detectability trade-off is a design choice, not a physical limit.
- **vs SIR/XSIR**: These are also semantic-invariant but use a single head. DualGuard extends them to a "Standard + Adversarial" dual-head setup with contrastive loss.
- **vs An et al. (2025)**: They perform post-hoc watermark removal for spoofed text, which cannot locate or attribute segments. DualGuard enables "identification + attribution."

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First to integrate spoofing defense into watermarking and provide attribution capabilities.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers 2 backbones × 4 datasets × 9 baselines; includes attribution analysis but lacks adaptive adversary experiments.
- **Writing Quality**: ⭐⭐⭐⭐ Clear motivation and logic; some details in Appendix slightly hinder flow.
- **Value**: ⭐⭐⭐⭐⭐ Directly addresses the "false attribution" risk in commercial LLM deployment; high industrial feasibility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Multi-Faceted Attack: Exposing Cross-Model Vulnerabilities in Defense-Equipped Vision-Language Models](../../AAAI2026/llm_safety/multi-faceted_attack_exposing_cross-model_vulnerabilities_in_defense-equipped_vi.md)
- [\[ACL 2026\] Detecting RAG Extraction Attack via Dual-Path Runtime Integrity Game](detecting_rag_extraction_attack_via_dual-path_runtime_integrity_game.md)
- [\[ICLR 2026\] Stop Tracking Me! Proactive Defense Against Attribute Inference Attack in LLMs](../../ICLR2026/llm_safety/stop_tracking_me_proactive_defense_against_attribute_inference_attack_in_llms.md)
- [\[ACL 2026\] SharedRequest: Privacy-Preserving Model-Agnostic Inference for Large Language Models](sharedrequest_privacy-preserving_model-agnostic_inference_for_large_language_mod.md)
- [\[CVPR 2026\] Multi-Paradigm Collaborative Adversarial Attack Against Multi-Modal Large Language Models](../../CVPR2026/llm_safety/multi-paradigm_collaborative_adversarial_attack_against_multi-modal_large_langua.md)

</div>

<!-- RELATED:END -->
