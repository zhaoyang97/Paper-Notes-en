---
title: >-
  [Paper Note] DMAP: A Distribution Map for Text
description: >-
  [ICLR 2026][AIGC Detection][distribution map for text] This paper proposes DMAP (Distribution Map), a mathematical framework that maps text to i.i.d. samples on $[0…
tags:
  - "ICLR 2026"
  - "AIGC Detection"
  - "distribution map for text"
  - "machine-generated text detection"
  - "statistical testing"
  - "token probability"
  - "language model analysis"
date: 2026-05-08
content_hash: b45480733aa2aed6
---

# DMAP: A Distribution Map for Text

**Conference**: ICLR 2026
**arXiv**: [2602.11871](https://arxiv.org/abs/2602.11871)
**Code**: [https://github.com/Featurespace/dmap](https://github.com/Featurespace/dmap)
**Area**: AIGC Detection
**Keywords**: distribution map for text, machine-generated text detection, statistical testing, token probability, language model analysis

## TL;DR

This paper proposes DMAP (Distribution Map), a mathematical framework that maps text to i.i.d. samples on $[0,1]$ via next-token probability rankings from a language model. A formal theorem proves that purely sampled text yields a uniform distribution, enabling $\chi^2$-based verification of generation parameters, exposing the root cause of the complete failure of probability-curvature detectors under pure sampling, and visualizing statistical fingerprints left by post-training (SFT/RLHF) in downstream models.

## Background & Motivation

**Background**: The next-token probability distributions of language models encode rich statistical information about text. Existing methods primarily analyze text or detect machine-generated text via scalar metrics such as perplexity, log-likelihood, and log-rank. DetectGPT pioneered the "probability curvature" paradigm—perturbing text and comparing likelihood shifts to determine whether it is machine-generated; FastDetectGPT improved efficiency via conditional probability normalization; Binoculars performs zero-shot detection using the probability ratio of two models.

**Limitations of Prior Work**: All probability-curvature methods implicitly assume that machine-generated text systematically favors the head of the probability distribution (i.e., selecting high-probability tokens), so that its "probability curvature" is opposite to that of human text. This assumption holds only when truncated sampling strategies such as top-k, top-p, or low temperature are used. Under pure sampling (temperature = 1.0, no truncation), the assumption fails entirely: FastDetectGPT's AUROC collapses from 0.702 to 0.200, and Binoculars' from 0.825 to 0.325—both worse than random guessing. More critically, the authors discover a systematic data error in the existing detection literature: HuggingFace previously enabled top-k=50 by default, causing several top-venue papers (DetectGPT, FastDetectGPT, Binoculars) to unknowingly use top-k=50 in experiments claimed to use pure sampling.

**Key Challenge**: Existing metrics such as perplexity and log-rank suffer from a "contextualization" problem—whether a token's log-likelihood is anomalously high depends on the shape of the conditional distribution at that position (i.e., how many plausible candidates exist), yet these metrics completely ignore such contextual information. Different text genres (poetry vs. news vs. technical writing) systematically affect the shape of conditional distributions, so the same probability value carries entirely different meaning across contexts.

**Goal**: (1) Establish a text statistical representation framework that simultaneously encodes rank and probability information with rigorous mathematical guarantees; (2) use this framework to reveal the fundamental causes of failure in existing detection methods; (3) provide efficient tools for data integrity verification and post-training analysis.

**Key Insight**: Map each token to a subinterval of $[0,1]$ according to its rank in the conditional probability distribution—high-probability tokens correspond to the left side (near 0) and low-probability tokens to the right (near 1), with interval length equal to the token's conditional probability. This mapping is essentially a dynamically ranked extension of the Probability Integral Transform (PIT) to discrete distributions.

**Core Idea**: DMAP maps text to a distribution over $[0,1]$; pure sampling corresponds to an exactly uniform distribution, and any deviation from uniformity is a quantifiable signal of the generation strategy or text properties.

## Method

### Overall Architecture

Given text $w_1 \cdots w_T$ and a scoring language model $p$, DMAP performs the following at each position $i$: (1) rank all tokens in the vocabulary in descending order by $p(\cdot|w_1 \cdots w_{i-1})$; (2) construct the interval $I_i = [a_i, b_i] \subset [0,1]$ corresponding to token $w_i$; (3) draw a DMAP sample $x_i \sim U(I_i)$. The resulting sequence $x_1 \cdots x_T$ is binned into $k=40$ equal-width bins to produce a histogram—the text's "distributional fingerprint." The framework supports three applications: generation parameter verification ($\chi^2$ testing), detection method design analysis, and post-training statistical fingerprint visualization.

### Key Designs

1. **DMAP Mapping and Uniformity Theorem**:

    - **Function**: Maps each token to a point on $[0,1]$, simultaneously encoding its probability magnitude and rank.
    - **Mechanism**: For position $i$, define $V_i^+ = \{v \in V : p(v|w_1 \cdots w_{i-1}) > p(w_i|w_1 \cdots w_{i-1})\}$ as the set of tokens more probable than $w_i$, and $a_i = \sum_{v \in V_i^+} p(v|w_1 \cdots w_{i-1})$ as their cumulative probability, with $b_i = a_i + p(w_i|w_1 \cdots w_{i-1})$. The interval $I_i = [a_i, b_i]$ encodes rank information in its left endpoint and probability magnitude in its length; then $x_i \sim U(a_i, b_i)$. The core theorem (Proposition 3.1) proves that when text is purely sampled from model $p$, $x_1 \cdots x_T$ are i.i.d. uniform on $[0,1]$. The proof is concise: for any subinterval $(c,d) \subset [a,b)$ where $[a,b)$ is the interval of some token $v$, $\mathbb{P}(x_i \in (c,d)) = p(v|\text{context}) \cdot \frac{d-c}{b-a} = (b-a) \cdot \frac{d-c}{b-a} = d-c$. No assumptions about the language model are required, so the theorem also applies to distributions modified by decoding strategies (as long as generation and evaluation use the same strategy).
    - **Design Motivation**: The uniformity theorem provides a precise null hypothesis for all subsequent analyses—any deviation from uniformity encodes meaningful signal (generation strategy, model discrepancy, human text characteristics, etc.).

2. **Entropy-Weighted DMAP ($\hat{D}$)**:

    - **Function**: Eliminates randomness and assigns higher weight to informative positions, improving sensitivity.
    - **Mechanism**: For each position $i$, compute the entropy $h_i$ of the next-token distribution, and let $h_i' = \min(h_i, \lambda)$ with truncation threshold $\lambda=2$. Define the deterministic weighted density $\hat{D}(\underline{w}) = \frac{\sum_i h_i' \cdot \chi_{I_i}/|I_i|}{\sum_i h_i'}$, where $\chi_{I_i}/|I_i|$ is the normalized indicator function on interval $I_i$. Compared with the stochastic sampling variant, this formulation eliminates random noise and focuses analysis on positions where the model is "uncertain," via entropy weighting.
    - **Design Motivation**: At low-entropy positions (e.g., high-probability function words), token selection is nearly identical for both humans and machines and contributes little discriminative signal. Experiments (Appendix F) show that DMAP computed solely from low-entropy positions is nearly perfectly uniform, containing minimal information. Entropy weighting effectively amplifies signals from high-entropy positions.

3. **$\chi^2$ Quantitative Verification Framework**:

    - **Function**: Provides a rigorous statistical hypothesis test to verify the generation parameters of a given text.
    - **Mechanism**: Partition $[0,1]$ into $k$ equal-width bins (with $k = (2T)^{1/3}$ per the Terrell–Scott rule), compute the observed frequency $f_i$ for each bin, and form $\chi^2 = Tk \sum_{i=1}^{k}(f_i - 1/k)^2$. By the i.i.d. uniformity of Proposition 3.1, this statistic is asymptotically $\chi^2_{k-1}$-distributed, enabling direct p-value computation to test whether "the text was generated by the specified strategy." The empirical guideline is that p-values are reliable when $T \geq 10k$.
    - **Design Motivation**: Provides a quantitative tool beyond visual inspection, capable of detecting generation parameter errors in data with extremely high confidence (e.g., the authors used this approach to uncover the top-k=50 data error in several top-venue papers).

### Theoretical DMAP Shapes for Different Sampling Strategies

Different decoding strategies produce highly characteristic DMAP shapes, enabling inference of generation parameters: pure sampling yields a uniform distribution; top-p=$\pi$ sampling is nearly flat on $[0, \pi]$ followed by a sharp drop (since the top-p set has total probability mass slightly exceeding $\pi$); top-k sampling is approximately flat near $[0, 0.5]$ then smoothly declines; temperature sampling with $\tau < 1$ produces a smooth left-skewed deformation. These shapes are determined by the statistical regularities of top-k/top-p sets within the conditional probability distribution space.

## Key Experimental Results

### Main Results: Probability-Curvature Detectors Completely Fail Under Pure Sampling

| Method | Generator | XSum (k=50) | XSum (Pure) | SQuAD (k=50) | SQuAD (Pure) | Writing (k=50) | Writing (Pure) |
|--------|-----------|-------------|-------------|--------------|--------------|----------------|----------------|
| FastDetectGPT | Llama-3.1-8B | 0.702 | **0.200** | 0.739 | **0.208** | 0.915 | **0.289** |
| FastDetectGPT | Mistral-7B | 0.770 | 0.276 | 0.819 | 0.299 | 0.906 | 0.339 |
| FastDetectGPT | Qwen3-8B | 0.765 | 0.289 | 0.612 | 0.320 | 0.923 | 0.377 |
| DetectGPT | Llama-3.1-8B | 0.606 | 0.408 | 0.527 | 0.299 | 0.723 | 0.422 |
| DetectGPT | Mistral-7B | 0.679 | 0.486 | 0.586 | 0.365 | 0.688 | 0.457 |
| Binoculars | Llama-3.1-8B | 0.825 | **0.325** | 0.849 | **0.365** | 0.942 | **0.410** |
| Binoculars | Mistral-7B | 0.823 | 0.350 | 0.851 | 0.416 | 0.931 | 0.404 |
| Binoculars | Qwen3-8B | 0.857 | 0.416 | 0.752 | 0.467 | 0.949 | 0.492 |

### Post-Training Fingerprint Analysis (Pythia 1B + Various SFT Data)

| SFT Data | DMAP Distribution Characteristic | Interpretation |
|----------|----------------------------------|----------------|
| No fine-tuning (Pythia base) | Pronounced right skew (tail-biased) | Large discrepancy between base model conditional distributions and the small scoring model |
| OASST2 human data | Slight right skew + significant tail-collapse | Human-authored instruction data exhibits a distinctive sharp tail decay in DMAP |
| OASST2 + Llama T=1.0 pure sampling | Close to base model, slight right skew | Statistical characteristics of pure-sampling data transfer to the downstream model |
| OASST2 + Llama T=0.7 temperature sampling | **Left skew (head-biased)** | The only model exhibiting left skew; head preference from temperature sampling transfers directly |

### Key Findings

- **The probability curvature assumption fully inverts under pure sampling**: All three detectors achieve AUROC < 0.5 under pure sampling, meaning their discriminative direction is reversed. This is not merely that detection becomes harder—the probability curvature assumption is fundamentally invalid in this setting: base model pure-sampling text evaluated cross-model is tail-biased, in the same or even more extreme direction as human text.
- **The HuggingFace default top-k=50 data error has widespread impact**: DMAP's $\chi^2$ test can detect this error with confidence $p < 10^{-10}$ using only 10,000 tokens, yet multiple top-venue papers' experimental conclusions rest on this erroneous data.
- **DMAP is robust to paraphrase attacks**: Machine-generated text paraphrased with DIPPER remains clearly distinguishable from human text in DMAP; paraphrasing only slightly flattens the distribution while the characteristic shape is preserved.
- **The statistical fingerprint of SFT data transfers faithfully to downstream models**: Fine-tuning on synthetic data sampled at temperature 0.7 produces a head-biased model, whereas fine-tuning on human data or pure-sampling data preserves tail-biased behavior, demonstrating that the DMAP fingerprint of training data is faithfully propagated into the generation distribution.
- **An anomalous density spike in the last bin is observed in instruction-tuned models**: This may reflect mild overfitting; DMAP could be used to guide early stopping in SFT.
- **Fast convergence**: Clear characteristic shapes emerge with as few as 2,000 tokens, and noise is largely eliminated beyond 20,000 tokens; for very short texts, reducing the number of bins (e.g., to 5) can partially mitigate the limitation.

## Highlights & Insights

- **A perfect synthesis of mathematical elegance and practical utility**: The proof of Proposition 3.1 requires only a few lines yet provides a precise null hypothesis (uniform distribution), giving all subsequent analyses a rigorous statistical foundation. This paradigm of "simple theorem + rich applications" is rare in machine learning papers.
- **Joint encoding of rank and probability is the key extension of DMAP over PIT**: Classical PIT requires a natural ordering over categorical variables; DMAP removes this constraint by dynamically re-ranking tokens according to model probability. The authors compare against randomly ordered PIT in the appendix, confirming that no useful information can be extracted without dynamic ranking, thereby validating its necessity.
- **Meta-research value of data error discovery**: DMAP functions not only as an analytical tool but also as a "data auditor"—uncovering systematic data errors in multiple top-venue papers caused by HuggingFace default settings. This highlights the need for more rigorous data integrity verification practices in LLM experimentation.
- **Effective operation with OPT-125m**: DMAP requires only a single forward pass; combined with small models such as OPT-125m, the analysis can be completed in minutes on consumer hardware, greatly lowering the barrier to adoption.

## Limitations & Future Work

- **Positioned as an analysis tool rather than a detector**: DMAP does not directly output a binary human/machine classification; in detection scenarios, an independent decision layer must be built on top of DMAP, but the paper provides no concrete proposal or AUROC results for this direction.
- **Scoring model assumption**: DMAP requires a specified scoring language model; cross-model evaluation naturally produces tail-biased distributions between base models, which may obscure the signal under analysis. The authors recommend using DMAP to calibrate the direction of bias before designing detectors in such settings.
- **Short-text limitation**: The $\chi^2$ test requires $T \geq 10k$ (at least 400 tokens for 40 bins), yielding insufficient statistical power for short texts such as individual tweets or brief comments. Reducing the number of bins alleviates this but also increases information loss.
- **Entropy truncation threshold $\lambda$**: The paper fixes $\lambda=2$ without providing ablation studies or an adaptive selection strategy. Entropy distributions vary substantially across domains (code vs. literary writing), and a fixed threshold may be suboptimal.
- **Modern self-supervised detection methods not explored**: The comparisons are limited to the DetectGPT family and Binoculars; no comprehensive evaluation against watermarking-based methods, trained detectors (e.g., RoBERTa-based), or more recent approaches (e.g., the multi-observer MOSAIC framework) is provided.

## Related Work & Insights

- **vs. DetectGPT/FastDetectGPT**: Built on the probability curvature assumption, DMAP precisely characterizes the condition under which these methods fail—under pure sampling, cross-model evaluation is tail-biased, opposite to the expected curvature direction (AUROC < 0.5). DMAP's principle of "first use visualization to determine head/tail bias direction, then design the detector" is more principled than assuming a fixed curvature direction.
- **vs. Binoculars**: Uses a dual-model probability ratio for normalization, but its theoretical justification is unclear (the original authors state "the theoretical justification for their normalization scheme remains unclear"), whereas DMAP's uniformity theorem provides a transparent theoretical guarantee.
- **vs. GLTR**: Both perform token-level visualization, but GLTR only applies discrete coloring based on rank (top-10/100/1000), making it a coarse discrete approximation of DMAP. DMAP retains full probability and rank information through a continuous mapping.
- **Intersection with model calibration literature**: The DMAP perspective complements research on "overconfidence in instruction-tuned models"—Luo et al. 2025 and Shen et al. 2024 study overconfidence from a calibration standpoint, while DMAP provides visualization and quantification tools from the perspective of the generation distribution; the two lines of work are mutually complementary.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The "pure sampling = uniform distribution" theorem in Proposition 3.1 is concise and powerful, offering an entirely new perspective on text analysis.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Three application scenarios are thoroughly demonstrated (parameter verification, detection method analysis, SFT fingerprinting), though quantitative comparison as a standalone detector is relatively weak.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Mathematical derivations are concise and rigorous, intuitions are clearly explained, and the appendices are exceptionally detailed (covering prompt sensitivity, convergence analysis, and adversarial robustness).
- **Value**: ⭐⭐⭐⭐⭐ Uncovers systematic data errors in multiple top-venue papers, and provides rigorous theoretical tools and new design principles for text analysis and detection method development.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Is Your Paper Being Reviewed by an LLM? Benchmarking AI Text Detection in Peer Review](is_your_paper_being_reviewed_by_an_llm_benchmarking_ai_text_detection_in_peer_re.md)
- [\[ACL 2026\] Reasoning-Based Refinement of Unsupervised Text Clusters with LLMs](../../ACL2026/aigc_detection/reasoning-based_refinement_of_unsupervised_text_clusters_with_llms.md)
- [\[AAAI 2026\] Optimized Algorithms for Text Clustering with LLM-Generated Constraints](../../AAAI2026/aigc_detection/optimized_algorithms_for_text_clustering_with_llm-generated_constraints.md)
- [\[ACL 2026\] When Personalization Tricks Detectors: The Feature-Inversion Trap in Machine-Generated Text Detection](../../ACL2026/aigc_detection/when_personalization_tricks_detectors_the_feature-inversion_trap_in_machine-gene.md)
- [\[ACL 2026\] Frankentext: Stitching Random Text Fragments into Long-Form Narratives](../../ACL2026/aigc_detection/frankentext_stitching_random_text_fragments_into_long-form_narratives.md)

</div>

<!-- RELATED:END -->
