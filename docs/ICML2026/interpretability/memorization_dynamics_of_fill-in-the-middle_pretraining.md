---
title: >-
  [Paper Note] Memorization Dynamics of Fill-in-the-Middle Pretraining
description: >-
  [ICML 2026][Interpretability][Fill-in-the-Middle] The authors train a pair of Llama 3.2 3B models (one standard LTR, one FIM) using identical architectures, data…
tags:
  - "ICML 2026"
  - "Interpretability"
  - "Fill-in-the-Middle"
  - "verbatim memorization"
  - "pretraining objectives"
  - "prefix-suffix probe"
  - "Llama 3.2"
date: 2026-05-08
content_hash: bd165b8fe37f0a21
---

# Memorization Dynamics of Fill-in-the-Middle Pretraining

**Conference**: ICML 2026  
**arXiv**: [2605.22981](https://arxiv.org/abs/2605.22981)  
**Code**: https://github.com/tobiasvonarx/memorization-study-fim  
**Area**: Interpretability / LLM Pretraining / Memorization & Privacy  
**Keywords**: Fill-in-the-Middle, verbatim memorization, pretraining objectives, prefix-suffix probe, Llama 3.2

## TL;DR
The authors train a pair of Llama 3.2 3B models (one standard LTR, one FIM) using identical architectures, data, and compute to systematically compare verbatim memorization behavior on repeated Gutenberg corpora. They find that FIM distributes probability mass over more "partial reconstructions" (stronger short-span/overlapping recall that grows approximately linearly with repetition), while LTR excels at high-confidence continuations of long spans; FIM memorization remains heavily dependent on prefixes, with suffixes serving only as auxiliary signals.

## Background & Motivation

**Background**: The fact that large models replicate training data verbatim is a known issue. From early canary exposure scores to subsequent real extraction attacks by Carlini et al., the community has quantified "memorization" across various dimensions: exact extraction, probabilistic extraction, book-level extraction, and membership inference. Fill-in-the-Middle (FIM), a pretraining objective that moves the target span behind a prefix-suffix separated by sentinels, has been widely adopted by models like DeepSeek-v3, InCoder, StarCoder, and Code Llama to equip causal LMs with middle-filling capabilities.

**Limitations of Prior Work**: Previous research on FIM almost exclusively focused on "infilling utility"—whether it can complete code or sentences—but lacks a systematic quantification of how FIM alters memorization behavior. Intuitively, FIM sees bidirectional context and "should" memorize more easily; however, the same text appears with different prefix/middle/suffix splits during FIM training, which might weaken the memorization of a single long continuation. No controlled experiments have addressed how these two forces compete.

**Key Challenge**: Evaluating "memorization" depends on many confounding factors—model scale, tokenization, prompt position, prior predictability, and near-duplicates all affect extraction rates. To decouple differences caused by the FIM objective itself from differences in model quality or data distribution, rigorous paired training is required.

**Goal**: The study is divided into three specific sub-questions: (i) What is the shape of FIM's impact on verbatim extraction across different target span lengths, extraction thresholds, and repetition counts? (ii) Under native FIM prompts, how much do the prefix context, suffix context, and sentinel tokens each contribute to memorization? (iii) Are observed differences caused by the geometry of extraction (probe format) or differences in model capability?

**Key Insight**: A pair of Llama 3.2 3B models with identical architectures, data sources, and token counts (one LTR, one FIM) is trained on the same FineWeb + Gutenberg dataset. Gutenberg is divided into 12 bins with 1–128 repetitions. A FineWeb-only model is first used to filter out "pre-memorized" windows, making the number of repetitions the sole variable.

**Core Idea**: The study quantitatively decouples the differences in memorization mechanisms between FIM and LTR through controlled paired training, dual extraction metrics (exact $p_z$ extraction + ROUGE-L overlap), and native FIM prefix/suffix distractor probes.

## Method

### Overall Architecture
This research is essentially a comparative experimental design rather than a "model method." The pipeline consists of three steps: (A) Constructing comparable training pairs—splitting homologous data into LTR and FIM corpora, with the Gutenberg portion binned by repetition; (B) Measuring verbatim extraction under prefix-only probes to obtain extraction curves for FIM/LTR across different repetitions, span lengths, and thresholds; (C) Switching to native FIM prompts (prefix + sentinel + suffix) to quantitatively isolate the memorization contributions of the prefix and suffix by allocating a 100-token budget and using distractor texts. Both models are evaluated on 8 downstream tasks via the LM Evaluation Harness to ensure nearly identical performance, thereby ruling out the "capability difference" hypothesis.

### Key Designs

1.  **Paired Training + Repetition-Binned Corpus**:
    - **Function**: Provides a controlled environment where all variables except the "pretraining objective" are locked for the FIM/LTR comparison.
    - **Mechanism**: FineWeb 100B is used for the bulk corpus, and Project Gutenberg for controlled memorization. A Llama 3.2 model trained only on FineWeb scores 4096-token windows in Gutenberg to filter out outliers, duplicates, and pre-memorized windows. Remaining excerpts are balanced by prior perplexity and distributed into 12 bins with exposure counts $\{1, 2, \dots, 128\}$ (2810 excerpts per bin). The LTR corpus maintains autoregressive order, while the FIM corpus rewrites samples into sentinel-separated prefix–suffix–middle segments (50% FIM for FineWeb, 100% FIM for Gutenberg). Critically, multiple repetitions of the same excerpt in FIM use different random split points, making repetition a "document-level exposure" rather than "fixed middle span exposure."
    - **Design Motivation**: To isolate extraction differences from model capability, tokenization, or prior memorability, while retaining the realistic usage of FIM (random splitting) to ensure conclusions generalize to actual pretraining.

2.  **Dual Extraction Metrics (exact $p_z$ + ROUGE-L)**:
    - **Function**: Avoids missing important differences by relying on a single span length or probe format.
    - **Mechanism**: With a fixed 100-token prefix and target span $M=32$, the first metric is the exact extraction probability $p_z = \prod_{i=1}^{M} q_i$, where $q_i$ is the probability of the $i$-th target token after top-$k=40$ renormalization ($T=1$). A threshold of $p_z \geq 0.1\%$ counts as extractable. The second metric is the ROUGE-L score between a 32-token autoregressive generation and the ground truth, with $\geq 0.5$ indicating high-overlap recall. Scans of thresholds and lengths $M \in \{20, 30, 40, 50\}$ are performed at repetition=128. FIM accumulates more mass in the medium $p_z$ range and performs better on ROUGE-L and top-$k$ support, whereas LTR has a heavier right tail, extracting more windows at the strict $0.1\%$ threshold and for long spans.
    - **Design Motivation**: A single threshold would be dominated by LTR's heavy tail, and a single span length would miss FIM's "partial reconstructions." Together, they reveal the different "shapes" of memorization—LTR as "high peaks" and FIM as "small hills."

3.  **Native FIM Probe + Prefix/Suffix Distractor Ablation**:
    - **Function**: Isolates the contributions of prefix context, suffix context, and sentinel tokens under the bidirectional prompt format where FIM is actually used.
    - **Mechanism**: With a 100-token context budget, the prefix/suffix split ratio is scanned around the target. As context shifts from pure suffix to pure prefix, perplexity drops monotonically from 60.23 to 27.93, and top-$k$ support rises from 77.60% to 85.52%. In the distractor experiment, the prefix, suffix, or both are replaced with irrelevant Gutenberg text of the same length. Replacing the prefix nearly eliminates memorization, while replacing the suffix causes only a minor drop. Attention analysis (Table 1) shows that FIM allocates higher attention to the prefix (0.646 vs. 0.604 for LTR) and looks less at already generated target tokens.
    - **Design Motivation**: Suffixes and prefixes of equal length show asymmetric roles. Despite FIM appearing "bidirectional," the autoregressive nature of causal LMs keeps the prefix as the anchor of memorization. Distrator experiments ensure that support gains are not artifacts of prompt length or sentinel structure.

### Loss & Training
Both models are Llama 3.2 3B implemented via Megatron-LM: 28 layers, hidden size 3072, 24 attention heads, 8 KV heads, FFN 8192, vocab 128256, RoPE base 500000, bfloat16, no dropout. They use packed THD sequences of length 16384, micro-batch 1, global batch 2048, running on 64 GH200 GPUs at 33.5M tokens per step. LTR runs for 3057 steps (~102.58B tokens), and FIM runs for 3064 steps (~102.81B tokens), ensuring strict alignment.

## Key Experimental Results

### Main Results

| Metric | Repetition Range | FIM | LTR | Meaning |
| :--- | :--- | :--- | :--- | :--- |
| Exact extraction count ($p_z \geq 0.1\%$, $M=32$) | 1–128 Bins Total | 2,230 | 3,279 | LTR extracts more under strict thresholds |
| Average ROUGE-L | Same | 0.198 | 0.190 | FIM has slightly higher overlap recall |
| Average top-$k$ support ($k=40$) | Same | 87.09% | 86.18% | FIM distributes more mass into top-$k$ |
| Prefix-only probe extraction | Repetition=128 | Higher than LTR | Lower than FIM | FIM overtakes at high repetition |
| Long span ($M=50$) extraction | High Repetition | Lower than LTR | Higher than LTR | LTR's heavy tail excels for long spans |

### Ablation Study

| Configuration | top-$k$ Support | Description |
| :--- | :--- | :--- |
| Pure prefix 100 tokens (Native FIM) | 85.52% | Upper bound anchor |
| Pure suffix 100 tokens | 77.60% | Significant drop, target perplexity 60.23 |
| True prefix + True suffix (Full prompt) | Highest overall | Upper reference for distractor experiments |
| True prefix + Distractor suffix | Slight decrease | Suffix interference has limited impact |
| Distractor prefix + True suffix | Substantial drop | Prefix is the anchor of memorization |
| Distractors on both sides | Collapse | Rules out "prompt length/sentinel structure" hypothesis |
| FIM Attention (prefix-only probe) | prefix 0.646 / prev-target 0.354 | FIM relies more on the prefix |
| LTR Attention (prefix-only probe) | prefix 0.604 / prev-target 0.396 | LTR looks more at generated tokens |

### Key Findings
- FIM and LTR differ not in "which is more likely to memorize," but in the "shape of memorization": LTR clusters probability mass into a few high peaks (friendly to heavy tails and long spans), while FIM spreads it across many small hills (friendly to partial reconstruction and nearly linear growth with repetition).
- For $M=32$ at a $0.1\%$ threshold, FIM extraction overtakes LTR at high repetition levels; however, as the target span length increases, FIM requires more repetitions to overtake LTR—long spans are the domain of LTR’s heavy tail.
- Native FIM prompts appear bidirectional but are actually heavily prefix-anchored: replacing the prefix nearly erases memorization, while replacing the suffix is only a minor penalty. This suggests FIM training does not truly enable "bidirectional memorization."
- FIM and LTR performance on 8 LM Evaluation Harness tasks is nearly identical, ruling out the hypothesis that capability differences drive extraction differences.

## Highlights & Insights
- Systematic scanning of "dual metrics + multiple span lengths + multiple repetition bins" yields the "distribution shape" of memorization for FIM/LTR rather than just a single score—a 2D perspective unavailable from point numbers.
- The use of prefix/suffix distractor experiments to prove "prefix reliance" is more convincing than simply reporting attention distributions.
- Controlled repetition binning combined with filtering "pre-memorized" windows via a FineWeb-only model provides a clean engineering paradigm for memorization audits that can be adopted by future work.
- The difference in attention allocation (FIM attending more to the prefix) explains why it does not stack probability mass for long continuations like LTR—a mechanistic discovery.

## Limitations & Future Work
- The model scale is capped at 3B (with 1B ablations), which may not generalize to frontier scales where the relationship between capacity and memorization might shift the relative positions of FIM/LTR.
- The repetition range (1–128) covers practical scenarios but does not allow for trend extrapolation at the limit.
- A conceptual limitation is "attribution": because FIM uses random splitting, the probe span does not necessarily correspond to a specific middle span seen during training.
- Additional Observation: Probes are based on Gutenberg (linguistic text) and do not cover code, which is the primary industrial application of FIM. Memorization dynamics on code might diverge from these conclusions.
- Future work could include scanning for prompt position sensitivity (referencing positional fragility in Xu et al. 2026) and exact mapping from span to training instances to verify if these patterns hold for longer extraction windows.

## Related Work & Insights
- **vs. Carlini et al. 2023 (Quantifying Memorization)**: That study established scaling laws for LTR where memorization grows logarithmically with capacity/repetition; this paper extends the comparison to FIM, finding that FIM's curve is nearly linear rather than logarithmically saturated.
- **vs. Cooper et al. 2026 (Book-level Extraction)**: This paper adopts the $p_z$ extraction metric but uses $M=32$ to compare ROUGE-L and exact extraction on the same windows, shifting the analysis from "how many books are extracted" to "shape differences between FIM/LTR."
- **vs. Huang et al. 2024 (Demystifying Verbatim Memorization)**: Corroborates that significant repetition is needed for memorization and generalizes this from LTR to FIM, finding that FIM starts slower but grows more steadily.
- **vs. Bavarian et al. 2022 (Original FIM Paper)**: The original work proved FIM does not significantly harm downstream capabilities; this paper shifts the perspective to "security/privacy audits," revealing that FIM changes the geometry rather than the total amount of memorization.

## Rating
- Novelty: ⭐⭐⭐⭐ First systematic quantification of memorization differences between FIM and LTR; clean design with paired training and dual metrics.
- Experimental Thoroughness: ⭐⭐⭐⭐ Paired 3B training + repetition bins + dual probes + distractor ablations + 1B/downstream task baseline; pushed to the limit of feasible academic experimentation.
- Writing Quality: ⭐⭐⭐⭐ Clear structure; the progression from "prefix-only → native FIM → distractor" is natural, and methodological lessons are stated plainly.
- Value: ⭐⭐⭐⭐ Direct guidance for pretraining objective selection and memorization audits; the finding that FIM remains prefix-anchored affects privacy assessments for infilling models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Grokking in LLM Pretraining? Monitor Memorization-to-Generalization without Test](../../ICLR2026/interpretability/grokking_in_llm_pretraining_monitor_memorization-to-generalization_without_test.md)
- [\[ICML 2026\] Is One Layer Enough? Understanding Inference Dynamics in Tabular Foundation Models](is_one_layer_enough_understanding_inference_dynamics_in_tabular_foundation_model.md)
- [\[ICML 2026\] Dissecting Multimodal In-Context Learning: Modality Asymmetries and Circuit Dynamics in modern Transformers](dissecting_multimodal_in-context_learning_modality_asymmetries_and_circuit_dynam.md)
- [\[ICML 2026\] Tracing the Dynamics of Refusal: Exploiting Latent Refusal Trajectories for Robust Jailbreak Detection](tracing_the_dynamics_of_refusal_exploiting_latent_refusal_trajectories_for_robus.md)
- [\[CVPR 2026\] Pixel2Phys: Distilling Governing Laws from Visual Dynamics](../../CVPR2026/interpretability/pixel2phys_distilling_governing_laws_from_visual_dynamics.md)

</div>

<!-- RELATED:END -->
