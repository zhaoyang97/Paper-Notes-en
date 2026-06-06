---
title: >-
  [Paper Note] Annotations Mitigate Post-Training Mode Collapse
description: >-
  [ICML 2026][LLM Pretraining][Mode Collapse] The authors observe that SFT aligns models to a low-entropy semantic prior, leading to a "the larger the instruction model…
tags:
  - "ICML 2026"
  - "LLM Pretraining"
  - "Mode Collapse"
  - "SFT"
  - "Semantic Entropy"
  - "Annotation Anchoring"
  - "Diversity-Quality Tradeoff"
date: 2026-05-08
content_hash: 57d82d45172e6b05
---

# Annotations Mitigate Post-Training Mode Collapse

**Conference**: ICML 2026  
**arXiv**: [2605.09995](https://arxiv.org/abs/2605.09995)  
**Code**: Not explicitly released  
**Area**: LLM Pretraining / Post-training Alignment / Generation Diversity  
**Keywords**: Mode Collapse, SFT, Semantic Entropy, Annotation Anchoring, Diversity-Quality Tradeoff

## TL;DR
The authors observe that SFT aligns models to a low-entropy semantic prior, leading to a "the larger the instruction model, the more boring" reverse scaling effect. They propose "annotation-anchored training": during pretraining, semantic tags are paired with documents; during SFT, loss on tag tokens is masked. At inference, the model first samples semantics, then generates responses, thereby narrowing the semantic diversity gap by 85% while retaining instruction-following ability.

## Background & Motivation
**Background**: The mainstream alignment pipeline involves unsupervised pretraining on massive web data, followed by SFT (plus RLHF) on a relatively narrow instruction dataset, sculpting the base model into an obedient, well-formatted assistant.

**Limitations of Prior Work**: The base model outputs are uneven in quality but semantically broad; after SFT, responses to the same prompt become highly similar in topic, names, locations, and style—this is "semantic mode collapse." Worse, the authors reproduce and extend NoveltyBench's findings: base models become more diverse as scale increases, but SFT models collapse more as scale increases, and this collapse persists even with diversity prompting strategies like brainstorming or multiple sampling.

**Key Challenge**: SFT optimizes for "matching the model distribution to the post-training data distribution," but the post-training data itself is a low-entropy semantic set (few annotator-preferred modes). Maximum likelihood training indiscriminately injects this low-entropy prior into the model. Thus, SFT changes two things at once—"how to write a response given semantics" (desired) and "which semantics are expressed" (undesired), which are entangled.

**Goal**: Decouple these two distributions so that post-training only updates conditional response behavior $Q^\star(y\mid x,z)$, while anchoring the semantic distribution $R(z\mid x)$ at the high-entropy pretraining state.

**Key Insight**: The authors introduce an explicit semantic variable $z$ (a set of key-value tags, e.g., topic / location / entities / genre), factorizing response generation as $P^\star_R(y\mid x)=\int R(z\mid x)\,Q^\star(y\mid x,z)\,dz$. As long as $R$ retains high entropy during training, diversity is preserved.

**Core Idea**: During pretraining, each chunk is annotated and `<z>x` sequences are interleaved; during SFT, loss on z tokens is masked; at inference, the model first outputs an annotation, then a response. The diversity of annotation sampling naturally inherits from the pretraining distribution.

## Method
The method, annotation-anchored training, consists of four stages: annotation, pretraining, post-training, and inference. The core trick is minimal: the training data format and loss mask are slightly changed, but the rest is standard autoregressive LM—no new modules or losses are introduced.

### Overall Architecture
- **Offline**: Qwen3-30B-A3B-Instruct is used as the annotator to extract semantic tags in `<key>:<value>` format (topic, domain, action, entities, location, etc.) for each chunk of pretraining corpus, and similarly for each target response in SFT data.
- **Pretraining**: Each document is split into chunks $x_1,\dots,x_n$ by double newlines, and interleaved as $\langle z_1\rangle x_1 \langle z_2\rangle x_2 \cdots \langle z_n\rangle x_n$ for next-token prediction, with no tokens masked. This enables the model to learn both "generate annotation from context" and "generate text from annotation."
- **Post-training (SFT)**: Training samples are formatted as `prompt <annotation> response`, with loss only backpropagated on response tokens; annotation tokens are fully masked. Thus, the response distribution is updated by instruction data, but the annotation distribution remains frozen from pretraining.
- **Inference**: Direct sampling—model first outputs an annotation (diversity inherited from pretraining), then generates a response conditioned on it. Only the response part is retained for evaluation.

### Key Designs

1. **Explicit Semantic Variable z as Distribution Factorization Bridge**:

    - **Function**: Decomposes response generation into "decide what semantics to express" and "decide how to express them," allowing independent control of the two distributions.
    - **Mechanism**: The authors write the pretraining distribution as $P(y)=\int R(z)Q(y\mid z)dz$, and the post-training distribution as $P^\star(y\mid x)=\int R^\star(z\mid x)Q^\star(y\mid x,z)dz$. When $R^\star$ has much lower entropy than $R$, collapse occurs. The anchoring target is $P^\star_R(y\mid x)=\int R(z\mid x)Q^\star(y\mid x,z)dz$—retaining the pretraining semantic prior while absorbing post-training conditional response ability. z is expressed as a variable-length set of `<key>:<value>` tags, which are natural language tokens and fully LM-compatible.
    - **Design Motivation**: Traditional entropy regularization or KL constraints only act on the overall response distribution, unable to distinguish "what to change, what to keep." Explicit z directly targets the issue, locking only the "semantic marginal distribution" and allowing "conditional response" to change freely.

2. **Chunk-level Annotation + Interleaved Sequence Pretraining**:

    - **Function**: Enables the model to learn a high-entropy but locally coherent $z\mid x$ conditional distribution, which is the source of annotation diversity at inference.
    - **Mechanism**: Using global annotation for the entire document loses local signals, so the authors split by double newlines, annotate each chunk independently, and interleave as `<z_i>x_i` for standard autoregressive training. The model thus learns the conditional chain "previous context → next segment semantics → next segment text"; this conditional distribution matches the SFT stage's "prompt → annotation → response" prior, enabling seamless transfer.
    - **Design Motivation**: The annotator is noisy and not all tags are accurate; but averaging over chunk-level multi-labels yields a high-entropy annotation distribution, which is what needs to be preserved—making the method robust to individual tag errors.

3. **Annotation-masked SFT + Anchored Inference**:

    - **Function**: During instruction fine-tuning, only updates "how to respond given semantics," without contaminating the "how semantics are sampled" distribution.
    - **Mechanism**: SFT data is formatted as `prompt <annotation> response`, with annotation tokens fully masked in the loss (except for 0.3% cases where only tag values are masked for format stability). Since annotation tokens do not participate in gradient updates, the model's $p(z\mid x)$ prediction retains the high-entropy distribution learned during pretraining; only $p(y\mid x,z)$ is shaped by SFT data. At inference, decoding with temperature 1, the model samples an annotation then generates a response, and the randomness in annotation brings semantic diversity to the output.
    - **Design Motivation**: This is the method's most crucial "one line of code"—it tells the training algorithm "what to learn, what to protect." It is equivalent to imposing an implicit KL constraint on the response distribution, but more precise than explicit KL, as it only constrains $R$ and not $Q$.

### Loss & Training
Pretraining: standard next-token loss, no masking.  
SFT: next-token loss, but annotation token positions are masked.  
Hyperparameters: annotation occupies part of the total token budget ("annotations replace content tokens" to match FLOPs/token); LR tuned via validation perplexity; trained with Chinchilla-optimal token counts (0.6B with 12B tokens, 1B with 20B, 2.5B with 50B).

## Key Experimental Results

### Main Results
Compared with standard SFT on four diversity benchmarks: Stories / NoveltyBench / WildChat / InfinityChat. Stories uses LLM judge to extract 8 attribute tags and compute semantic entropy; dialogues use Qwen3-Embedding-0.6B for pairwise cosine distance.

| Metric | Model Size | Standard SFT | Annotation-Anchored | Notes |
|--------|------------|--------------|---------------------|-------|
| Stories Semantic Entropy | 2.5B | Significant collapse (far below base) | Gap with base reduced to ~15% | 6× less collapse |
| NoveltyBench / WildChat / InfinityChat | 0.6B / 1B / 2.5B | Collapse at all scales | All scales higher than SFT | Consistent across benchmarks |
| Inference Sampling | temp 0.6~1.1 | Raising temperature drops quality, not diversity | Diversity-quality Pareto fully shifted outward | See Fig 6 |

| Task Ability | 0.6B | 1B | 2.5B | Notes |
|--------------|------|----|------|-------|
| ARC/BoolQ/HellaSwag etc. 9-task avg (SFT) | 53.1 | 57.7 | 62.4 | Standard baseline |
| Same (Annotated) | 54.5 | 56.6 | 62.3 | All scale gaps ≤ 1 point |
| GSM8k (SFT / Annotated) | 11.3 / 10.8 | 19.4 / 18.9 | 35.4 / 36.4 | Reasoning ability nearly unaffected |

### Ablation Study

| Configuration | Main Conclusion |
|---------------|----------------|
| Anchored SFT only (no annotated pretraining) | Performance drops significantly, showing high-entropy annotation distribution in pretraining is necessary |
| Increased SFT data volume | Standard SFT diversity drops further; Annotated maintains/improves |
| Higher validation likelihood | Standard SFT: higher likelihood, lower diversity (strong negative correlation); Annotated: positive correlation |
| Different temperature (0.6~1.1) | Annotated always higher diversity than SFT at all temperatures, similar quality |

### Key Findings
- "Tighter fit to post-training data → lower diversity" holds across hyperparameters and data volumes; annotation-anchored reverses this negative correlation, indicating it solves collapse mechanistically, not by trickery.
- Reverse scaling (larger instruction models are narrower) only appears in the SFT pipeline; annotation-anchored fully restores positive scaling (larger = more diverse).
- At 2.5B scale, the semantic diversity gap on Stories closes by 85%, indicating that the "upper bound" of collapse is not inevitable in SFT, but due to suboptimal objective design.

## Highlights & Insights
- **Elegant Distribution Factorization**: The simple $R \cdot Q$ decomposition conceptually separates "semantic prior" and "conditional response," explaining reverse scaling and directly suggesting a remedy—a rare "theoretical clarity + engineering simplicity" combination.
- **Mask loss as anchor**: Not updating annotation tokens is equivalent to imposing an implicit KL constraint on $R$ at zero cost, more precise than explicit KL or entropy regularization.
- **Extensible paradigm**: The authors note that annotation need not be topic/entity; it could be "proof strategy" for math, "algorithm idea" for code—essentially, "any latent variable whose diversity you want to preserve," offering a new template for chain-of-thought and reasoning path diversity.

## Limitations & Future Work
- Annotation requires a relatively strong LLM for extraction (Qwen3-30B-A3B used), which is a nontrivial cost for small teams; however, the annotator need not be accurate, just high-entropy.
- Experiments max out at 2.5B parameters and 50B tokens, still far from the true frontier of 70B+/trillion tokens; scaling up will require retuning SFT data format (annotation token ratio, tag schema).
- Inference incurs extra annotation generation overhead (though short); latency-sensitive deployments need to weigh this.
- Annotation schema is manually designed and coupled to task type; how to automatically discover "which z dimensions should be diverse" remains an open question.

## Related Work & Insights
- **vs NoveltyBench (Zhang et al. 2025b)**: They first reported reverse scaling but only diagnosed the issue; this paper provides a mechanistic explanation and engineering solution, turning "problem" into "method."
- **vs Entropy Regularization / KL-constrained SFT**: Those methods constrain the overall response distribution, unable to distinguish what to preserve; this work uses explicit z to precisely anchor the semantic marginal, offering finer control.
- **vs Latent variable / CVAE dialogue models**: Both introduce latent variables for conditional diversity; the difference is this work aims not for "user-controllable attributes" but for "preserving pretraining semantic priors," with entirely different application scenarios.
- **vs Verbalized Sampling / Diverse Beam Search**: Those are inference-time diversity tricks, orthogonal to this work and can be combined.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Combining "distribution factorization + loss mask" with SFT is a novel and minimalist perspective
- Experimental Thoroughness: ⭐⭐⭐⭐ Solid—across 4 diversity benchmarks, 9 ability benchmarks, multiple temperatures; but scale capped at 2.5B
- Writing Quality: ⭐⭐⭐⭐⭐ Clear concepts, formulas and figures match, factorization argument is elegant
- Value: ⭐⭐⭐⭐⭐ Directly inspires LLM alignment, creative generation, synthetic data diversity; may become a default component of next-gen alignment pipelines

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] FlowMo: Flow to the Mode — Mode-Seeking Diffusion Autoencoders for State-of-the-Art Image Tokenization](../../ICCV2025/llm_pretraining/flow_to_the_mode_mode-seeking_diffusion_autoencoders_for_state-of-the-art_image_.md)
- [\[ICLR 2026\] Explaining Grokking and Information Bottleneck through Neural Collapse Emergence](../../ICLR2026/llm_pretraining/explaining_grokking_and_information_bottleneck_through_neural_collapse_emergence.md)
- [\[CVPR 2026\] Evidential Transformation Network: Turning Pretrained Models into Evidential Models for Post-hoc Uncertainty Estimation](../../CVPR2026/llm_pretraining/evidential_transformation_network_post_hoc_uncertainty_estimation.md)
- [\[ICML 2026\] On Training Large Language Models for Long-Horizon Tasks: An Empirical Study of Horizon Length](on_training_large_language_models_for_long-horizon_tasks_an_empirical_study_of_h.md)
- [\[NeurIPS 2025\] Flatness is Necessary, Neural Collapse is Not: Rethinking Generalization via Grokking](../../NeurIPS2025/llm_pretraining/flatness_is_necessary_neural_collapse_is_not_rethinking_generalization_via_grokk.md)

</div>

<!-- RELATED:END -->
