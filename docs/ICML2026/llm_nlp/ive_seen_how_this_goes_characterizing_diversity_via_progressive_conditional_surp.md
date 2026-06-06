---
title: >-
  [Paper Note] "I've Seen How This Goes": Characterizing LLM and Human Writing Diversity via Progressive Conditional Surprisal
description: >-
  [ICML 2026][LLM/NLP][Diversity Metrics] This paper proposes $D_{Ca_n}=C\cdot a_n$, a diversity metric that requires no embeddings, reference corpora…
tags:
  - "ICML 2026"
  - "LLM/NLP"
  - "Diversity Metrics"
  - "Conditional Surprisal"
  - "In-Context Learning"
  - "Mode Collapse"
  - "RLHF Evaluation"
date: 2026-05-08
content_hash: be05893722b39551
---

# "I've Seen How This Goes": Characterizing LLM and Human Writing Diversity via Progressive Conditional Surprisal

**Conference**: ICML 2026  
**arXiv**: [2606.01811](https://arxiv.org/abs/2606.01811)  
**Code**: https://github.com/AMindToThink/icl-diversity (Available)  
**Area**: LLM Evaluation / Diversity Metrics / Information Theory  
**Keywords**: Diversity Metrics, Conditional Surprisal, In-Context Learning, Mode Collapse, RLHF Evaluation

## TL;DR
This paper proposes $D_{Ca_n}=C\cdot a_n$, a diversity metric that requires no embeddings, reference corpora, or human labels. By using a base model $\theta$ to read all responses in a single forward pass, it measures the "per-byte conditional surprisal of the final response after seeing $n-1$ previous ones" multiplied by the "overall readability of the responses." This metric approaches SentBERT on the McDiv human evaluation benchmark and captures the monotonic decline in diversity across the OLMo-2-7B base→SFT→DPO→RLVR pipeline, accurately detecting mode collapse induced by post-training.

## Background & Motivation

**Background**: Current methods for evaluating generative model diversity primarily follow two paths: surface-level $n$-gram/self-BLEU (Li 2016, Zhu 2018) or embedding-based distances/clustering (Du & Black 2019, SentBERT). The former is limited to literal overlap, while the latter depends on an additional sentence vector model. The McDiv benchmark (Tevet & Berant 2021), which uses human binary classification plus OCA/Spearman ratings, has become the de facto standard for diversity metrics.

**Limitations of Prior Work**: $n$-gram metrics fail to capture "semantic redundancy"—responses with paraphrased content or similar styles may have sparse literal distributions but remain highly redundant. Embedding metrics "outsource" diversity to a sentence vector model, introducing a black-box trained independently; diversity scores thus depend on the preferences of the embedding model, which may fail to recognize unseen patterns (e.g., specific character styles or metaphorical structures). Both paths lack the capability to "recognize arbitrary potential commonalities like a human."

**Key Challenge**: Diversity is essentially "how much new information the next response provides after seeing the previous ones"—an information-theoretic quantity. However, existing work (e.g., MMI decoding, AIM) uses mutual information as a training or decoding objective rather than a *diagnostic tool during evaluation*, as it was generally assumed that base models lack the ability to "learn while reading."

**Goal**: (1) Construct a scalar diversity metric independent of embeddings, reference corpora, or human labels; (2) Achieve correlation with human judgments on the Tevet & Berant benchmark comparable to SentBERT; (3) Detect the "RLHF-induced mode collapse" commonly reported in industry within real post-training pipelines (e.g., OLMo-2-7B base→SFT→DPO→RLVR).

**Key Insight**: The ICL phenomenon (Brown et al. 2020) suggests that base models "learn while reading" when processing concatenated contexts. Therefore, defining a diversity measure based on "how much surprisal remains for the $k$-th response after the base model has read the previous $k-1$ responses" is natural. If a policy $\pi$ is truly diverse, the model $\theta$ cannot learn a pattern via ICL, and conditional surprisal remains high; if $\pi$ cycles through a few modes, $\theta$ identifies them quickly, and conditional surprisal drops toward zero.

**Core Idea**: Use the ICL capability of a base model as a "diversity microscope." Define a progressive conditional surprisal curve $a_k=-\log_2\theta(r_k\mid r_{<k},p)/\|r_k\|$, using the final point $a_n$ to measure the residual per-byte surprisal of the $n$-th response. Multiply this by a coherence weight $C=1/\mathrm{PPL}_\theta(\pi,p)$ to prevent pure noise (which is incompressible) from being misidentified as high diversity.

## Method

### Overall Architecture
The input consists of a prompt $p$, $n$ responses $r_1,\dots,r_n$ sampled from policy $\pi$, and a fixed "scoring base model" $\theta$ (Qwen2.5-3B in experiments). The pipeline operates on two parallel tracks:

- **Conditional Track**: All responses are labeled (e.g., "Response A:", "Response B:") and concatenated into a long context. $\theta$ performs a single forward pass (once per permutation $\sigma$). The total surprisal for each response is extracted and divided by its UTF-8 byte count to obtain a per-byte value. These are averaged by position $k$ to yield $\bar a_k$, where the endpoint $\bar a_n$ represents "residual diversity." To eliminate ordering bias, multiple random permutations (25–50 in experiments) are averaged.
- **Unconditional Track**: Each $r_i$ is scored individually using the prompt as context. The per-byte cross-entropy $h_\theta(r_i\mid p)$ is calculated, and the geometric mean is used to derive coherence $C=1/\mathrm{PPL}_\theta(\pi,p)$.

The final scalar is $D_{Ca_n}=C\times a_n$ (bits/byte), interpreted as "how much surprisal remains per byte after $\theta$ has learned what it can via ICL, weighted by response credibility." The pipeline requires no embeddings, reference corpora, human labels, or auxiliary classifiers—only the per-token probabilities already output by $\theta$.

### Key Designs

1. **Progressive Conditional Surprisal Curve $a_k$ (Per-byte Normalization + Permutation Averaging)**:
    - **Function**: Operationalizes "diversity" as an information-theoretic quantity—the residual per-byte conditional surprisal of the $k$-th response after $k-1$ responses.
    - **Mechanism**: Defined as $a_k=-\log_2\theta(r_k\mid r_{<k},p)/\|r_k\|$, which is equivalent to $a_n=H_\theta(r_n\mid p)-I_\theta(r_n;r_1,\dots,r_{n-1}\mid p)$. High $a_n$ requires responses to be individually surprising yet mutually unpredictable. Normalization uses UTF-8 bytes instead of tokens for cross-tokenizer comparability. Random permutations (25–50 iterations) eliminate positional bias.
    - **Design Motivation**: Existing metrics are either static or require trained classifiers. Using base model ICL as a probe requires zero labeling and can identify high-order commonalities (templates, narrative arcs, styles) invisible to embeddings/$n$-grams. The metric also scales in sensitivity as base models improve.

2. **Coherence Term $C=1/\mathrm{PPL}_\theta(\pi,p)$**:
    - **Function**: Suppresses cases of "high $a_n$ but junk content"—pure noise remains unlearnable by ICL, keeping $a_n$ high.
    - **Mechanism**: The geometric mean of per-byte cross-entropy $C=2^{-\frac{1}{n}\sum_i h_\theta(r_i\mid p)}$ is used. This is the inverse of the geometric mean per-byte perplexity. The geometric form is crucial: a single garbled response with extreme perplexity will pull $C$ toward zero, preventing individual fluent responses from "saving" the score.
    - **Design Motivation**: Forces diversity to be defined within the "readability manifold"—high diversity requires responses that are both mutually unpredictable and individually fluent.

3. **Multiplicative Synthesis $D_{Ca_n}=C\times a_n$ and (responses, prompt, scoring model) Triplet**:
    - **Function**: Combines components into a single comparable scalar and defines diversity relative to the scoring model $\theta$ rather than an intrinsic property of $\pi$.
    - **Mechanism**: The product represents "bits of reasonable surprisal remaining per byte." Five scenarios (pure noise, multi-mode incoherent, multi-mode coherent, one-mode, and mixed) were used to calibrate behavior. Noise is suppressed by $C$ (0.04 on GPT-2), one-mode is suppressed by $a_n$ (0.10), and mixed ranks highest (0.52).
    - **Design Motivation**: (a) Simplifies cross-prompt averaging and paired testing; (b) The triplet definition allows the metric to "sharpen" as models improve, avoiding the "fixed embedding model" ceiling.

### Loss & Training
No training is required—$\theta$ is an off-the-shelf base model (e.g., Qwen2.5-3B, GPT-2, OLMo-2-7B-base). The metric is calculated via a single forward pass. The only "hyperparameters" are the number of permutations (25–50) and responses per prompt ($K=10$). In OLMo experiments, responses were truncated at the UTF-8 byte level to the minimum length within each prompt group to eliminate artifacts of decreasing cross-entropy over long contexts.

## Key Experimental Results

### Main Results
Using Qwen2.5-3B and 50 permutations on McDiv / ConTest benchmarks, evaluated via OCA (Optimal 1D Accuracy) and Spearman $\rho$:

| Dataset / Sub-task | Metric | $D_{Ca_n}$ (Ours) | SentBERT (SOTA) | Distinct-$n$ |
|---|---|---|---|---|
| McDiv prompt_gen (full) | OCA | **0.846** | 0.897 | 0.746 |
| McDiv prompt_gen (full) | $\rho$ | +0.729 | +0.796 | +0.476 |
| McDiv_nuggets prompt_gen | OCA | 0.785 | 0.850 | 0.675 |
| ConTest story_gen | OCA | 0.828 | 0.896 | 0.772 |

OLMo-2-7B Post-training Pipeline (Scoring model $\theta=$ Qwen2.5-3B):

| Stage | AlpacaEval $D_{Ca_n}$ mean | NoveltyBench-curated mean |
|---|---|---|
| Base | 0.481 | 0.481 |
| SFT | 0.329 | 0.369 |
| DPO | 0.286 | 0.312 |
| Instruct (RLVR) | 0.281 | 0.303 |

Pre-registered one-sided paired Wilcoxon tests (Bonferroni $\times 3$) were all significant: on AlpacaEval, Base>SFT $p_\text{Bonf}=3.4\times 10^{-24}$, SFT>DPO $1.7\times 10^{-13}$.

### Ablation Study
Component contributions from Table 2:

| Configuration | McDiv prompt_gen OCA | ConTest story_gen OCA | Description |
|---|---|---|---|
| $D_{Ca_n}=C\times a_n$ (Full) | 0.846 | 0.828 | Complete metric |
| $a_n$ only (No coherence) | 0.781 | 0.684 | Dropped 6.5pp on McDiv, 14.4pp on story_gen |
| $C$ only (1/PPL) | 0.565 | 0.632 | Slightly better than chance |

The coherence term $C$ is essential for filtering noise; pure noise has high $a_n$ but is suppressed by $C$ (0.04), while one-mode is suppressed by low $a_n$ (0.10).

### Key Findings
- **Coherence weight is vital**: Removing $C$ causes a 14.4pp drop on ConTest story_gen, indicating that poor-quality human responses degrade $a_n$ discriminative power.
- **Stronger models yield sharper metrics**: Qwen2.5-3B's ICL detected commonalities in "shuffled templates" that GPT-2 missed, validating the "auto-sharpening" design.
- **Temperature $\neq$ Creativity**: While $a_n$ correlates highly with sampling temperature ($\rho=+0.932$), this is a sanity check on entropy scaling rather than a direct measure of creativity.
- **McDiv_nuggets artifacts**: Low-diversity sets in nuggets are often paraphrases of high-drama endings, making them inherently more surprising to base models ($a_1$ gap). The metric's success here is partly due to this artifact.

## Highlights & Insights
- **ICL as an Evaluation Microscope**: Unlike previous works using mutual information for training, this approach uses "learning-while-reading" as a zero-shot evaluation probe.
- **Per-byte Normalization + Geometric Mean Coherence**: This combination ensures cross-tokenizer comparability and allows a single poor response to veto the diversity of the entire set.
- **Triplet Definition**: Explicitly defining diversity as relative to (responses, prompt, scoring model) avoids the "fixed embedding" trap and provides a clear upgrade path.
- **Statistical Rigor**: The use of pre-registered tests and Bonferroni corrections transforms qualitative "mode collapse" observations into strong statistical evidence.
- **$\bar a_k$ Curve Visualization**: The shape of the curve (steep drop vs. stable plateau) provides diagnostic value beyond a single scalar.

## Limitations & Future Work
- **Diversity $\neq$ Utility**: The metric is task-agnostic; responses can be diverse but fail the task entirely. It is a "health check" tool, not a quality reward.
- **Scaling Costs**: Computational cost grows with $n^2$ and the number of permutations. Evaluating long-form writing may require larger $\theta$ models (7B+).
- **Truncation Artifacts**: Aligning lengths to the minimum byte count can lead to data loss for shorter models/tasks.
- **Categorical Bias**: Some benchmarks contain inherent surprisal differences in the prompts themselves which can confound diversity signals.

## Related Work & Insights
- **vs $n$-gram / self-BLEU**: Moves beyond surface overlap to distribution-level conditional surprisal, capturing "paraphrased homogeneity."
- **vs SentBERT Clustering**: Replaces external black-box embeddings with an auto-sharpening ICL probe. McDiv gap is ~5pp.
- **vs RLHF Diversity Studies**: Complements qualitative observations with robust statistical evidence across stages (Base > SFT > DPO > RLVR).

## Rating
- **Novelty**: ⭐⭐⭐⭐ Uses ICL as a diversity microscope in a clean, principled way.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Wide range of benchmarks and post-training stages with rigorous statistical controls.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear notation, prudent claims, and excellent use of diagnostic scenarios.
- **Value**: ⭐⭐⭐⭐ An immediately usable open-source diagnostic tool for RLHF and creative generation researchers.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Writing in Symbiosis: Mapping Human Creative Agency in the AI Era](../../NeurIPS2025/llm_nlp/writing_in_symbiosis_mapping_human_creative_agency_in_the_ai_era.md)
- [\[ICML 2026\] Optimizing Diversity and Quality through Base-Aligned Model Collaboration](optimizing_diversity_and_quality_through_base-aligned_model_collaboration.md)
- [\[ICML 2026\] How Many Different Outputs Can a Transformer Generate?](how_many_different_outputs_can_a_transformer_generate.md)
- [\[ICML 2026\] Escaping Mode Collapse in LLM Generation via Geometric Regulation](escaping_mode_collapse_in_llm_generation_via_geometric_regulation.md)
- [\[ACL 2026\] Clozing the Gap: Exploring Why Language Model Surprisal Outperforms Cloze Surprisal](../../ACL2026/llm_nlp/clozing_the_gap_exploring_why_language_model_surprisal_outperforms_cloze_surpris.md)

</div>

<!-- RELATED:END -->
