---
title: >-
  [Paper Note] Is Chain-of-Thought Really Not Explainability? Chain-of-Thought Can Be Faithful without Hint Verbalization
description: >-
  [ACL 2026][LLM Reasoning][Chain-of-Thought] The paper systematically refutes the popular recent conclusion that "CoT does not count as explainability." Using four complementary metrics—Filler Tokens, FUR, faithful@k…
tags:
  - "ACL 2026"
  - "LLM Reasoning"
  - "Chain-of-Thought"
  - "Faithfulness Evaluation"
  - "hint verbalization"
  - "causal mediation analysis"
  - "faithful@k"
date: 2026-05-08
content_hash: dd00aaadd788c4fd
---

# Is Chain-of-Thought Really Not Explainability? Chain-of-Thought Can Be Faithful without Hint Verbalization

**Conference**: ACL 2026  
**arXiv**: [2512.23032](https://arxiv.org/abs/2512.23032)  
**Code**: https://github.com/KeremZaman/IsCotExplainability (Available)  
**Area**: LLM Reasoning / Interpretability / CoT Faithfulness  
**Keywords**: Chain-of-Thought, Faithfulness Evaluation, hint verbalization, causal mediation analysis, faithful@k

## TL;DR
The paper systematically refutes the popular recent conclusion that "CoT does not count as explainability." Using four complementary metrics—Filler Tokens, FUR, faithful@k, and Causal Mediation Analysis—it demonstrates that over half of the CoT samples judged "unfaithful" by Biasing Features (hint verbalization) actually reflect the model's reasoning faithfully through other means. Unfaithfulness primarily stems from "incompleteness" caused by natural language acting as a lossy compression of distributed computation, rather than true misalignment. Increasing the sampling budget allows the hint verbalization probability to reach 90%, and even non-explicitly stated hints can causally propagate influence through the CoT.

## Background & Motivation
**Background**: Over the past two years, the academic community has repeatedly debated whether CoT is trustworthy. The mainstream "CoT is unfaithful" narrative is almost entirely built on the Biasing Features (hint verbalization) metric: injecting a hint (e.g., "A Stanford professor thinks the answer is A") into the input. If the model's answer changes but the CoT does not explicitly mention the hint, it is judged unfaithful. Lanham 2023, Turpin 2023, Chen 2025, and Chua 2025 all used this to reach pessimistic conclusions that $\ge 80\%$ of CoTs are unfaithful.

**Limitations of Prior Work**: (a) This definition is too narrow—it treats "the model not writing the hint down" as "the model's reasoning is inconsistent with the CoT." However, Transformer reasoning is inherently highly distributed, and natural language can only provide lossy compression; missing words $\neq$ false statements. (b) The single-metric perspective reduces faithfulness to one dimension (whether the hint appears), completely ignoring the degree of alignment between CoT and the model's decision-making computation. (c) If this narrative is absorbed into future training pipelines, it will incentivize "verbalization for the sake of verbalization" rather than actually improving interpretability.

**Key Challenge**: Faithfulness $\neq$ completeness. Biasing Features actually measures "verbalized sensitivity to a known intervention," which is a useful reporting measure but has been incorrectly elevated to a synonym for faithfulness.

**Goal**: (1) Re-evaluate samples judged unfaithful by Biasing Features using other existing faithfulness metrics to see if they remain unfaithful; (2) test whether unfaithfulness is simply caused by token constraints by increasing the sampling budget (faithful@k); (3) quantify whether "non-verbalized hints" propagate through the CoT using Causal Mediation Analysis.

**Key Insight**: Break down "the hint not being in the CoT" into two possibilities: incompleteness (incomplete writing) vs. unfaithfulness (no actual influence). Empirical separation of the two is achieved through multi-metrics, multi-k budgets, and causal mediation.

**Core Idea**: Broad descriptions of unfaithfulness are inaccurate. CoT remains a viable interpretability tool. By using multiple validations (FUR / Filler Tokens / faithful@k / CMA), one can avoid being misled by the narrow metric of hint verbalization.

## Method

### Overall Architecture
Ours is an analytical/critical work, with a pipeline consisting of four stages:

1.  **Baseline Replication for Biasing Features**: On OpenbookQA / StrategyQA / ARC-Easy, using 3 hint types (Professor / Metadata / Black Squares) $\times$ 3 instruct models (Llama-3-8B-Instruct, Llama-3.2-3B-Instruct, gemma-3-4b-it), to produce standard results showing "unfaithfulness rates $\ge 80\%$."
2.  **Re-evaluation with Filler Tokens + FUR**: Samples judged unfaithful by Biasing Features are processed through two other metrics.
3.  **faithful@k Testing for Incompleteness**: 128 CoTs are sampled for each item to calculate the probability that the hint is verbalized at least once in $k$ samples, analogous to pass@k.
4.  **Logit Lens + Causal Mediation Analysis**: Trace the hint information flow at the token and layer levels to quantify the proportion of the hint $\to$ prediction effect carried by CoT as a "mediating variable" (NIE vs. NDE).

Finally, key experiments are repeated on Llama-3.3-70B-Instruct (4-bit quantization) and Qwen-3-32B (thinking mode) to verify generalizability to larger/reasoning models.

### Key Designs

1.  **Multi-metric Comparison = Exposing "Unfaithfulness" as a Pseudo-proposition**:
    *   **Function**: Re-examine samples judged unfaithful by Biasing Features using two independent mechanisms: Filler Tokens (replacing CoT with "..." to see if the prediction changes; change = CoT is functional = faithful) and FUR (unlearning a single reasoning step to see if it affects the prediction).
    *   **Mechanism**: $\mathcal{F}_{\mathrm{FT}}=\mathbb{1}[\hat y_{h,\text{corr}}\ne\hat y_h]$ and $\mathcal{F}_{\mathrm{FUR}}=\mathbb{1}[\exists\,r_i: M(x_h)\ne M^{(i)*}(x_h)]$. The former measures "contextual faithfulness" (CoT was indeed used during reasoning), and the latter measures "parametric faithfulness" (CoT reflects true reasoning steps within the model parameters).
    *   **Design Motivation**: The blind spots of a single metric can only be illuminated by complementary metrics. Any independent metric judging "faithful" refutes the "unfaithful" conclusion of Biasing Features. This multi-mirror approach is the paper's strongest logical weapon.

2.  **faithful@k: Separating Token Budget from Confounding Factors**:
    *   **Function**: By increasing the sample size $k$, test whether "unfaithfulness" is essentially "the model simply failed to write the hint in one greedy decoding instance but was originally capable of doing so."
    *   **Mechanism**: Define $\text{faithful@k}=\mathbb{E}[1-\binom{n-c}{k}/\binom{n}{k}]$, where $c$ is the number of samples verbalizing the hint and $n$ is the number of samples where the answer changed to the hinted one. If faithful@k significantly increases as $k$ grows, it indicates that non-verbalization is primarily due to incompleteness; if it remains static, it indicates true unfaithfulness.
    *   **Design Motivation**: Judging unfaithfulness based on a single "missing hint" in greedy decoding is like judging a movie by a single frame. faithful@k explicitly isolates "sampling uncertainty," exposing "latent faithfulness" that cannot be seen in a single trajectory. For the Professor hint, faithful@16 rises to 0.9 (gemma-3-4b), whereas it remains almost static under the Black Squares hint—hint types determine the "model's ability to write out the hint," and this differentiated trajectory is the best defense against "random sampling" counter-arguments.

3.  **Causal Mediation Analysis: Causally Verifying if CoT is a Mediator**:
    *   **Function**: Decompose the "total prediction change caused by adding a hint" into a "Natural Direct Effect (NDE, the hint directly changing the final prediction)" and a "Natural Indirect Effect (NIE, the hint changing the prediction by first altering the CoT)," thereby determining whether CoT is a true causal mediator or a post-hoc rationalization.
    *   **Mechanism**: $\text{NDE}=\mathbb{E}_x[p_h(x_h,c)-p_h(x,c)]$ (fixing the original CoT, changing to hinted input), $\text{NIE}=\mathbb{E}_x[p_h(x,c_h)-p_h(x,c)]$ (fixing the original input, changing to hinted CoT). A significant non-zero NIE implies that CoT causally carries the hint's influence. Simultaneously tracking $p_{\bar h}=\sum_{c\ne L_h}p_c$ distinguishes whether "CoT is raising the hinted answer or depressing other options."
    *   **Design Motivation**: Correlation alone cannot answer whether CoT is the true cause of the model's prediction. CMA directly constructs counterfactual interventions to quantitatively separate the two causal paths. Ours finds that NIE is significantly non-zero almost everywhere, and under the Black Squares hint, NIE often $>$ NDE—meaning that even if the CoT doesn't write the hint, it remains the primary pipe for hint $\to$ prediction.

### Loss & Training
Ours is an analytical/evaluation work and does not use training losses. The unlearning involved in FUR uses Negative Preference Optimization (NPO) + KL constraint from Tutek 2025. Standard learning rates are used for Llama; for gemma-3-4b, a grid search across 7 learning rates was performed to find $5e-6$, which maximized efficacy with specificity $\ge 95\%$. faithful@k uses default sampling for each model (Llama: T=0.6, top-p=0.9; gemma: top-k=64, top-p=0.95; Qwen: top-k=20, top-p=0.95, T=0.6), with 128 samples per instance.

## Key Experimental Results

### Main Results
Comparison of Biasing Features with alternative metrics—proportion of "samples judged unfaithful by Biasing Features but faithful by alternative metrics" across three hint types:

| Model | Hint Type | Filler Tokens Faithfulness | FUR Faithfulness |
| :--- | :--- | :--- | :--- |
| Llama-3.2-3B-Instruct | Black Squares | **60%** | $\ge 50\%$ across all tasks |
| Llama-3.2-3B-Instruct | Professor | 38.6% (ARC-Easy avg) | 65.1% (ARC-Easy) |
| Llama-3.2-3B-Instruct | Metadata | 47.5% (ARC-Easy) | 56.6% (ARC-Easy) |
| Llama-3-8B-Instruct | Black Squares | 50% (ARC-Easy) | 60% |
| Llama-3-8B-Instruct | Professor | 56.4% (ARC-Easy) | 89.3% |
| gemma-3-4b-it | Professor | 45% (ARC-Easy) | 33.6% (ARC-Easy) |

Llama-3.2-3B FUR results are $\ge 50\%$ across all tasks and hints; for Llama-3-8B on OpenbookQA Professor, FUR reaches 86.7%.

Trends in faithful@k (averaged across tasks):

| Model | Professor faithful@1 $\to$ @16 | Black Squares faithful@1 $\to$ @16 |
| :--- | :--- | :--- |
| gemma-3-4b-it | ~0.3 $\to$ **~0.90** | Virtually unchanged |
| Llama-3.2-3B-Instruct | ~0.4 $\to$ ~0.5 | Flat |
| Llama-3-8B-Instruct | Moderate increase | Flat |
| Llama-3.3-70B-Instruct | 0.4 $\to$ > 0.8 (StrategyQA) | Flat |
| Qwen-3-32B (reasoning) | Increases for all hints | Slow increase |

### Ablation Study (Causal Mediation Analysis)

Core conclusions for NDE / NIE (on hinted answer probability) under Professor hints (based on 10,000 bootstrap iterations):

| Model | Task | NDE sig. $\ne 0$ | NIE sig. $\ne 0$ | Magnitude |
| :--- | :--- | :--- | :--- | :--- |
| Llama-3-8B-Instruct | StrategyQA | Yes | **Yes** | NIE > NDE |
| Llama-3-8B-Instruct | OpenbookQA | Yes | **Yes** | NIE > NDE |
| gemma-3-4b-it | OpenbookQA | Yes | Yes | NDE > NIE |
| gemma-3-4b-it | ARC-Easy | Yes | Yes | NDE > NIE |
| Llama-3.2-3B-Instruct | All | Yes | Yes | Comparable |

LLM-as-judge meta-evaluation: The authors manually annotated 100 cases against gpt-oss-20b judge: accuracy 80%, precision 36%, recall 31%, false negative rate only 12% (reducing to 1% with gpt-4o). Re-running Filler Tokens / FUR on a stricter subset where "the hint does not even appear literally" showed identical trends.

### Key Findings
- **Large Models Cannot Fix Biasing Features**: In Llama-3.3-70B and Qwen-3-32B, the unfaithfulness rate remains $\ge 65\%$. However, Filler Tokens still identify up to 72% "substantially faithful" samples under Black Squares hints, making the conclusion highly generalizable.
- **Hint Type Determines Explainability Reachability**: Under Professor hints, faithful@k rises sharply (indicating the model can verbalize, but failed in a single sample); under Black Squares, it remains flat (the hint is too implicit for even large $k$ to trigger verbalization). This differentiated trajectory directly falsifies the "faithful@k is cherry-picking" counter-argument.
- **CoT is a True Causal Mediator, Not Just Post-hoc Rationalization**: NIE is significantly non-zero almost everywhere. For Llama-3-8B on StrategyQA/OpenbookQA, NIE > NDE. Qwen-3-32B even shows negative NIE under Metadata hints (CoT acts as a suppressor), suggesting reasoning models sometimes actively "overrule" explicit hints.
- **Logit Lens Reveals Hint Signal Peaks at Layers 20–25**: Even if CoT does not write the hint, hint-related tokens frequently appear in the top-5 logits of the MHA in middle layers, concentrated in three positions: (a) near "answer," (b) contrastive conjunctions (however/on the other hand), and (c) starts of reasoning step numbering. The latter is crucial, showing hints directly shape the CoT structure.
- **Gemma Shows Maximum Contrast**: High Filler Tokens (context sensitive) but low FUR (weak parametric alignment) proves different models excel in "different dimensions of faithfulness," making any single metric necessarily narrow.
- **Side-effect Control is Sufficient**: All conclusions remain consistent when restricted to the stricter subset where the hint never appears literally, proving robustness against LLM-as-judge recall bias.

## Highlights & Insights
- **Paradigm Correction = Largest Contribution**: At a time when the LLM safety and interpretability communities are dominated by the "CoT cannot be trusted" narrative, this paper provides a sober correction using solid multi-metric experiments. This "counter-narrative" work is especially valuable when the academic atmosphere is swayed by sentiment.
- **Conceptual Distinction: Incompleteness $\neq$ Unfaithfulness**: Distinguishing between "compressed statements" (natural language naturally capturing only a fraction of distributed computation) and "systematic misleading" provides the explainability community with a more refined analytical vocabulary.
- **Clever Design of faithful@k**: Adapting pass@k for faithfulness evaluation makes "latent faithfulness that cannot be seen in greedy decoding" explicit; the differentiated trajectory of hint types directly refutes the "sampling deception" suspicion—this is a truly reusable methodological contribution.
- **CMA + Logit Lens Visualize Non-verbalized Causal Mediation**: While previous work could only say "the hint is not in the CoT," this paper uses a causal framework to prove that "even if not written, CoT remains a causal channel from hint to prediction"—this conclusion fundamentally changes the assessment of CoT interpretability.

## Limitations & Future Work
- Authors acknowledge: (a) faithful@k cannot distinguish between "every reasoning is affected by hint, just occasionally not verbalized" and "only occasionally faithful" at the instance level, though aggregate trends support the former; (b) it cannot distinguish incompleteness from non-exhaustiveness (CoT might reflect one of many reasoning paths); (c) LLM-as-judge recall is only 31%, potentially inflating the "judged unfaithful" base.
- Personal observations: (d) Only verified on multi-hop QA tasks; may not hold for long chain-of-thought mathematical reasoning or code generation; (e) all three hints "suggest answer options," missing more complex interventions like "suggesting intermediate steps" or "misleading evidence"; (f) FUR evaluation can only run on samples where prediction-with-CoT matches prediction-without-CoT, causing small sample sizes in some settings; (g) the paper doesn't explicitly state whether "verbalization-tuning" should be recommended since unfaithfulness is mainly incompleteness—but it warns against optimizing for verbalization to avoid gaming metrics, which is slightly contradictory.
- Improvement ideas: Extend CMA to a "step-by-step" granularity to map the causal contribution of each CoT step to the final prediction; establish a standardized CoT-faithfulness benchmark fusing "verbalization rate / NIE / FUR / Filler Tokens"; study the "compression-faithfulness" trade-off curve under multiple hint injections.

## Related Work & Insights
- **vs. Turpin 2023 / Chen 2025 / Chua 2025**: They used Biasing Features to conclude high CoT unfaithfulness; ours replicates their numbers but provides a substantial rebuttal using three complementary metrics and causal analysis.
- **vs. Lanham 2023 (Filler Tokens / Early Answering)**: Ours reuses Filler Tokens as a control, proving it identifies contextual faithfulness missed by hint-based evaluations.
- **vs. Tutek 2025 (FUR)**: Ours reuses FUR as a "parametric faithfulness" measure, finding Llama models are generally $\ge 50\%$ faithful under this metric, proving hint-based and parametric metrics often yield opposite results.
- **vs. Paul 2024**: They also studied the CoT-prediction relationship using CMA; ours differs by focusing on whether CoT acts as a causal mediator under "hint injection," concluding that CoT indeed carries hint effects.
- **vs. Barez 2025 / Korbak 2025**: They called for causal validation to trust CoT; ours is a concrete implementation of that suggestion, providing a complete evaluation pipeline.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of counter-narrative, faithful@k design, CMA, and Logit Lens is a truly new methodological contribution, though individual metrics have precursors.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 3 datasets $\times$ 3 models $\times$ 3 hints $\times$ 4 metrics + large/reasoning model generalization + LLM-as-judge meta-evaluation + strict subset robustness tests.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear logical argumentation, distinct conceptual layers (faithfulness/completeness/plausibility), and substantial visualizations.
- Value: ⭐⭐⭐⭐⭐ Directly challenges a popular conclusion and provides actionable evaluation advice, offering immediate directional impact for the AI safety/interpretability community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Chain-of-Thought Reasoning in the Wild Is Not Always Faithful](../../ICML2026/llm_reasoning/chain-of-thought_reasoning_in_the_wild_is_not_always_faithful.md)
- [\[ACL 2026\] Learning to Edit Knowledge via Instruction-based Chain-of-Thought Prompting](learning_to_edit_knowledge_via_instruction-based_chain-of-thought_prompting.md)
- [\[ACL 2026\] Decoupling the Effect of Chain-of-Thought Reasoning: A Human Label Variation Perspective](decoupling_the_effect_of_chain-of-thought_reasoning_a_human_label_variation_pers.md)
- [\[ACL 2026\] Render-of-Thought: Rendering Textual Chain-of-Thought as Images for Visual Latent Reasoning](render-of-thought_rendering_textual_chain-of-thought_as_images_for_visual_latent.md)
- [\[ICML 2026\] Hidden Error Awareness in Chain-of-Thought Reasoning: The Signal Is Diagnostic, Not Causal](../../ICML2026/llm_reasoning/hidden_error_awareness_in_chain-of-thought_reasoning_the_signal_is_diagnostic_no.md)

</div>

<!-- RELATED:END -->
