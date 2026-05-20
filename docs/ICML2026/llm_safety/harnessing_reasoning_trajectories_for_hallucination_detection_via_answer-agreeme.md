---
title: >-
  [Paper Note] Harnessing Reasoning Trajectories for Hallucination Detection via Answer-agreement Representation Shaping
description: >-
  [ICML 2026][LLM Safety][Large Reasoning Model] This paper proposes ARS for hallucination detection in large reasoning models (LRMs): instead of perturbing the reasoning trace at the text level…
tags:
  - "ICML 2026"
  - "LLM Safety"
  - "Large Reasoning Model"
  - "Hallucination Detection"
  - "Latent Space Perturbation"
  - "Counterfactual Answer"
  - "Contrastive Representation Shaping"
date: 2026-05-08
content_hash: e1d1b5a9f1ba2175
---

# Harnessing Reasoning Trajectories for Hallucination Detection via Answer-agreement Representation Shaping

**Conference**: ICML 2026  
**arXiv**: [2601.17467](https://arxiv.org/abs/2601.17467)  
**Code**: https://github.com/radiolab-ntu/ars_icml2026 (available)  
**Area**: Large Reasoning Model Hallucination Detection / Counterfactual Representation Learning  
**Keywords**: Large Reasoning Model, Hallucination Detection, Latent Space Perturbation, Counterfactual Answer, Contrastive Representation Shaping

## TL;DR
This paper proposes ARS for hallucination detection in large reasoning models (LRMs): instead of perturbing the reasoning trace at the text level, it **directly applies small perturbations to the latent representation at the end of the trace and continues decoding** to obtain counterfactual answers. Using "answer agreement" as a label, a lightweight contrastive head is trained to shape the trace-conditioned answer embedding, enabling subsequent embedding-based detectors to better separate hallucinations from truthful answers (AUROC on TruthfulQA improves from $66.85\to 86.64$).

## Background & Motivation
**Background**: LRMs (e.g., Qwen3, DeepSeek-R1) first generate a long reasoning trace and then produce an answer. Common hallucination detection methods fall into four categories: (i) logit/perplexity-based uncertainty; (ii) multi-sample consistency (Semantic Entropy, SelfCKGPT); (iii) prompting the model to verbally report confidence; (iv) embedding-based probes (CCS, HaloScope, EigenScore) that classify from internal states.

**Limitations of Prior Work**: Directly using the reasoning trace as a signal empirically **decreases rather than increases** performance—the authors compare "with/without trace" representations on Qwen3-8B for TruthfulQA and find that the trace actually masks answer-level hallucination signals (Fig. 1). Two reasons: (1) The same answer can be supported by many traces, whose surface forms vary greatly, causing detectors to overfit to style rather than answer validity; (2) Hallucination is inherently an answer-level property, but the trace spans many tokens and layers, so irrelevant style variations drown out the true signal.

**Key Challenge**: The trace **does** contain information about answer stability—the intuition is "truthful answers are stable in internal representation, hallucinated answers are fragile and change with small perturbations"—but conventional embeddings contain both this stability signal and a lot of style noise, making it unusable by detectors.

**Goal**: (1) Extract answer-centric stability signals from the trace; (2) Inject this signal into the trace-conditioned answer embedding so any downstream embedding-based detector can benefit; (3) Avoid manual hallucination labeling and multi-sample inference.

**Key Insight**: Treat the trace as context and focus on the moment when the trace ends and the answer begins—i.e., the hidden state of the last token of the reasoning trace at the penultimate layer, $\boldsymbol h$. This is the state where the model has "seen all reasoning but has not yet committed to an answer." Perturbing this state and continuing decoding to obtain counterfactual answers most cleanly reflects "how strongly the model is committed to the current answer."

**Core Idea**: Use latent space perturbation to generate counterfactual answers → use "whether the counterfactual matches the original answer" as a contrastive label → train a lightweight projection to explicitly shape the stability signal into the embedding.

## Method

### Overall Architecture
Freeze the LRM $\pi_\theta$. For a sample $(\boldsymbol x, \boldsymbol r, \boldsymbol a)$ (prompt / trace / answer):

1. Take the penultimate layer hidden state of the last token in the trace: $\boldsymbol h = \boldsymbol h_L(\boldsymbol x\oplus\boldsymbol r)$.
2. Sample $M$ Gaussian perturbations $\boldsymbol\delta_j\sim\mathcal N(0,\sigma^2\boldsymbol I)$, and from $\boldsymbol h + \boldsymbol\delta_j$ continue decoding to obtain counterfactual answers $\tilde{\boldsymbol a}_j$ and their embeddings $\tilde{\boldsymbol u}_j$.
3. Use LLM-as-judge to determine $\text{Agr}(\boldsymbol a, \tilde{\boldsymbol a}_j)\in\{0,1\}$, assigning $\tilde{\boldsymbol u}_j$ to the agreement set $\mathcal U^+$ or disagreement set $\mathcal U^-$.
4. Train a lightweight linear mapping $g_\phi: \mathbb R^d\to\mathbb R^k$ (single linear projection without bias) so that $\boldsymbol z = g_\phi(\boldsymbol u)$ brings $\mathcal U^+$ closer and pushes $\mathcal U^-$ further apart (InfoNCE style).

At test time: For a new sample, run a single forward pass to get $\boldsymbol u$, pass through $g_\phi$ to get $\boldsymbol z$, and feed it to any embedding-based detector (CCS, Probing, HaloScope, EigenScore) for binary classification.

### Key Designs

1. **Latent Space Perturbation at Trace Boundary → Counterfactual Answers**:

    - Function: Generate "other possible answers the model could produce from its current internal state" at minimal cost, shifting the multi-sample cost from inference to one-time training.
    - Mechanism: $\tilde{\boldsymbol h}=\boldsymbol h + \boldsymbol\delta,\ \boldsymbol\delta\sim\mathcal N(0,\sigma^2\boldsymbol I)$, then $\tilde{\boldsymbol a}=\text{Decode}_\theta(\boldsymbol x\oplus\boldsymbol r;\tilde{\boldsymbol h})$. The perturbation is deliberately applied at the **trace end / answer start** boundary—the paper explains: perturbing mid-trace rewrites the entire reasoning style, so the effect is dominated by trace style rather than answer change; perturbing mid-answer is constrained by already generated answer tokens, allowing only local edits without semantic reversal. The trace boundary is where the model has fully absorbed the reasoning but not yet committed to an answer, i.e., the point of maximal answer freedom.
    - Design Motivation: Previous methods either perturb the trace in text space (deletion/reordering/paraphrase), which requires careful design and often changes semantics; or use multi-sample output space, incurring $M\times$ inference cost. Directly perturbing latent space is cheap and leverages the model's own decision geometry, requiring no text design.

2. **Answer Agreement as Automatic Supervision Signal**:

    - Function: Obtain contrastive pairs with zero manual labeling.
    - Mechanism: For each original sample $(\boldsymbol x, \boldsymbol r, \boldsymbol a)$, generate $M$ $\tilde{\boldsymbol a}_j$, and use $\text{Agr}$ (instantiated by text similarity or LLM-as-judge) to partition them. Note that $\text{Agr}$ **does not require ground truth $y$**—it only checks whether the counterfactual answer is equivalent to the original. $\mathcal U^+=\{\tilde{\boldsymbol u}_j: \text{Agr}=1\}$ collects "internal states that still yield the same answer under small perturbations"; $\mathcal U^-=\{\tilde{\boldsymbol u}_j:\text{Agr}=0\}$ collects "internal states where small perturbations flip the answer." Intuition: hallucinated samples have larger $\mathcal U^-$ (smaller stability margin), while truthful samples are dominated by $\mathcal U^+$.
    - Design Motivation: Distill the model's own decision stability into a training signal, without needing hallucination ground truth—enabling ARS to be trained unsupervised (the entire CCS pipeline is zero supervision), while also being compatible with supervised Probing.

3. **InfoNCE-style Shaping Objective**:

    - Function: Use contrastive loss to explicitly encode the stability signal into the geometry of $\boldsymbol z$.
    - Mechanism: Take the original answer's $\boldsymbol z = g_\phi(\boldsymbol u)$ as anchor, positive sample $\tilde{\boldsymbol z}^+ \sim g_\phi(\mathcal U^+)$, negative set $\mathcal Z^- = g_\phi(\mathcal U^-)$, and loss $\mathcal L_{\text{ARS}}=-\frac{\text{sim}(\boldsymbol z,\tilde{\boldsymbol z}^+)}{\tau}+\log\sum_{\tilde{\boldsymbol z}'\in\{\tilde{\boldsymbol z}^+\}\cup\mathcal Z^-}\exp(\frac{\text{sim}(\boldsymbol z,\tilde{\boldsymbol z}')}{\tau})$, where sim is cosine similarity. The mapping $g_\phi$ is a single linear projection without bias—extremely lightweight. Theoretically, Proposition 4.2 gives $\Pr(\hat y\neq y)\leq C(1-\eta_\phi)+e_\alpha$, decomposing detection error into "whether answer stability separates true/false" $e_\alpha$ and "whether shaping successfully separates positive/negative pairs" $1-\eta_\phi$; optimizing $\mathcal L_{\text{ARS}}$ directly tightens the second term.
    - Design Motivation: Transform the intuition "hallucination = instability" from requiring multi-sample inference (e.g., Semantic Entropy) into a geometric property computable in a single forward pass, plug-and-play compatible with existing detectors, without modifying downstream models.

### Loss & Training

- $\mathcal L_{\text{ARS}}$ as above; Adam optimizer, lr $1\text{e-}4$, weight decay $1\text{e-}5$, cosine decay, batch size 128.
- $g_\phi$ is implemented as a single linear projection; input is the embedding of the last answer token at the penultimate layer of the LRM (per Azaria & Mitchell 2023).
- Hyperparameters $\sigma, k, \tau, M$ and training layer are selected on a 100-sample validation split; TruthfulQA uses 25% for testing.

## Key Experimental Results

### Main Results

| Model | Dataset | Detector | Vanilla AUROC | ARS-Shaped AUROC | Gain |
|-------|---------|----------|--------------|------------------|------|
| Qwen3-8B | TruthfulQA | CCS | 66.85 | **86.64** | $+19.79$ |
| Qwen3-8B | TriviaQA | CCS | 59.24 | **88.54** | $+29.30$ |
| Qwen3-8B | GSM8K | CCS | 57.98 | **90.37** | $+32.39$ |
| Qwen3-8B | MATH-500 | CCS | 55.64 | **78.66** | $+23.02$ |
| Qwen3-8B | TruthfulQA | Probing | 78.66 | **83.66** | $+5.00$ |
| Qwen3-8B | MATH-500 | Probing | 67.03 | **78.17** | $+11.14$ |
| DeepSeek-R1-Distill-Llama-8B | TriviaQA | CCS | 63.99 | **88.86** | $+24.87$ |
| DeepSeek-R1-Distill-Llama-8B | MATH-500 | CCS | 54.44 | **86.38** | $+31.94$ |

| Dataset | Model | ARS | TSV (Park 2025) | G-Detector (Zhang 2026) | Semantic Entropy |
|---------|-------|-----|----------------|-------------------------|-----------------|
| TruthfulQA | Qwen3-8B | **86.64** | 77.08 | 71.86 | 65.60 |
| TriviaQA | Qwen3-8B | **91.62** (Probing) | 89.67 | 90.52 | 58.37 |
| GSM8K | Qwen3-8B | **90.37** | 83.15 | 83.78 | 72.51 |
| MATH-500 | Qwen3-8B | **78.66** | 63.12 | 57.67 | 56.13 |

### Ablation Study

| Configuration | TruthfulQA AUROC | Notes |
|---------------|------------------|-------|
| ARS (trace-boundary intervention) | **86.64** | Default |
| Mid-trace perturbation | Significant drop | Answer change entangled with trace style |
| Mid-answer perturbation | Significant drop | Subsequent tokens strongly constrained by answer, only surface edits possible |
| Text deletion (10–90%) | All worse than ARS | Text perturbation is design-sensitive |
| Text mask / paraphrase | Same as above | Same as above |
| Cross-dataset (GSM8K→TriviaQA) | 87.80 | Close to in-domain 91.62, strong transfer |
| Qwen3-14B (larger model) | 77.47 (vs TSV 73.41, G-Det 69.89) | Still leads after scaling up |

### Key Findings
- **Trace boundary is the only correct perturbation point**: Ablations show mid-trace/mid-answer/text perturbations all underperform, with trace end clearly superior—providing a non-obvious practical insight for "how to use reasoning traces."
- **CCS (unsupervised) + ARS outperforms Probing (supervised) + ARS**: On TruthfulQA and GSM8K, CCS-ARS surpasses Probing-ARS, indicating that the shaped embedding is already sufficiently separable, and unsupervised CCS maximally exploits it. This means ARS greatly narrows the gap between "with vs. without labels."
- **Strong cross-domain transfer**: $g_\phi$ trained on GSM8K achieves 87.80 on TriviaQA, indicating ARS captures dataset-agnostic stability geometry rather than overfitting to surface style.
- **Scale-friendly**: Still stably outperforms the strongest baselines (TSV/G-Detector) on 14B models.
- **Zero extra sampling at inference**: Unlike Semantic Entropy / SelfCKGPT, which require $M$ forward passes, ARS needs only one forward + linear projection at test time, making it highly practical for industrial deployment.

## Highlights & Insights
- **Latent space perturbation replaces text perturbation**: Directly adding small Gaussian noise to the hidden state and continuing decoding bypasses all semantic/format design issues of text perturbation—a very clean trick, transferable to any task studying "input space vs. answer space stability."
- **Perturb at training / clean at inference**: Amortizes the cost of multi-sample perturbation to training, so inference only requires a single forward to obtain the shaped embedding—this "expensive train / cheap test" pattern is especially reusable in large model deployment.
- **Answer agreement as self-supervision**: $\text{Agr}$ can be judged entirely by the model itself, with zero manual labeling; this allows ARS to have almost zero cold-start cost in new domains/models.
- **Theory and algorithm strictly aligned**: Proposition 4.2 directly links the contrastive loss optimization $\eta_\phi$ to the detection error bound—this "loss directly corresponds to a provable bound" paradigm is rare in representation learning.
- **Physical intuition for the trace boundary anchor**: The authors explain it as the state where the model has fully absorbed the reasoning but not yet committed to an answer, closely matching human intuition about deliberation.

## Limitations & Future Work
- $\text{Agr}$ is implemented via LLM-as-judge, so quality depends on the judge model; if the judge itself hallucinates, contrastive pairs may be contaminated (the paper uses Qwen3-32B as judge to partially mitigate this, but does not ablate judge choice).
- Isotropic Gaussian perturbation may not be optimal—model sensitivity varies greatly by direction; future work could consider adaptive perturbation along principal components or Fisher information directions.
- Main benchmarks are QA/math; it remains untested whether hallucination in long-form generation (e.g., summarization) can be precisely detected.
- $g_\phi$ is a single linear projection, so its capacity may be limited; when LRM internal representations are already geometrically collapsed (e.g., after RLHF), linear projection may not recover the stability signal.
- The paper's placement in the causal_inference folder is not ideal (core is counterfactual but the task is trustworthy LLM); future reclassification should move it to hallucination/llm_reasoning subfields.

## Related Work & Insights
- **vs Semantic Entropy (Kuhn 2023)**: Both are based on "hallucination = instability," but SE requires $M$ output space samples, while ARS shifts this to one-time training and zero inference cost, shaping the signal into the embedding rather than outputting a single score.
- **vs HaloScope / CCS / Probing**: These methods use raw embeddings; ARS is their *upstream* enhancer—raw embeddings are shaped via $g_\phi$ before being passed to them, so ARS is orthogonally compatible with all embedding-based detectors.
- **vs TSV (Park 2025)**: TSV also modifies representations but requires (semi-)supervised signals; ARS uses answer agreement for complete zero-supervision, outperforming TSV by about 9.5 points on TruthfulQA.
- **vs RHD / RACE / G-Detector (LRM-specific)**: These start from trace text/structure, not representation reshaping; ARS directly shapes representations, bypassing text style noise.
- **Insights**: The paradigm of latent space perturbation + continued decoding to generate counterfactual labels can be extended to—feature importance in interpretability, jailbreak robustness in safety alignment, tool-use stability in agents, and nearly any "answer robustness" problem.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ "Latent space perturbation + answer agreement contrastive shaping" is the first of its kind in hallucination detection and very natural
- Experimental Thoroughness: ⭐⭐⭐⭐ 4 datasets × 2 main models + 14B extension + cross-domain transfer + 4 downstream detectors + multiple perturbation ablations
- Writing Quality: ⭐⭐⭐⭐ Physical intuition for trace boundary is well explained, method diagrams are clear, theory and algorithm are tightly linked
- Value: ⭐⭐⭐⭐⭐ Zero inference cost, zero manual labeling, strong cross-domain transfer, plug-and-play, immediately deployable in industry

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Tracing the Dynamics of Refusal: Exploiting Latent Refusal Trajectories for Robust Jailbreak Detection](tracing_the_dynamics_of_refusal_exploiting_latent_refusal_trajectories_for_robus.md)
- [\[ACL 2026\] Enhancing Hallucination Detection via Future Context](../../ACL2026/llm_safety/enhancing_hallucination_detection_via_future_context.md)
- [\[ICLR 2026\] Enhancing Hallucination Detection through Noise Injection](../../ICLR2026/llm_safety/enhancing_hallucination_detection_through_noise_injection.md)
- [\[ICML 2026\] Watermarking LLM Agent Trajectories (ACTHOOK)](watermarking_llm_agent_trajectories.md)
- [\[ICLR 2026\] Veritas: Generalizable Deepfake Detection via Pattern-Aware Reasoning](../../ICLR2026/llm_safety/veritas_generalizable_deepfake_detection_via_pattern-aware_reasoning.md)

</div>

<!-- RELATED:END -->
