---
title: >-
  [Paper Note] Harnessing Reasoning Trajectories for Hallucination Detection via Answer-agreement Representation Shaping
description: >-
  [ICML 2026][Causal Inference][Large Reasoning Models] This paper proposes ARS for hallucination detection in Large Reasoning Models (LRMs). Instead of perturbing reasoning traces at the text level…
tags:
  - "ICML 2026"
  - "Causal Inference"
  - "Large Reasoning Models"
  - "Hallucination Detection"
  - "Latent Space Perturbation"
  - "Counterfactual Answers"
  - "Contrastive Representation Shaping"
date: 2026-05-08
content_hash: 209b06a4ae5e3a50
---

# Harnessing Reasoning Trajectories for Hallucination Detection via Answer-agreement Representation Shaping

**Conference**: ICML 2026  
**arXiv**: [2601.17467](https://arxiv.org/abs/2601.17467)  
**Code**: https://github.com/radiolab-ntu/ars_icml2026 (Available)  
**Area**: Large Reasoning Model Hallucination Detection / Counterfactual Representation Learning  
**Keywords**: Large Reasoning Models, Hallucination Detection, Latent Space Perturbation, Counterfactual Answers, Contrastive Representation Shaping

## TL;DR
This paper proposes ARS for hallucination detection in Large Reasoning Models (LRMs). Instead of perturbing reasoning traces at the text level, ARS **directly applies minor perturbations to the latent representations at the end of the trace followed by continued decoding** to obtain counterfactual answers. Using "answer agreement" as labels, a lightweight contrastive head is trained to shape trace-conditioned answer embeddings, enabling subsequent embedding-based detectors to better distinguish hallucinations from truthful responses (AUROC improved from $66.85 \to 86.64$ on TruthfulQA).

## Background & Motivation
**Background**: LRMs (such as Qwen3, DeepSeek-R1) generate long reasoning traces before producing answers. Common hallucination detection methods fall into four categories: (i) logit/perplexity-based uncertainty; (ii) multi-sampling consistency (Semantic Entropy, SelfCKGPT); (iii) verbalized confidence; (iv) embedding-based probes (CCS, HaloScope, EigenScore) from internal states.

**Limitations of Prior Work**: Using reasoning traces directly as signals empirically results in **decreased** performance—the authors compared "with/without trace" representations using Qwen3-8B on TruthfulQA and found that traces obscure answer-level hallucination signals (Fig. 1). There are two reasons: (1) multiple traces can support the same answer, and the surface form of traces varies significantly, causing detectors to overfit to style rather than validity; (2) hallucinations are inherently answer-level properties, but traces span many tokens and layers, where irrelevant stylistic variations drown out the true signals.

**Key Challenge**: Traces **do** contain signals regarding answer stability—the intuition is that "truthful answers are stable in internal representations, while hallucinated answers are fragile and flip under small perturbations"—but standard embeddings contain both this stability signal and substantial stylistic noise, making them difficult for detectors to utilize directly.

**Goal**: (1) Extract answer-centric stability signals from traces; (2) inject these signals into trace-conditioned answer embeddings to benefit downstream embedding-based detectors; (3) avoid reliance on human hallucination annotations and multi-sampling during inference.

**Key Insight**: Treating the trace as a given context, the focus is placed on the moment the trace ends and the answer begins—the hidden state $\boldsymbol h$ of the last trace token in the penultimate layer. This represents the state where the model has "processed all reasoning but not yet locked into an answer." Continued decoding from this state with small perturbations provides counterfactual answers that cleanly reflect the model's commitment to the current answer.

**Core Idea**: Generate counterfactual answers via latent space perturbations $\to$ use "counterfactual-original answer agreement" as contrastive labels $\to$ train a lightweight projection to explicitly shape the stability signal into the embeddings.

## Method

### Overall Architecture
Freeze the LRM $\pi_\theta$. For a sample $(\boldsymbol x, \boldsymbol r, \boldsymbol a)$ (prompt / trace / answer):

1. Extract the penultimate layer hidden state of the last trace token: $\boldsymbol h = \boldsymbol h_L(\boldsymbol x\oplus\boldsymbol r)$.
2. Sample $M$ Gaussian perturbations $\boldsymbol\delta_j\sim\mathcal N(0,\sigma^2\boldsymbol I)$, and decode from $\boldsymbol h + \boldsymbol\delta_j$ to obtain counterfactual answers $\tilde{\boldsymbol a}_j$ and their embeddings $\tilde{\boldsymbol u}_j$.
3. Use an LLM-as-judge to determine $\text{Agr}(\boldsymbol a, \tilde{\boldsymbol a}_j)\in\{0,1\}$, partitioning $\tilde{\boldsymbol u}_j$ into an agreement set $\mathcal U^+$ or a disagreement set $\mathcal U^-$.
4. Train a lightweight linear mapping $g_\phi: \mathbb R^d\to\mathbb R^k$ (a single linear projection without bias) such that $\boldsymbol z = g_\phi(\boldsymbol u)$ pulls $\mathcal U^+$ closer and pushes $\mathcal U^-$ further away (InfoNCE objective).

During testing: Run a single forward pass for a new sample, extract $\boldsymbol u$, pass it through $g_\phi$ to get $\boldsymbol z$, and feed it into any embedding-based detector (CCS, Probing, HaloScope, EigenScore) for binary classification.

### Key Designs

1. **Latent Space Perturbation at the Trace Boundary $\to$ Counterfactual Answers**:
    - **Function**: Generate "alternative answers the model might produce under its current internal state" at minimal cost, shifting multi-sampling costs from inference to one-time training.
    - **Mechanism**: $\tilde{\boldsymbol h}=\boldsymbol h + \boldsymbol\delta,\ \boldsymbol\delta\sim\mathcal N(0,\sigma^2\boldsymbol I)$, followed by $\tilde{\boldsymbol a}=\text{Decode}_\theta(\boldsymbol x\oplus\boldsymbol r;\tilde{\boldsymbol h})$. The perturbation location is intentionally chosen at the **trace-end / answer-start** boundary. The paper explains that mid-trace perturbations cause subsequent reasoning to be rewritten, where style dominates over answer changes; mid-answer perturbations are overly constrained by already generated tokens. The trace boundary is the position of "maximum answer freedom" after the model has fully absorbed the reasoning.
    - **Design Motivation**: Previous methods either perturbed traces in text space (deletions/reordering), which requires careful design and often alters semantics, or used multi-sample output spaces, increasing inference cost $\times M$. Directly manipulating the latent space is inexpensive and reflects the model's own decision geometry without needing text-level design.

2. **Answer Agreement as an Automated Supervision Signal**:
    - **Function**: Obtain contrastive pairs without human annotation.
    - **Mechanism**: For each original sample $(\boldsymbol x, \boldsymbol r, \boldsymbol a)$, produce $M$ $\tilde{\boldsymbol a}_j$ and partition them using $\text{Agr}$ (instantiated via text similarity or LLM-as-judge). Note that $\text{Agr}$ **does not require the ground truth $y$**—it only judges if the counterfactual answer is equivalent to the original. $\mathcal U^+=\{\tilde{\boldsymbol u}_j: \text{Agr}=1\}$ collects states that still lead to the same answer under perturbation; $\mathcal U^-=\{\tilde{\boldsymbol u}_j:\text{Agr}=0\}$ collects states that fail under perturbation. Intuition: Hallucinated samples usually have larger $\mathcal U^-$ (smaller stability margins), while truthful samples are dominated by $\mathcal U^+$.
    - **Design Motivation**: Distill the model's own decision stability into a training signal without needing ground truth hallucination labels. This allows ARS to be trained unsupervised (zero supervision for the entire pipeline when paired with CCS) while remaining compatible with supervised probing.

3. **InfoNCE-style Shaping Objective**:
    - **Function**: Explicitly encode stability signals into the $\boldsymbol z$ geometry using a contrastive loss.
    - **Mechanism**: Using the original answer's $\boldsymbol z = g_\phi(\boldsymbol u)$ as an anchor, positive examples $\tilde{\boldsymbol z}^+ \sim g_\phi(\mathcal U^+)$, and a set of negative examples $\mathcal Z^- = g_\phi(\mathcal U^-)$, the loss is $\mathcal L_{\text{ARS}}=-\frac{\text{sim}(\boldsymbol z,\tilde{\boldsymbol z}^+)}{\tau}+\log\sum_{\tilde{\boldsymbol z}'\in\{\tilde{\boldsymbol z}^+\}\cup\mathcal Z^-}\exp(\frac{\text{sim}(\boldsymbol z,\tilde{\boldsymbol z}')}{\tau})$, where sim is cosine similarity. The mapping $g_\phi$ is a single linear projection—highly lightweight. Proposition 4.2 provides a bound $\Pr(\hat y\neq y)\leq C(1-\eta_\phi)+e_\alpha$, splitting the error rate into answer stability separability $e_\alpha$ and shaping success $1-\eta_\phi$; optimizing $\mathcal L_{\text{ARS}}$ directly tightens the second term.
    - **Design Motivation**: Transform the intuition of "hallucination = instability" from a multi-sample inference requirement (like Semantic Entropy) into a geometric property obtainable via a single forward pass, ensuring plug-and-play compatibility with existing detectors.

### Loss & Training
- $\mathcal L_{\text{ARS}}$ as defined above; Adam optimizer, lr $1\text{e-}4$, weight decay $1\text{e-}5$, cosine decay, batch size 128.
- $g_\phi$ implemented as a single linear projection; input is the LRM's penultimate layer embedding of the final answer token (following Azaria & Mitchell 2023).
- Hyperparameters $\sigma, k, \tau, M$ and the training layer are selected on a 100-sample validation split; 25% of TruthfulQA is used for testing.

## Key Experimental Results

### Main Results

| Model | Dataset | Detector | Vanilla AUROC | ARS-Shaped AUROC | Gain |
|------|--------|----------|--------------|------------------|------|
| Qwen3-8B | TruthfulQA | CCS | 66.85 | **86.64** | $+19.79$ |
| Qwen3-8B | TriviaQA | CCS | 59.24 | **88.54** | $+29.30$ |
| Qwen3-8B | GSM8K | CCS | 57.98 | **90.37** | $+32.39$ |
| Qwen3-8B | MATH-500 | CCS | 55.64 | **78.66** | $+23.02$ |
| Qwen3-8B | TruthfulQA | Probing | 78.66 | **83.66** | $+5.00$ |
| Qwen3-8B | MATH-500 | Probing | 67.03 | **78.17** | $+11.14$ |
| DeepSeek-R1-Distill-Llama-8B | TriviaQA | CCS | 63.99 | **88.86** | $+24.87$ |
| DeepSeek-R1-Distill-Llama-8B | MATH-500 | CCS | 54.44 | **86.38** | $+31.94$ |

| Dataset | Model | Ours (ARS) | TSV (Park 2025) | G-Detector (Zhang 2026) | Semantic Entropy |
|--------|------|-----|----------------|-------------------------|-----------------|
| TruthfulQA | Qwen3-8B | **86.64** | 77.08 | 71.86 | 65.60 |
| TriviaQA | Qwen3-8B | **91.62** (Probing) | 89.67 | 90.52 | 58.37 |
| GSM8K | Qwen3-8B | **90.37** | 83.15 | 83.78 | 72.51 |
| MATH-500 | Qwen3-8B | **78.66** | 63.12 | 57.67 | 56.13 |

### Ablation Study

| Configuration | TruthfulQA AUROC | Description |
|------|----------------|------|
| ARS (trace-boundary intervention) | **86.64** | Default |
| Mid-trace perturbation | Significant Drop | Answer changes entangled with trace style changes |
| Mid-answer perturbation | Significant Drop | Subsequent tokens constrained by answer history |
| Text deletion (10–90%) | Worse than ARS | Sensitive text perturbation design |
| Text mask / paraphrase | Same as above | Same as above |
| Cross-dataset (GSM8K→TriviaQA) | 87.80 | Close to in-domain 91.62, strong transfer |
| Qwen3-14B (Larger model) | 77.47 (vs TSV 73.41, G-Det 69.89) | Still leads after scaling |

### Key Findings
- **Trace boundary is the correct intervention point**: Ablations on mid-trace/mid-answer/text perturbations confirm trace-end superiority—offering a non-obvious practical insight on utilizing reasoning traces.
- **CCS (Unsupervised) + ARS outperforms Probing (Supervised) + ARS**: On TruthfulQA and GSM8K, CCS-ARS exceeds Probing-ARS, suggesting that shaped embeddings are sufficiently separated such that unsupervised CCS utilizes them optimally, narrowing the "labeled vs unlabeled" gap.
- **Strong cross-domain transfer**: $g_\phi$ trained on GSM8K still achieves 87.80 on TriviaQA, indicating ARS captures dataset-agnostic stability geometry rather than overfitting to surface style.
- **Scale-friendly**: Consistently outperforms strong baselines (TSV/G-Detector) on the 14B model.
- **Zero additional inference sampling**: Unlike Semantic Entropy/SelfCKGPT requiring $M$ forward passes, ARS testing requires only one forward pass and a linear projection, making it deployment-friendly.

## Highlights & Insights
- **Latent perturbation replaces text perturbation**: Adding small Gaussian noise to hidden states and continuing decoding avoids the semantic/formatting design issues of text perturbation. This clean trick is transferable to any task studying input-to-answer stability.
- **Train-time perturbation / Clean inference**: The cost of multi-sampling is amortized to the training phase. One forward pass to get shaped embeddings during inference follows an "expensive train / cheap test" pattern highly valuable for LLM deployment.
- **Answer agreement as self-supervision**: $\text{Agr}$ can be judged by the model itself, requiring zero human annotation. This allows ARS to have nearly zero cold-start costs for new domains/models.
- **Theoretical and algorithmic alignment**: Proposition 4.2 links the contrastive loss optimization of $\eta_\phi$ directly to the detection error bound. This "loss directly corresponding to provable bound" paradigm is rare in representation learning.
- **Physical intuition of the trace boundary**: The authors explain this as a state of "full reasoning absorption without answer commitment," which aligns with human intuition regarding deliberation.

## Limitations & Future Work
- $\text{Agr}$ is implemented via LLM-as-judge; quality depends on the judge. If the judge hallucinates, contrastive pairs are corrupted (Qwen3-32B was used to mitigate this, but judge selection was not ablated).
- Isotropic Gaussian noise may not be optimal—model sensitivity varies greatly across directions; future work could consider adaptive perturbations along principal components or Fisher information directions.
- Benchmarks are primarily QA/Math; the precision for long-form generation (e.g., summarization) is yet to be verified.
- $g_\phi$ is a single linear projection; its capacity may be limited. If internal representations suffer from geometric collapse (e.g., after RLHF), linear projections might not recover the stability signal.
- The paper's placement in a causal inference context is slightly imprecise (core is counterfactual, but the task is trustworthy LLM); it is recommended to reclassify this under hallucination/llm_reasoning subfields.

## Related Work & Insights
- **vs Semantic Entropy (Kuhn 2023)**: Both share the "hallucination = instability" intuition, but SE requires $M$ output space samples. ARS moves this to a training-time one-off with zero inference overhead and shapes signals into embeddings rather than just a score.
- **vs HaloScope / CCS / Probing**: These methods use raw embeddings. ARS acts as an *upstream* enhancement—shaping raw embeddings through $g_\phi$ before passing them to these detectors, making it orthogonally compatible.
- **vs TSV (Park 2025)**: TSV modifies representations but needs (semi-)supervised signals; ARS uses zero-supervision via answer agreement, outperforming TSV by ~9.5 points on TruthfulQA.
- **vs RHD / RACE / G-Detector (LRM-specific)**: These focus on trace text/structure without reshaping at the representation layer. ARS shapes representations directly, bypassing stylistic noise.
- **Inspiration**: The paradigm of using latent space perturbation + continued decoding to create counterfactual labels can be extended to feature importance in interpretability, jailbreak robustness in safety alignment, and tool-use stability in agents.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ "Latent perturbation + answer agreement contrastive shaping" is a first and highly natural approach in hallucination detection.
- Experimental Thoroughness: ⭐⭐⭐⭐ 4 datasets × 2 main models + 14B scaling + cross-domain transfer + 4 downstream detectors + various perturbation position ablations.
- Writing Quality: ⭐⭐⭐⭐ Physical intuition of the trace boundary is clear; method diagrams and theoretical-algorithmic links are well-defined.
- Value: ⭐⭐⭐⭐⭐ Zero inference overhead, zero human annotation, strong cross-domain transfer, and plug-and-play capability make it ready for industrial deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] MUG: Multi-agent Undercover Gaming — Hallucination Removal via Counterfactual Test for Multimodal Reasoning](../../AAAI2026/causal_inference/multi-agent_undercover_gaming_hallucination_removal_via_coun.md)
- [\[ICML 2026\] Unveiling the Structure of Do-Calculus Reasoning via Derivation Graphs](unveiling_the_structure_of_do-calculus_reasoning_via_derivation_graphs.md)
- [\[ICLR 2026\] SelfReflect: Can LLMs Communicate Their Internal Answer Distribution?](../../ICLR2026/causal_inference/selfreflect_can_llms_communicate_their_internal_answer_distribution.md)
- [\[ICLR 2026\] Journey to the Centre of Cluster: Harnessing Interior Nodes for A/B Testing under Network Interference](../../ICLR2026/causal_inference/journey_to_the_centre_of_cluster_harnessing_interior_nodes_for_ab_testing_under_.md)
- [\[CVPR 2026\] Fighting Hallucinations with Counterfactuals: Diffusion-Guided Perturbations for LVLM Hallucination Suppression](../../CVPR2026/causal_inference/fighting_hallucinations_with_counterfactuals_diffusion-guided_perturbations_for_.md)

</div>

<!-- RELATED:END -->
