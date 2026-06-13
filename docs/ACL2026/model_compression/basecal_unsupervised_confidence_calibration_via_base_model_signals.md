---
title: >-
  [Paper Note] BaseCal: Unsupervised Confidence Calibration via Base Model Signals
description: >-
  [ACL 2026][Model Compression][confidence calibration] Observing that base LLMs remain well-calibrated on free-form QA while post-trained LLMs (PoLLM) are severely overconfident…
tags:
  - "ACL 2026"
  - "Model Compression"
  - "confidence calibration"
  - "post-trained LLM"
  - "base model"
  - "hidden state projection"
  - "unsupervised"
date: 2026-05-08
content_hash: 66055c5760aab918
---

# BaseCal: Unsupervised Confidence Calibration via Base Model Signals

**Conference**: ACL 2026  
**arXiv**: [2601.03042](https://arxiv.org/abs/2601.03042)  
**Code**: https://github.com/Tan-Hexiang/BaseCal (Available)  
**Area**: Model Calibration / LLM Reliability  
**Keywords**: confidence calibration, post-trained LLM, base model, hidden state projection, unsupervised

## TL;DR
Observing that base LLMs remain well-calibrated on free-form QA while post-trained LLMs (PoLLM) are severely overconfident, BaseCal proposes two unsupervised schemes: feeding PoLLM responses into the base LLM to derive token probabilities as confidence (BaseCal-ReEval), or using a linear projection layer to map PoLLM's final-layer hidden states back to the base LLM space and passing them through the base output layer (BaseCal-Proj). This approach reduces ECE by an average of 42.9% relative to the best unsupervised baseline across 5 datasets $\times$ 3 model families.

## Background & Motivation
**Background**: Reliable confidence is a core lever for mitigating LLM hallucinations—calibrated confidence allows for rejection or alerting users. Calibration methods are categorized into two types: supervised (temperature scaling, calibration-tuning), which are difficult to scale due to dependence on human labels; and unsupervised (aggregated token probability, P(true), verbalized confidence, semantic entropy), which require no labels but extract signals solely from the PoLLM itself.

**Limitations of Prior Work**: Post-training (SFT / RLHF / DPO / RLVR) systematically pushes models toward overconfidence, often assigning probabilities as high as 0.9 to incorrect answers. Llama3.1-8B-Instruct shows a vanilla probability ECE as high as 0.5255 on SQuAD; SFT/DPO/Instruct checkpoints for Olmo2 consistently show that post-training damages calibration. All unsupervised methods relying on signals from the PoLLM itself are contaminated by this "overconfidence paint."

**Key Challenge**: For unsupervised calibration, it is necessary to find an **external reference signal that does not depend on the PoLLM's own probabilities**, while avoiding new training labels or model modifications to maintain the value of being unsupervised and "plug-and-play."

**Goal**: (i) Identify a naturally existing, cognate reference signal without requiring labels; (ii) design a low-cost method to map this signal to PoLLM responses without compromising generation quality.

**Key Insight**: The authors observe that since base LLMs are generally well-trained (with pretraining loss aligned to the true next-token distribution), they should be closer to the true probability distribution than fine-tuned PoLLMs. Calibration plots on TriviaQA, NQ, and the Qwen / Llama / Olmo families verify this: the reliability curves of base LLMs are close to the diagonal, while PoLLMs consistently lie below it (overconfident).

**Core Idea**: Use the cognate base LLM as an "honest reference," mapping PoLLM-generated answer scores to the probability space of the base LLM to restore calibration. A linear projection replaces the base LLM forward pass to amortize inference costs.

## Method

### Overall Architecture
Let $\mathcal{M}_p$ be the PoLLM and $\mathcal{M}_b$ be the cognate base LLM. $\mathcal{M}_p$ generates an answer $y^p=(y_1^p,\dots,y_T^p)$ for a prompt $x$ as usual. BaseCal does not modify the generation process of $\mathcal{M}_p$ but takes over the "confidence calculation" stage. Two routes are proposed: (1) **BaseCal-ReEval**: Feed $(x, y^p)$ into $\mathcal{M}_b$ for forced decoding and use the average token probability assigned by $\mathcal{M}_b$ to $y_t^p$ as confidence; (2) **BaseCal-Proj**: Train a linear mapping $\phi_\theta:\mathbb{R}^d\to\mathbb{R}^d$ to project the last-layer hidden states of $\mathcal{M}_p$ into the last-layer space of $\mathcal{M}_b$, then pass through the base output layer $W_b^o$ to obtain approximate base probability distributions, avoiding the full forward pass of the base model. Both schemes are plug-and-play, unsupervised (no ground-truth labels required), and do not modify model parameters.

### Key Designs

1.  **BaseCal-ReEval (Direct Route)**:
    - **Function**: Assigns probabilities to PoLLM answers using the base LLM as a confidence measure.
    - **Mechanism**: For a generated answer $y^p$, confidence is defined as $c_b(x,y^p)=\frac{1}{T}\sum_{t=1}^T P_{\mathcal{M}_b}(y_t^p\mid x,y_{<t}^p)$. This involves performing teacher-forcing of the PoLLM's output sequence on the base LLM and averaging the probability of each target token from the base model's perspective.
    - **Design Motivation**: The probability distribution of the base LLM is closer to the true distribution. Thus, the overall probability for an incorrect answer will be lower, and higher for a correct one, providing naturally "calibrated confidence." This is a simple, strong baseline, but it requires an additional full forward pass of the base model during inference.

2.  **BaseCal-Proj (Lightweight Projection Route)**:
    - **Function**: Uses a $d\times d$ linear layer to approximate the base LLM's output, eliminating the overhead of the base model's Transformer forward pass.
    - **Mechanism**: For each $(x, y^p)$ in a training set, the last-layer hidden states $(h^p_{t-1}, h^b_{t-1})$ from both $\mathcal{M}_p$ and $\mathcal{M}_b$ are extracted at each position. $\phi_\theta(h^p)=Wh^p+b$ is trained via MSE to fit $h^b$. During inference, only $\text{softmax}(W_b^o\,\phi_\theta(h^p_{t-1}))[y_t^p]$ is computed to obtain target token probabilities, which are then averaged. This is equivalent to "borrowing the base head but skipping its Transformer blocks."
    - **Design Motivation**: Although BaseCal-ReEval is effective, it increases latency. Hidden states contain richer information than probabilities and are orthogonally separable from the output layer. A single linear projection is sufficient to shift PoLLM states back to the base representation space—TSNE visualizations show that projected hidden states highly overlap with base states.

3.  **Training via "Question Sets" — Entirely Unsupervised**:
    - **Function**: BaseCal-Proj can be trained using only questions (without answer labels).
    - **Mechanism**: The training set consists of 10k **questions** sampled from TriviaQA / NQ / SQuAD / WebQ, paired with answers generated by the PoLLM. The supervision signal is the hidden state of the base LLM under the same input. Early stopping is triggered by the MSE on a 2k-question validation set. This process requires neither ground-truth answers nor correctness labels.
    - **Design Motivation**: Converting "calibration" from supervised post-hoc fitting (like temperature scaling, which requires correctness labels) into "representation space alignment" avoids overfitting to specific dataset accuracy distributions, resulting in nearly no performance drop during OOD evaluation (see RQ2).

### Loss & Training
The default $\phi_\theta$ is a single linear mapping, with loss $\mathcal{L}_{\text{MSE}}=\frac{1}{T}\sum_t \|\phi_\theta(h^p_{t-1})-h^b_{t-1}\|_2^2$. Comparisons with MAE / Cosine / 3-layer MLP show that MSE and MAE are similarly stable. Cosine loss leads to failure on TriviaQA (ECE 0.5+), indicating that angular alignment alone is insufficient. $\mathcal{M}_p$ and $\mathcal{M}_b$ are frozen throughout training; only $W,b$ are updated.

## Key Experimental Results

### Main Results
ECE↓ for five datasets $\times$ three PoLLMs (selected):

| Method | Unsup. | TriviaQA (Llama) | NQ (Llama) | SQuAD (Llama) | TriviaQA (Qwen) | MMLU (Qwen) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Temp. Scaling (supervised) | ✗ | 0.0226 | 0.0460 | 0.0911 | 0.0895 | 0.2261 |
| Vanilla (avg token prob) | ✓ | 0.1725 | 0.4532 | 0.5255 | 0.3406 | 0.2569 |
| P(true) | ✓ | 0.2476 | 0.4439 | 0.5532 | 0.2113 | 0.3204 |
| Verbalization | ✓ | 0.1769 | 0.2689 | 0.3603 | 0.2889 | 0.1972 |
| Semantic Entropy | ✓ | 0.2443 | 0.4927 | 0.4645 | 0.3583 | 0.2858 |
| DACA (multi-choice only) | ✓ | – | – | – | – | 0.0703 |
| **BaseCal-Proj** | ✓ | **0.0387** | 0.2488 | 0.3134 | 0.1393 | 0.0889 |
| **BaseCal-ReEval** | ✓ | **0.0309** | **0.2462** | **0.2959** | **0.1120** | **0.0393** |

BaseCal achieves the best results in 29 out of 30 settings. BaseCal-ReEval reduces ECE by an average of 42.9% relative to the strongest unsupervised baseline, while BaseCal-Proj reduces it by 35.3% with almost no additional inference overhead. On TriviaQA / MMLU, BaseCal parity matches or exceeds supervised Temperature Scaling.

### Ablation Study

| Dimension | Configuration | TriviaQA ECE | Remarks |
| :--- | :--- | :--- | :--- |
| Projection Arch (Llama) | 1-layer Linear | **0.0387** | Default |
| Projection Arch (Llama) | 3-layer MLP+ReLU | 0.1526 | Increased complexity is worse |
| Loss Function (Llama) | MSE | **0.0387** | Default |
| Loss Function (Llama) | MAE | 0.0447 | Similar to MSE |
| Loss Function (Llama) | Cosine | 0.6125 | Angle alignment fails |
| Model Scale (Qwen, TriviaQA) | 7B vanilla→Proj→ReEval | 0.3406 → 0.1393 → 0.1120 | Gains at all scales |
| Model Scale (Qwen, TriviaQA) | 14B | 0.2687 → 0.0778 → 0.0663 | |
| Model Scale (Qwen, TriviaQA) | 32B | 0.2662 → 0.0854 → 0.0542 | |
| Model Scale (Qwen, TriviaQA) | 72B | 0.2089 → 0.0502 → 0.0440 | Stronger base yields higher gains |
| Post-train Phase (Olmo2, TriviaQA) | SFT / DPO / Instruct | 0.0582 / 0.0269 / 0.0314 | Effective across all post-training |

### Key Findings
- **Base LLMs remain calibrated on free-form QA**: Figure 2 shows that the reliability bars of the Qwen / Llama / Olmo base models are close to the diagonal, while PoLLMs are consistently overconfident—this is the empirical foundation of the work.
- **Simple linear projection is sufficient**: A 3-layer MLP provides no gain or even degrades performance, verifying that "calibration information is not destroyed by post-training; it simply undergoes a representation space shift."
- **Strong cross-dataset generalization**: BaseCal-Proj shows $\Delta\text{ECE}\approx +0.0005$ when training and testing sets are swapped among SQuAD/NQ/TriviaQA/WebQ (nearly no drop), whereas Temperature Scaling shows $\Delta\text{ECE}\approx -0.0886$ (severe overfitting to training set accuracy).
- **Larger models benefit more**: On 72B models, BaseCal-Proj cuts ECE from 0.21 to 0.05, likely because larger base LLMs are better calibrated, providing a stronger alignment target.
- **Downstream gains**: Under selective classification (thresholds 0.5–0.95), BaseCal-Proj achieves higher accuracy than vanilla at all cutoffs, indicating its high-confidence samples are more reliable.
- **Failure modes**: Verbalization performed well on Olmo2-7B-NQ but collapsed to 0.4718 on Qwen2.5-7B, showing that methods relying on instruction-following are unstable; BaseCal remains top-2 in all 30 settings.

## Highlights & Insights
- **"Finding an honest cognate reference" is a new paradigm**: While previous unsupervised calibration attempted to extract information from the PoLLM alone, this work asks "who is the PoLLM's honest sibling," using the base LLM as an external reference. This logic can be extended to reward modeling, hallucination detection, and other trust-related tasks.
- **Linear alignment of hidden states implies post-training preserves representations**: The fact that a single linear mapping restores calibration and remains stable across datasets suggests that post-training induces relatively mild geometric transformations on internal representations. This aligns with RLHF/DPO often using KL constraints and provides evidence for designing "calibration heads" that survive post-training.
- **BaseCal-Proj reduces inference cost to near zero**: It requires only one $d\times d$ matrix multiplication and one base output layer softmax, making it significantly faster than semantic entropy or verbalization which require multiple samples or forward passes.
- **Consistent effectiveness across post-training strategies**: SFT, DPO, and RLVR are all improved by the same method, suggesting that overconfidence is a common side effect of post-training rather than an RL-specific issue.

## Limitations & Future Work
- Requires access to the base LLM's final hidden states and output head (not applicable to closed-source APIs like OpenAI/Anthropic); better suited for open-source and proprietary internal models.
- Evaluation is primarily on factual short-answer QA and MMLU; it remains to be verified whether base models are more calibrated in long-form generation, complex multi-step reasoning, or code.
- Explains "what" (base is more calibrated) but not "why"—whether collapse is due to the cross-entropy objective of pretraining or bias introduced by RLHF remains an open question.
- BaseCal-Proj requires 10k questions for training; the data requirement for specialized small-scale domains (medical/legal) needs further validation.
- Extensions: Integrating the base model as an "honest prior" during the RLHF training phase as a calibration regularizer, or extending to cross-modal base-to-PoLLM alignment.

## Related Work & Insights
- **vs DACA (Luo et al., 2025)**: DACA performs single-temperature rescaling at the probability level and only works when base and PoLLM top-1 tokens match, limiting it to multiple-choice questions. BaseCal performs alignment at the hidden state level, natively supporting free-form QA and outperforming DACA on MMLU (0.0393 vs 0.0703 on Qwen).
- **vs Temperature Scaling**: TS is a supervised post-hoc fit that depends on correctness labels and overfits to the training accuracy distribution; BaseCal is unsupervised and stable across datasets.
- **vs Semantic Entropy / P(true) / Verbalization**: These extract signals from the PoLLM itself and thus carry overconfidence bias; BaseCal avoids this by introducing an external honest reference.
- **vs Calibration-aware Fine-tuning (Xiao 2025, Wang 2025)**: These modify PoLLM parameters to incorporate calibration; BaseCal is entirely plug-and-play.
- **vs Hidden State Probing for Hallucination (Orgad 2025)**: Also uses last-layer hidden states, but Orgad et al. use supervised probes for hallucination detection, whereas BaseCal uses unsupervised projection for probability calibration.

## Rating
- Novelty: ⭐⭐⭐⭐ The "base as honest sibling" concept + hidden state linear projection is a simple yet powerful combination.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive ablation across 5 datasets, 3 model families, 4 model scales, and 3 post-training phases.
- Writing Quality: ⭐⭐⭐⭐ The motivation flows clearly from observation to derivation, supported by intuitive TSNE and reliability plots.
- Value: ⭐⭐⭐⭐ Provides a zero-intrusion, low-cost calibration solution for existing open-source PoLLMs with high engineering utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] CadLLM: Improving the Throughput of Diffusion-based LLMs via Training-Free Confidence-Aware Calibration](improving_the_throughput_of_diffusion-based_large_language_models_via_a_training.md)
- [\[NeurIPS 2025\] PPG-Distill: Efficient Photoplethysmography Signals Analysis via Foundation Model Distillation](../../NeurIPS2025/model_compression/ppg-distill_efficient_photoplethysmography_signals_analysis_via_foundation_model.md)
- [\[ACL 2026\] VecCISC: Improving Confidence-Informed Self-Consistency with Reasoning Trace Clustering and Candidate Answer Selection](veccisc_improving_confidence-informed_self-consistency_with_reasoning_trace_clus.md)
- [\[ICML 2026\] Multi-Adapter Representation Interventions via Energy Calibration](../../ICML2026/model_compression/multi-adapter_representation_interventions_via_energy_calibration.md)
- [\[ACL 2026\] SRA: Span Representation Alignment for Large Language Model Distillation](sra_span_representation_alignment_for_large_language_model_distillation.md)

</div>

<!-- RELATED:END -->
