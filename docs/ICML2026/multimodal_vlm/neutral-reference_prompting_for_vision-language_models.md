---
title: >-
  [Paper Note] Neutral-Reference Prompting for Vision-Language Models
description: >-
  [ICML 2026][Multimodal VLM][Base-Novel Trade-off] This paper re-attributes the Base-New Trade-off (BNT) in efficient VLM transfer to the "uneliminated asymmetric class preferences from pre-training on unseen classes." It…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "Base-Novel Trade-off"
  - "Asymmetric Confusion"
  - "Neutral-Reference Prompt"
  - "Bayesian Prior"
  - "Plug-and-play Correction"
date: 2026-05-08
content_hash: eed5ebb2c9132303
---

# Neutral-Reference Prompting for Vision-Language Models

**Conference**: ICML 2026  
**arXiv**: [2605.15615](https://arxiv.org/abs/2605.15615)  
**Code**: https://github.com/Sheldon04/NeRP (Available)  
**Area**: Multimodal VLM / Prompt Tuning / Efficient Transfer  
**Keywords**: Base-Novel Trade-off, Asymmetric Confusion, Neutral-Reference Prompt, Bayesian Prior, Plug-and-play Correction

## TL;DR
This paper re-attributes the Base-New Trade-off (BNT) in efficient VLM transfer to the "uneliminated asymmetric class preferences from pre-training on unseen classes." It proposes NeRP: using a semantically neutral text prompt and the "mean of training images" as reference inputs to estimate zero-parameter per-class prior shifts on a trained VLM. A Bayesian-style proxy score is then used to perform local flipping between confusable class pairs, improving novel class accuracy without modifying model parameters while preserving base class performance.

## Background & Motivation

**Background**: Efficient VLM transfer in the CLIP era (CoOp, CoCoOp, MaPLe, PromptSRC, TCP, MMA, etc.) almost exclusively relies on "learning a set of prompts/adapters on base classes" for downstream adaptation. While base class accuracy improves, novel (zero-shot unseen) class accuracy often declines, constituting the Base-New Trade-off.

**Limitations of Prior Work**: Mainstream explanations attribute BNT to "overfitting on base classes," leading various methods to focus on "anti-overfitting"—adding regularization, constraining prompt drift, introducing external knowledge, or sharing representations. However, the authors point out this is only half the story: poor novel class accuracy also stems from an independent, more subtle source—**asymmetrical confusion**. This is characterized by samples of class A being systematically misjudged as class B, while B is rarely misjudged as A, which differs fundamentally from conventional "symmetrical difficulty" confusion.

**Key Challenge**: Asymmetrical confusion originates from imbalances in pre-training data, forming preferences for certain classes in both image and text modalities. During fine-tuning, cross-entropy on base classes can suppress these inherited preferences (as ground-truth labels correct the decision boundaries), but predictions for novel classes rely entirely on zero-shot geometry, leaving pre-training preferences intact.

**Goal**: (1) Verify that such asymmetrical confusion indeed exists and is distinct from overfitting; (2) Identify and correct the shift direction of each novel class without modifying model parameters or re-training; (3) Avoid damaging correctly predicted samples.

**Key Insight**: The authors ask—"If a semantically empty image is fed into a VLM, which class would it choose?"—The answer reveals implicit class preferences. Using the "class scores corresponding to meaningless inputs" as a prior allows for measuring the intensity and direction of shifts between class pairs.

**Core Idea**: Construct "neutral-reference prompts" (class-agnostic text such as "a photo of an object" and the pixel mean of training images as neutral inputs). The per-class scores obtained from the VLM are treated as class priors. A Bayesian-style $\text{posterior}=\text{evidence}+\text{prior}$ format is used for post-hoc correction, triggering local flips only on samples where "priors are strong but evidence is weak" to avoid disrupting correct predictions.

## Method

### Overall Architecture
NeRP is a plug-and-play post-hoc correction module that does not modify any VLM parameters. Pipeline: (1) Given a downstream domain $D$, construct a text neutral anchor $u_{\mathrm{txt}}^0(D)=\text{norm}(g_{\mathrm{txt}}^0(\tau(D)))$ and an image neutral anchor $u_{\mathrm{img}}(D)=f_{\mathrm{img}}(\bar{x}^D)$ ($\bar{x}^D$ is the preprocessed pixel mean of training images); (2) Calculate per-class prior logits $\pi_{\mathrm{txt}}(c;D)$, $\pi_{\mathrm{img}}(c;D)$ against (fine-tuned) class prototypes $t(c)$ or zero-shot prototypes $t^0(c)$, and construct class-pair prior differences $\Sigma_{i,j}(D)$; (3) For a test image $x$ and the top-1 class $i$, compute a Bayesian proxy score $s_{ij}(x)=m_{ij}(x)+\Sigma_{i,j}(D)+\hat{\beta}(D)$ over confusable neighbors $j\in\mathcal{A}(i)$; (4) If the prior is strong ($\Sigma_{i,j}\ge\tau-\hat{\beta}$) and evidence is weak ($m_{ij}(x)\le\delta$), flip $i$ to $j$; otherwise, retain the original prediction.

### Key Designs

1. **Neutral-Reference Prompting and Category Prior Estimation**:

    - **Function**: Measure the intensity of implicit preferences for each class in the VLM from "semantically empty inputs" without modifying the model, using these as prior logits.
    - **Mechanism**: On the text side, a class-agnostic prompt $\tau(D)$ (e.g., "a photo of an object.") is passed through the zero-shot text encoder to get a neutral vector $u_{\mathrm{txt}}^0(D)$, which is then dot-producted with each (fine-tuned) prototype $t(c)$ to obtain $\pi_{\mathrm{txt}}(c;D)=\langle t(c),u_{\mathrm{txt}}^0(D)\rangle$. On the image side, the training set pixel mean $\bar{x}^D$ is passed through the image encoder to obtain $u_{\mathrm{img}}(D)$, dot-producted with the zero-shot prototypes to get $\pi_{\mathrm{img}}(c;D)$. The resulting class-pair differences $\Delta\pi_{\mathrm{txt}}(i,j)$ and $\Delta\pi_{\mathrm{img}}(i,j)$ serve as scales for measuring the same pre-trained inter-class direction $\Delta_{ij}^0=t^0(i)-t^0(j)$.
    - **Design Motivation**: The authors use a low-rank deformation model to express fine-tuning as $g=g^0+Ub$, proving that fine-tuning primarily reshapes the low-dimensional subspace $S$ spanned by base prototypes, while zero-shot geometry between novel classes remains mostly stable (Assumption 3.1 + 3.2). Thus, for novel class pairs $t(i)-t(j)\approx \Delta_{ij}^0$, and the prior difference $\Delta\pi$ shares the same sign as the expected logit difference $\mu_{ij}(D)$ (Prop. 3.5). On base classes, since $\Delta_{ij}^0\in S$ and the anchor energy on $S$ is small, the prior naturally diminishes (Lemma 3.4), avoiding interference with already learned base decisions.

2. **Residual Prior + Global Intercept for Semantically Diverse Data**:

    - **Function**: On datasets like ImageNet with extreme inter-class semantic variance, the original prior variance across class pairs is too large. The authors introduce a residual form and use base pairs to learn a global bias $\hat{\beta}(D)$ to absorb common drift.
    - **Mechanism**: Define the text residual prior $\tilde{\pi}_{\mathrm{txt}}(c;D)=\langle t(c),u_{\mathrm{txt}}^0(D)\rangle-\langle t(c),u_{\mathrm{txt}}(D)\rangle$ (subtracting the zero-shot neutral anchor from the fine-tuned neutral anchor, leaving the anchor's displacement). The image side $\tilde{\pi}_{\mathrm{img}}$ is defined similarly. The class-pair residual prior $\Delta\tilde{\pi}\approx\langle\Delta_{ij}^0,u_{\mathrm{txt}}^0-u_{\mathrm{txt}}\rangle$ directly measures the projection of the "pre-trained inter-class direction" onto the anchor displacement. The intercept $\hat{\beta}(D)=\arg\min_\beta\sum_{\mathcal{B}\times\mathcal{B}}(\hat{\mu}_{ij}-\Sigma_{i,j}-\beta)^2$ is fitted via least squares on base pairs and merged into the threshold $\tau$ during inference.
    - **Design Motivation**: Direct priors have a common, class-independent bias on datasets with sharp inter-class distributions. Residualization eliminates the common component of the two anchors projected onto each class, leaving only the class-related displacement component. This significantly reduces cross-pair variance and stabilizes Bayesian correction while maintaining the sign-consistency of Prop. 3.5, usually with a tighter constant term.

3. **Bayesian-style Proxy Score + Local Flipping Gating**:

    - **Function**: Integrate the prior into decision-making but only flip predictions in "prior-dominated" regions to avoid mis-flipping originally correct samples.
    - **Mechanism**: For sample $x$, top-1 class $i$, and neighbor $j\in\mathcal{A}(i)$, define the proxy score $s_{ij}(x)\approx m_{ij}(x)+\Sigma_{i,j}(D)+\hat{\beta}(D)$, where $m_{ij}(x)=\ell_i(x)-\ell_j(x)$ is the observed logit difference (interpretable as a log-approximation of the vMF likelihood ratio under L2 normalization). Under a vMF model, this is equivalent to log-posterior odds. A "prior-dominated region" $\mathcal{R}_{i\to j}=\{\Sigma_{i,j}(D)\ge\tau-\hat{\beta}(D)\wedge m_{ij}(x)\le\delta\}$ is defined: where the prior is strong (gate $\tau$) but sample evidence is weak (gate $\delta$). The neighbor graph $G$ is symmetric, ensuring comparisons only between confusable pairs. Once triggered, the prediction flips from $i$ to $j$.
    - **Design Motivation**: Simply adding a prior would contaminate samples with strong evidence. The authors use Corollary 3.6 to provide a high-probability bound for sample-level sign-consistency $\Pr[\text{sign mismatch}]\le \sigma_m^2/(|\mu|-\gamma)^2$—the prior should only be trusted when $|\mu|$ is significantly larger than the noise $\sigma_m$. The gating $(\tau, \delta)$ implements this theoretical guarantee into two thresholds. The neighbor graph $\mathcal{A}(i)$ constrains flips to "semantically close and confusable" pairs, further reducing the probability of erroneous flips.

### Loss & Training
NeRP is **entirely training-free**: all quantities are calculated once on domain $D$ using off-the-shelf pre-trained and fine-tuned VLMs (including class prototypes, neutral anchors, $\hat{\beta}(D)$, and the neighbor graph). Inference involves only two additional anchor encodings, a few dot products, and top-1 neighbor enumeration, making the overhead negligible.

## Key Experimental Results

### Main Results
On 11 standard base-to-novel downstream datasets (ImageNet, Caltech101, OxfordPets, StanfordCars, Flowers, Food101, Aircraft, SUN397, DTD, EuroSAT, UCF101), NeRP is used in conjunction with 5 mainstream baselines (CoOp, CoCoOp, MaPLe, PromptSRC, others). Average Base/Novel/HM (Harmonic Mean) values are reported.

| Method | Average Base | Average Novel | Average HM | Notes |
|------|--------------|---------------|------------|------|
| CoOp (IJCV 22) | 82.69 | 63.22 | 71.66 | Single prompt baseline |
| MaPLe (CVPR 23) | 82.28 | 75.14 | 78.55 | Multimodal deep prompt |
| MaPLe + NeRP (Selected Trend) | Stable | Significant ↑ | ↑ | Base maintained, Novel gains |

In the full paper, after adding NeRP to each baseline, Novel and HM scores improved on almost all datasets, while Base remained stable (theoretical guarantee from Lemma 3.4: prior differences are suppressed on base classes, making flip probability extremely low).

### Ablation Study
| Configuration | Behavior | Conclusion |
|------|------|------|
| Text-only $\pi_{\mathrm{txt}}$ | Using only text-side prior | Significant novel gain already achieved |
| Image-only $\pi_{\mathrm{img}}$ | Using only image-side prior | Complementary to $\pi_{\mathrm{txt}}$ |
| $\pi_{\mathrm{txt}}+\pi_{\mathrm{img}}$ (Default $\Sigma$) | Combined priors | Better than either side alone |
| Residual version $\tilde{\Sigma}$ | Subtracting current anchor projection | More stable on diverse datasets like ImageNet |
| Removing evidence gate $\delta$ | Flipping based solely on prior | Base accuracy damaged, HM decreases |
| Removing neighbor graph $\mathcal{A}(i)$ | Flipping considered for all $C-1$ classes | Erroneous flip rate increases |

### Key Findings
- **Asymmetrical confusion is an independent cause of BNT**: t-SNE and per-class mean logit variance plots (Paper Fig.3) show that novel class asymmetrical shifts and base class overfitting are independent degradation paths; overfitting regularization (e.g., PromptSRC) cannot resolve the former.
- **The "training mean image" as a neutral anchor is unexpectedly effective**: It preserves domain style while erasing semantics, exposing the "VLM's image-side preference for that domain," which serves as an orthogonal complement to the text-side prior.
- **Gating thresholds $(\tau, \delta)$ are nearly harmless to base classes**: Lemma 3.4 states that prior differences are inherently small on base classes, so flips are rarely triggered under default thresholds—this is the fundamental reason NeRP "maintains base while boosting novel."

## Highlights & Insights
- **Re-attribution of BNT**: Moving the narrative from "overfitting" to "pre-training asymmetric preference + lack of correction on novel classes," and providing measurable, correctable quantities, demonstrates high research insight.
- **Portable "Neutral Input" prior probe design**: Can be applied to any vision-language retrieval/classification system. By using a class-agnostic template on the text side and training set means/noise on the image side, one can immediately obtain class priors for zero-cost deployment.
- **Synergy between Theory and Engineering**: The low-rank deformation model, base subspaces, and vMF log-likelihood ratios form a complete chain. Gating thresholds are not arbitrary but are engineering realizations of the high-probability bounds in Corollary 3.6.

## Limitations & Future Work
- Primarily validated on base-to-novel splits; scalability for true open-vocabulary or long-tail settings (e.g., thousands of novel classes) regarding neighbor graph construction and threshold selection needs further verification.
- Neutral anchor construction is sensitive to domain distribution: training image means provide strong signals on style-consistent datasets (EuroSAT, DTD) but may dilute preference signals on highly heterogeneous datasets; domain clustering for multi-anchor approaches could be considered.
- Thresholds $(\tau, \delta)$ and the neighbor graph $\mathcal{A}(i)$ still require selection on a validation set, slightly compromising the "zero-parameter" ideal; adaptive thresholds could be explored.
- Currently uses cosine + vMF approximations; adaptability after logit calibration (temperature scaling) or for non-CLIP style VLMs (e.g., BLIP, Flamingo) needs evaluation.

## Related Work & Insights
- **vs CoOp / CoCoOp / MaPLe / PromptSRC / TCP / MMA**: These works combat overfitting via "prompt/adapter design during the training phase." NeRP is completely orthogonal, performing post-hoc correction during inference, and can thus be stacked on top of any of them.
- **vs ProGrad / DPC (Gradient direction / Decoupled dual prompts)**: These also focus on "not washing away zero-shot knowledge," but ProGrad/DPC focus on training-side direction control, whereas NeRP focuses on inference-side direction detection and correction.
- **vs CLIP Bias Studies (So-B-IT, M4, CounterAnimal, Ghate et al.)**: This line of work mainly audits and debiases training data/representations. NeRP transforms these "known biases" into usable priors as correction tools, shifting from "diagnosis" to "utilization."

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The combination of asymmetrical confusion + neutral anchor probes + Bayesian gating is unique in BNT literature.
- Experimental Thoroughness: ⭐⭐⭐⭐ 11 datasets × 5 baselines, though primarily on base-to-novel splits; cross-domain evaluation is relatively brief.
- Writing Quality: ⭐⭐⭐⭐ Clear theoretical chain (Assumptions 3.1-3.2 → Lemma 3.3-3.4 → Prop. 3.5 → Cor. 3.6) with well-defined engineering mappings.
- Value: ⭐⭐⭐⭐⭐ Zero training parameters, low inference overhead, and compatibility with any prompt tuning method make it highly valuable for industrial deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] VP-Bench: A Comprehensive Benchmark for Visual Prompting in Multimodal Large Language Models](../../AAAI2026/multimodal_vlm/vp-bench_a_comprehensive_benchmark_for_visual_prompting_in_m.md)
- [\[AAAI 2026\] Graph-of-Mark: Promote Spatial Reasoning in Multimodal Language Models with Graph-Based Visual Prompting](../../AAAI2026/multimodal_vlm/graph-of-mark_promote_spatial_reasoning_in_multimodal_langua.md)
- [\[ICML 2026\] Vision Language Models Cannot Reason Physical Transformations](vision_language_models_cannot_reason_about_physical_transformation.md)
- [\[ICML 2026\] Uncovering Visual Counting Bottlenecks in Vision-Language Models](unveiling_the_visual_counting_bottleneck_in_vision-language_models.md)
- [\[ICML 2026\] Large Vision-Language Models Get Lost in Attention](large_vision-language_models_get_lost_in_attention.md)

</div>

<!-- RELATED:END -->
