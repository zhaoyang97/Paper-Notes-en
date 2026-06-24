---
title: >-
  [Paper Note] Can LLMs Understand Unvoiced Speech? Exploring EMG-to-Text Conversion with LLMs
description: >-
  [ACL 2025][LLM (Other)][EMG-to-Text] Academic paper note for Can LLMs Understand Unvoiced Speech? Exploring EMG-to-Text Conversion with LLMs.
tags:
  - ACL 2025
  - LLM (Other)
  - EMG-to-Text
date: 2026-05-08
content_hash: 710ea93015521aba
---
# Can LLMs Understand Unvoiced Speech? Exploring EMG-to-Text Conversion with LLMs

**Conference**: ACL 2025  
**arXiv**: [2506.00304](https://arxiv.org/abs/2506.00304)  
**Code**: [payalmohapatra/SilentSpeechLLM](https://github.com/payalmohapatra/SilentSpeechLLM)  
**Area**: LLM/NLP  
**Keywords**: EMG-to-Text, Silent Speech Interface, Biological Signal Understanding, Frozen LLM, Personalized Modeling

## Background & Motivation

This paper addresses a highly specific yet socially significant problem: whether it is possible to reconstruct intended speech into text relying solely on silent electromyography (EMG) signals.  
Traditional automatic speech recognition (ASR) assumes the presence of sound or at least acoustic representations. However, for individuals who are mute or unable to speak, this precondition does not hold.  
Silent surface electromyography (sEMG) captures the activation of articulatory muscles around the mouth and larynx, making it a crucial signal source for silent speech interfaces (SSIs).  
The issue lies in the fact that although many previous EMG-to-text methods claim to target "silent speech," they still rely on voiced EMG or synchronized audio as auxiliary supervision during training.  
This setting is impractical for users who are genuinely unable to speak, as they cannot provide voiced paired data.  
The authors establish this realistic constraint as the starting point: providing the model only with unvoiced EMG—without any audio or voiced EMG—to investigate whether LLMs can directly learn to decode it.  
Another core challenge is the extreme scarcity of data.  
Public single-subject closed-vocabulary datasets contain only about 26 minutes (500 samples) in total, and the practically usable training data can be as low as 6 minutes.  
If conventional data-driven approaches are applied, models are highly prone to overfitting or failing to learn stable mappings altogether.  
EMG also displays substantial user-specificity, with muscle activation patterns varying significantly across individuals; pilot experiments in this paper even demonstrate 96% accuracy in speaker identification.  
This implies that a unified large model might not directly generalize across users, highlighting the greater importance of lightweight personalized adaptation.  
Consequently, instead of training an end-to-end giant EMG model, the authors ask the reverse question: since LLMs have already internalized rich linguistic priors, is it possible to train only a small adapter that projects EMG signals into the LLM's embedding space, leaving the "linguistic aspect" to a frozen LLM?  
From the perspective of research motivation, this paper has three primary objectives:  
First, to verify whether LLMs can "understand" a linguistic modality they have barely encountered before, namely unvoiced biological signals.  
Second, to evaluate whether a frozen LLM combined with a small adapter is more data-efficient than application-specific models under extremely low-resource conditions.  
Third, to explore which adapter architectures, input features, and training objectives are best suited for EMG modalities characterized by high noise, small sample sizes, and strong individual variability.

## Method

The core of the proposed method is a trainable EMG adapter coupled with a completely frozen LLaMA model.  
Rather than directly representing EMG as discrete tokens, the authors encode it into a sequence of continuous embeddings, which are then concatenated with prompts and fed into the LLM for autoregressive text generation.  
In terms of system design, this is similar to many speech-to-LLM or vision-to-LLM frameworks. However, since EMG signals are scarcer, more individualized, and less interpretable, the adapter design cannot simply duplicate audio-based paradigms.

The input signal is denoted as $\mathbf{X}^E \in \mathbb{R}^{T \times C}$, where $C$ represents the number of EMG channels (8 channels in the primary dataset of this study).  
Due to a high raw sampling rate exceeding 800Hz, the sequence length is extremely long, rendering direct input to the LLM impractical.  
The authors first apply a 1D convolution with a stride of 6 for the initial temporal downsampling, reducing the sequence length to one-sixth.  
This is followed by stacking two residual convolutional blocks to extract local temporal patterns.  
The significance of the residual structure here is not to pursue extreme depth, but rather to stabilize training under small data regimes while preserving raw local electromyographic morphology.  
Following the residual blocks, a BiLSTM is introduced to model temporal dependencies.  
This represents a highly informative design choice: while many modern architectures employ Transformers by default, the authors empirically find that BiLSTM is significantly better suited for the current closed-vocabulary, short-sequence, and low-resource scenario.  
Subsequently, the model employs a second 1D convolution with a stride of 2 for further downsampling.  
Collectively, the total temporal compression ratio is approximately 48-fold.  
The compressed features are then projected via a linear layer to match the word embedding dimension of the LLM (4096 dimensions for LLaMA 2-7B, 3072 dimensions for LLaMA 3.2-3B).  
This step generates the sequence of EMG embeddings, which serve as "pseudo-input tokens" for the LLM.

To enable the frozen LLM to understand that the current input is not standard text, the authors design contextualized prompt concatenation.  
The text identifier "Unvoiced EMG:" is prefixed to the EMG embeddings, and the task description "Prompt: Convert unvoiced EMG embeddings to text" is appended.  
The prefix and suffix prompts are first processed by the tokenizer and word embedding layer into standard text embeddings before being concatenated with the intermediate EMG embeddings.  
This design can be conceptualized as constructing a task context for the model: the prefix declares the modality, while the suffix defines the task.  
Consequently, during inference, the LLM is not forced to make blind guesses from unusual continuous vectors, but instead performs the "transcription" task within a familiar prompt framework.

During training, only the adapter parameters are updated, while the LLM remains frozen.  
The loss function utilizes cross-entropy with temperature scaling where $\tau = 0.8$, optimized using AdamW with a learning rate of $5 \times 10^{-5}$.  
In the inference phase, autoregressive generation with a beam width of 4 is utilized.  
The authors also experiment with CTC, but find it performs worse than cross-entropy.  
This suggests that once EMG signals are projected into the LLM's embedding space, it is preferable to align with the LLM's inherent autoregressive training paradigm rather than reverting to traditional CTC-based speech recognition.

Another intriguing aspect of this work is the choice of input features.  
The authors evaluate not only raw EMG but also 112-dimensional hand-crafted features, including both time-domain and frequency-domain statistics.  
For traditional application-specific models, hand-crafted features yield worse performance; however, for the LLM adapter method, they yield significantly better results.  
This indicates that the bottleneck for a frozen LLM is not necessarily at the language layer, but rather in the front-end adapter's capacity to extract optimal representations from raw, high-noise EMG.  
In other words, when the trainable component contains only 6 million parameters, incorporating modest domain-specific feature engineering remains highly beneficial.

Beyond this, the authors conduct two types of exploratory investigations.  
First, comparing the difficulty of audio-to-LLM versus EMG-to-LLM reveals that even with a very simple audio interface, handling audio is easier for the LLM than handling EMG, indicating that the EMG modality is inherently more challenging rather than being a straightforward modality-swap.  
Second, incorporating voiced EMG data benefits the application-specific model significantly more than the LLM-based method.  
This suggests that current LLM adaptation schemes do not fully exploit voiced/unvoiced alignment signals, leaving room for future improvements through explicit cross-modal alignment or instruction tuning.

| Module | Specific Design | Function |
|------|----------|------|
| Temporal Downsampling 1 | 1D convolution with stride=6 | Compress high-sampling-rate raw EMG |
| Local Feature Extraction | 2 residual convolutional blocks | Extract stable local temporal patterns |
| Sequence Modeling | BiLSTM | Capture cross-temporal dependencies, superior to Transformer |
| Temporal Downsampling 2 | 1D convolution with stride=2 | Further reduce sequence length |
| Projection Layer | Fully Connected + GeLU | Align to LLM embedding dimension |
| Language Decoder | Frozen LLaMA | Generate text using pre-existing linguistic priors |

| Design Choice | Authors' Conclusion | Underlying Reason |
|----------|----------|-----------|
| Frozen LLM vs Direct Fine-tuning LLM | Frozen is more stable | Too little data, direct fine-tuning of the LLM easily leads to overfitting |
| BiLSTM vs Transformer | BiLSTM is better | Stronger local temporal inductive bias is needed in closed-vocabulary, short-sequence settings |
| Hand-crafted Features vs Raw EMG | Hand-crafted is better for the LLM method | Limited adapter capacity; requires noise-reduced input |
| CE vs CTC | CE is better | Better aligns with the training paradigm of decoder-only LLMs |

## Key Experimental Results

Experiments are primarily based on the single-speaker 8-channel closed-vocabulary dataset by Gaddy and Klein, which contains 67 words, approximately 26 minutes of unvoiced EMG data, and 500 samples.  
The evaluation metric is Word Error Rate (WER), reported under a 3-fold cross-validation scheme with an 8:1:1 training, validation, and testing split.  
Two types of baselines are compared: (1) the application-specific EMG-to-Text model by Gaddy and Klein with approximately 54 million parameters; and (2) the proposed EMG adapter with a frozen LLM, which trains only about 6 million parameters.

The main results are straightforward.  
Under raw EMG input, the best-performing LLM method is EMG-Ad + Llama3-3B, yielding a WER of 0.52, which is significantly better than the 0.75 achieved by the application-specific model.  
With hand-crafted feature inputs, both Llama2-7B and Llama3-3B achieve a WER of 0.49, whereas the application-specific model degrades to 0.84.  
This demonstrates that the core conclusion of this study is not merely that "LLMs perform slightly better," but that "under extremely low-data regimes, frozen LLMs can successfully translate linguistic priors into tangible EMG decoding gains."

| Model | Input Feature | Trainable Parameters | WER |
|------|----------|------------|-----|
| App-Specific Baseline | Raw EMG | 54M | 0.75 ± 0.06 |
| EMG-Ad + Llama2-7B | Raw EMG | 6M | 0.65 ± 0.01 |
| EMG-Ad + Llama3-3B | Raw EMG | 6M | **0.52 ± 0.05** |
| EMG-Ad + Fine-tuned Llama3-3B | Raw EMG | Higher | 0.62 ± 0.04 |
| App-Specific Baseline | Hand-crafted Features | 54M | 0.84 ± 0.06 |
| EMG-Ad + Llama2-7B | Hand-crafted Features | 6M | **0.49 ± 0.06** |
| EMG-Ad + Llama3-3B | Hand-crafted Features | 6M | **0.49 ± 0.04** |
| EMG-Ad + Fine-tuned Llama3-3B | Hand-crafted Features | Higher | 0.55 ± 0.02 |

In terms of relative improvement, the best result drops from 0.75 to 0.49, marking an absolute reduction of 0.26, which represents a substantial leap in performance.  
More notably, this gain is achieved under the constraint of only a few minutes of training data.  
The authors further perform data scaling-down experiments, progressively reducing the training set from 26 minutes to 6 minutes.  
Although the WER predictably increases with less data, the LLM-based method consistently outperforms the application-specific baseline by approximately 26% on average across all data sizes.  
This is critical for real-world scenarios, where users typically cannot provide large quantities of long-duration annotated data.

The ablation studies are also highly insightful.  
The authors compare several adapter variants, including fully connected layers only, residual blocks (ResBlocks) only, ResBlocks with a Transformer, and ResBlocks with an LSTM.  
Results show that ResBlock(2) + LSTM performs best with Llama3-3B, yielding a WER of 0.53, whereas incorporating a Transformer degrades performance to 0.79.  
Similarly, changing the training objective from CE to CTC on Llama2-7B causes the WER to decline from 0.65 to 0.70.  
These results collectively demonstrate that short, weak, and highly noisy signals like EMG are not inherently suited for highly "Transformerized" front-ends.

| Ablation Setting | Variant | WER |
|----------|------|-----|
| Adapter Architecture | Fully Connected | 0.70 |
| Adapter Architecture | ResBlock(2) | 0.64 |
| Adapter Architecture | ResBlock(2) + Transformer | 0.79 |
| Adapter Architecture | ResBlock(2) + LSTM | **0.53** |
| Training Objective | CE + Llama2-7B | **0.65** |
| Training Objective | CTC + Llama2-7B | 0.70 |

The authors also perform three complementary experiments to define task boundaries.  
First, a person identification experiment achieves a 96% classification accuracy using unvoiced EMG, highlighting the strong presence of individual characteristics in the signals, which in turn explains why personalized modeling is unavoidable.  
Second, data augmentation techniques such as temporal shift and Hilbert phase alignment show virtually no benefits, indicating that EMG is highly sensitive to temporal alignment and cannot easily benefit from generic augmentation tricks.  
Third, comparing audio-to-LLM and EMG-to-LLM confirms that the audio task is much easier, underscoring that substantial representation learning challenges remain unresolved when integrating the EMG modality into LLMs.

Regarding overall experimental quality, instead of pursuing an all-encompassing large benchmark, the paper conducts solid, rigorous comparative analyses of data efficiency, front-end architecture, feature design, loss functions, and task boundaries in a highly specific setting.  
This is far more informative than simply reporting a single SOTA number.

## Highlights & Insights

The primary highlight of this paper is its precise and practical task formulation: it relies neither on voiced EMG nor on audio, using only unvoiced EMG for transcription, which brings the settings closer to realistic assistive communication scenarios.  
Second, the combination of a frozen LLM and a tiny adapter under ultra-low resource budgets is proven feasible. This demonstrates that linguistic priors can cross-modally transfer to biological signal tasks, provided the alignment method is designed appropriately.  
Third, the hand-crafted features show utility for the LLM-based method but fail to benefit the application-specific model. This counter-intuitive finding is highly valuable, as it reveals a strong coupling between front-end feature engineering and downstream model capacity.  
Fourth, the authors refrain from overhyping the LLM, clearly noting through audio and voiced EMG comparisons that decoding EMG remains significantly more challenging, which increases the credibility of their findings.  
The key takeaway is that when integrating new modalities into LLMs under low-resource constraints, having a frozen large model does not mean the design of the front-end can be arbitrary. On the contrary, the adapter design and input feature representation act as critical bottlenecks for effectively leveraging the LLM's priors.

## Limitations & Future Work

First, the task remains limited to a closed-vocabulary setup of only 67 words, which is still far from open-vocabulary, natural-sentence-level input.  
Second, the main experiments evaluate a single speaker, leaving generalization across multiple users, devices, and languages unproven.  
Third, the current approach requires access to the LLM embedding layer, making it incompatible with closed-source commercial APIs.  
Fourth, the attempt at simple data augmentation yields limited gains, indicating that the challenge of data scarcity in EMG remains largely unresolved.  
Future directions can expand into three areas:  
First, scaling the closed vocabulary into a larger "controlled open vocabulary" and gradually transitioning to decoding natural sentences.  
Second, exploring rapid cross-subject adaptation techniques, such as LoRA, meta-learning, or prototype-based personalization.  
Third, unifying multiple biological signals such as EMG, EEG, and EOG into a single multimodal LLM to investigate which linguistic priors can be shared across signals and which must be modeled individually.

## Related Work & Insights

Compared to early silent speech interface works, the key distinction of this study is that it no longer relies on voiced signals as an indispensable intermediate.  
Compared to Gaddy and Klein's application-specific EMG-to-Text model, instead of stacking architectures inside the EMG model, this study outsources "language modeling" to a frozen LLM, utilizing the front-end solely for spatial mapping.  
Compared to methods such as Benster et al. that treat LLMs as post-processing error correctors, this work involves the LLM directly in modality understanding rather than just downstream sentence correction.  
Compared to speech- or video-to-LLM adapter works, this study indicates that biological signal modalities present higher difficulties and cannot just directly import mature pipelines.  
In terms of methodological transfer, this work is highly suggestive for brain-computer interfaces (BCIs), wearable sensor transcription, and assistive communication for neuromuscular disorders.  
Future works targeting EEG-to-Text or gesture-to-language can safely leverage the pipeline of "lightweight adapter + frozen LLM + highly constrained task prompt."  
Meanwhile, it is critical to keep the negative findings of this study in mind: when signals are extremely scarce and highly individualized, the capacity of the adapter, feature engineering, and task design play more vital roles than simply scaling up language model parameters.

## Rating

- Novelty: ⭐⭐⭐⭐☆ Directing the LLM to understand unvoiced EMG rather than performing post-processing is highly novel in both task formulation and methodology.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Main results, data reduction, architectural ablations, loss ablations, user identification, and augmentation experiments are reasonably complete, though the dataset scale remains small.
- Writing Quality: ⭐⭐⭐⭐☆ The paper is clearly presented with a coherent logical flow, particularly clarifying why a frozen LLM remains beneficial.
- Value: ⭐⭐⭐⭐⭐ High real-world value for assistive communication and low-resource multimodal LLM adaptation.
- Overall Evaluation: 8.8/10. A robust first step demonstrating that LLMs can indeed decode unvoiced EMG, though numerous systematic hurdles remain for open-world deployment.

---
title: >-
  [Paper Reading] Can LLMs Understand Unvoiced Speech? Exploring EMG-to-Text Conversion with LLMs
description: >-
  [ACL 2025][LLM/NLP][EMG-to-Text] This paper proposes a method based on a trainable EMG adapter network that maps unvoiced electromyography (EMG) signals into the input embedding space of Large Language Models (LLMs), achieving a Word Error Rate (WER) of 0.49 on a closed-vocabulary unvoiced EMG-to-text task, improving over application-specific models by approximately 20% with only 6 minutes of training data.
tags:
  - ACL 2025
  - LLM/NLP
  - EMG-to-Text
  - Silent Speech Interface
  - Multimodal LLM
  - Adapter Network
  - Biological Signals
---

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Can LLMs Interpret and Leverage Structured Linguistic Representations? A Case Study with AMRs](can_llms_interpret_and_leverage_structured_linguistic_representations_a_case_stu.md)
- [\[ACL 2025\] Zero-Shot Belief: A Hard Problem for LLMs](zero-shot_belief_a_hard_problem_for_llms.md)
- [\[ACL 2025\] Unintended Harms of Value-Aligned LLMs: Psychological and Empirical Insights](unintended_harms_of_value-aligned_llms_psychological_and_empirical_insights.md)
- [\[ACL 2025\] HiCUPID: Exploring the Potential of LLMs as Personalized Assistants](exploring_the_potential_of_llms_as.md)
- [\[ACL 2025\] LLMs Can Be Easily Confused by Instructional Distractions](llms_can_be_easily_confused_by_instructional_distractions.md)

</div>

<!-- RELATED:END -->
