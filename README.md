<p align="center">
  <img src="https://github.com/user-attachments/assets/dfb57123-8d1f-435d-9164-96e17edd1d96" width="450">
</p>

<p align="center">
   <a href="https://arxiv.org/pdf/2502.04528"><b>https://arxiv.org/pdf/2502.04528</b></a>
</p>

<p align="center">
   <b>Robust Classification Across Multiple Subgroups.</b>
</p>

**Are you still using threshold of 0.5 for classification?** Why do you use 0.5 for your decision threshold, not 0.7, 0.25, or 0.95? Why did you choose to use it? We found that AI text classifier's probability distribution is different by characteristics of the given text and the performance can be enhanced by employing adaptive thresholds. In short, it is better to give different thresholds to different groups. Experiment was conducted using four AI text classifiers on three comprehensive datasets that includes multiple LLMs, their variates, and topics.

**FairOPT** helps you to:
* **1. Figure out optimal thresholds to each group**: Suppose you have five subgroups, you can generate five thresholds to apply to each rather than using one unified thresholds to all.
* **2. Increase classification robustness**: There are significant error rates discrepancies between groups. FairOPT reduces the discrepancy.
* **3. Provide rationales for the decision threshold selection**: You can provide reason for the threshold selection using this model.

The experiemtn was conducted based on the AI text classifiers but our approach can be extended to all kinds of probability based classifiers (e.g. deepfake audio classifier, AI-generated image classifier, etc.).

## Overview
<p align="center">
  <img src="https://github.com/user-attachments/assets/0133b828-7251-44ae-abed-1ed1a4312191" width="750">
</p>

We find that using one universal probability threshold (often θ > 0.5) to classify AI and human-generated text can fail to account for subgroup-specific distributional variations, such as text length or writing style, leading to higher errors for certain demographics. We propose FairOPT to learn multiple decision thresholds tailored to each subgroup, which leads to more robust AI-text detection.

## Updates
- **[Feb 19, 2025]** The results and codes are now public. Check out the [Paper](https://arxiv.org/pdf/2502.04528) on arXiv.


## Distributional Variations




<p align="center">
  <img src="https://github.com/user-attachments/assets/b831e207-f323-459f-80a9-785424422e21" width="500">
</p>

Histograms show how AI-generated probability distributions differ by text length (Short in red, Medium in green, Long in blue). If we use the threshold of 0.5 (black dash line), human-written medium-length text shows higher error rate than other lengths. Each kernel density estimation (KDE) curve reflects the probability scores assigned by RoBERTa-large on the training dataset.


<p align="center">
  <img src="https://github.com/user-attachments/assets/08f327b1-3ec9-4cfe-8a52-de057c130776" width="500">
</p>
Top two highest and lowest distributional differences using Kolmogorov-Smirnoff (KS) test on RoBERTa-large detector probabilities. The visualization is based on KDE. The biggest discrepancy is observed with a KS statistic of 0.2078 (p < 0.01), while the smallest is 0.1001 (p < 0.01). These discrepancies indicate varying levels of divergence between groups based on the characteristics of the given text.



## FairOPT






