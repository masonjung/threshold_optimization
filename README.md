<p align="center">
  <img src="https://github.com/user-attachments/assets/dfb57123-8d1f-435d-9164-96e17edd1d96" width="450">
</p>

<p align="center">
   <a href="https://arxiv.org/pdf/2502.04528"><b>https://arxiv.org/pdf/2502.04528</b></a>
</p>

<p align="center">
   <b>Robust Classification Across Multiple Subgroups.</b>
</p>

</p>
<p align="center">
  <a href="https://github.com/masonjung/threshold_optimization/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg"/></a>
  <a href="https://arxiv.org/pdf/2502.04528"><img src="https://img.shields.io/badge/arXiv-2502.04528-b31b1b.svg"/></a>
</p>
<p align="center">
  <a href="https://www.media.mit.edu/groups/multisensory-intelligence">
    <img src="https://img.shields.io/badge/MIT-Multisensory%20Intelligence%20Group-ff9900.svg"/>
  </a>
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


## Observed Distributional Variations
The probability distirbution is significantly different by characteristics of writing (e.g., length of the text, and writing style) and this leads notably different error rates across subgroups when the universal threshold is applied for the classification:


<p align="center">
  <img src="https://github.com/user-attachments/assets/b831e207-f323-459f-80a9-785424422e21" width="500">
</p>

Histograms show how AI-generated probability distributions differ by text length (Short in red, Medium in green, Long in blue). If we use the threshold of 0.5 (black dash line), human-written medium-length text shows higher error rate than other lengths. Each kernel density estimation (KDE) curve reflects the probability scores assigned by RoBERTa-large on the training dataset.


<p align="center">
  <img src="https://github.com/user-attachments/assets/08f327b1-3ec9-4cfe-8a52-de057c130776" width="500">
</p>
Top two highest and lowest distributional differences using Kolmogorov-Smirnoff (KS) test on RoBERTa-large detector probabilities. The visualization is based on KDE. The biggest discrepancy is observed with a KS statistic of 0.2078 (p < 0.01), while the smallest is 0.1001 (p < 0.01). These discrepancies indicate varying levels of divergence between groups based on the characteristics of the given text.



## FairOPT
FairOPT generates adaptive thresholds to each group and overcomes the limitation of the universal threshold. 

<p align="center">
  <img src="https://github.com/user-attachments/assets/116d705b-c4f1-4a7c-889d-1b890f193714" width="1000">
</p>

Applying adaptive thresholds to different probability distributions by subgroup, as seen through cumulative density functions (CDFs). We partition texts by three length categories (short, medium, long) and five personality traits (extroversion, neuroticism, agreeableness, conscientiousness, openness), and use RADAR to infer AI-generated probabilities on the test dataset. Observe that a static classification threshold (0.5, in black) and a single optimized threshold (in gray at the right end side) does not account for subgroup-specific distributional variations.

FairOPT works based on this algorithm:
<p align="center">
  <img src="https://github.com/user-attachments/assets/ab13369e-bfe4-4b9f-ba78-6d8ab25b094e" width="500">
</p>


Our approach addresses disparity in classification outcomes. This technology can play a crucial role in AI detection for identifying misinformation, safeguarding the integrity of publication and academic organizations, and countering potential cybersecurity threats. Since it is crucial for these AI-content detection methods to be robust and fair across many potential users, our method takes a major step in this direction by formulating the problem setting and developing a new algorithm. 

## Citation

Please cite us as:

```
@article{jung2025group,
  title={Group-Adaptive Threshold Optimization for Robust AI-Generated Text Detection},
  author={Jung, Minseok and
          Fuertes Panizo, Cynthia and
          Dugan, Liam and
          May, Yi R. and
          Chen, Pin-Yu and
          Liang, Paul},
  journal={arXiv preprint arXiv:2502.04528},
  year={2025}
}
```

## Acknowledgement
Thanks to Hengzhi Li, Megan Tjandrasuwita, David Dai, and Jeongmin Kwon, and Chanakya Ekbote for constructive feedback and discussion.




