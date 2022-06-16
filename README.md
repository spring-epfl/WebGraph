# WebGraph

Artifact release for the paper "WebGraph: Capturing Advertising and Tracking Information Flows for Robust Blocking", published at USENIX Security 2022.

### Pipeline & Requirements

### Code Organization

### Paper

**WebGraph: Capturing Advertising and Tracking Information Flows for Robust Blocking**
Sandra Siby, Umar Iqbal, Steven Englehardt, Zubair Shafiq, Carmela Troncoso
_USENIX Security Symposium (USENIX), 2022_

**Abstract** -- Users rely on ad and tracker blocking tools to protect their privacy. Unfortunately, existing ad and tracker blocking tools are susceptible to mutable advertising and tracking content.  In this paper, we first demonstrate that a state-of-the-art ad and tracker blocker, AdGraph, is susceptible to such adversarial evasion techniques that are currently deployed on the web. Second, we introduce WebGraph, the first ML-based ad and tracker blocker that detects ads and trackers based on their action rather than their content. By featurizing the actions that  are fundamental to advertising and tracking information flows – e.g., storing an identifier in the browser or sharing an identifier with another tracker – WebGraph  performs nearly as well as prior approaches, but is significantly more robust to adversarial evasions. In particular, we show that WebGraph  achieves comparable accuracy to AdGraph, while significantly decreasing the success rate of an adversary from near-perfect for AdGraph to around 8% for WebGraph. Finally, we show that WebGraph remains robust to sophisticated adversaries that use adversarial evasion techniques beyond those currently deployed on the web.

The full paper can be found [here](https://www.usenix.org/system/files/sec22summer_siby.pdf).



### Citation

If you use the code/data in your research, please cite our work as follows:

```
@inproceedings{Siby22WebGraph,
  title     = {WebGraph: Capturing Advertising and Tracking Information Flows for Robust Blocking},
  author    = {Sandra Siby, Umar Iqbal, Steven Englehardt, Zubair Shafiq, Carmela Troncoso},
  booktitle = {USENIX Security Symposium (USENIX)},
  year      = {2022}
}
```

### Contact

In case of questions, please get in touch with [Sandra Siby](https://sandrasiby.github.io/). 

