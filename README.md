# Generative Explainability for Next-Generation Networks: LLM-Augmented XAI with Mutual Feature Interactions

Companion repository for the paper presented at the 1st IEEE International Workshop on Generative and eXplainable Artificial Intelligence for Networking (GenXNet), held in conjunction with IEEE WiMob 2025 in Marrakech, Morocco.

**Authors:** Kiarash Rezaei, Omran Ayoub, Sebastian Troia, Francesco Lelli, Paolo Monti, and Carlos Natalino

[IEEE Xplore](https://doi.org/10.1109/WiMob66857.2025.11257542) · [Full text](https://research.chalmers.se/publication/548704/file/548704_Fulltext.pdf) · [Workshop program](https://traffic-arclab.github.io/genxnet2025/#program)

## Overview

The framework uses a large language model to turn SHAP explanations of optical lightpath quality-of-transmission (QoT) predictions into natural-language explanations. The baseline prompt uses individual SHAP feature contributions. The proposed prompt adds pairwise SHAP interaction values to help explain how features jointly affect a prediction.

The paper evaluates both prompts using a 4-bit quantized DeepSeek-R1 model on 40 local SHAP explanations from an XGBoost regressor. Two domain experts independently rated the generated explanations.

| Metric | Baseline prompt | Proposed prompt |
| --- | ---: | ---: |
| Correctness | 96.3% | 97.5% |
| Scope | 88.8% | 95.0% |
| Usefulness (0–5) | 3.77 | 4.38 |

Values are the averages reported in Table I. Scope and usefulness were scored for explanations rated correct.

## Experiment materials

This repository currently contains only this README. Table A-I in the paper describes the proposed prompt structure, and Table A-II gives examples of both prompts and their outputs. The source lightpath dataset is available from the [QoT Dataset Collection](https://www.hhi.fraunhofer.de/en/pn-software/qot-dataset-collection.html).

## Citation

```bibtex
@inproceedings{rezaei2025generative,
  author    = {Rezaei, Kiarash and Ayoub, Omran and Troia, Sebastian and Lelli, Francesco and Monti, Paolo and Natalino, Carlos},
  title     = {Generative Explainability for Next-Generation Networks: {LLM}-Augmented {XAI} with Mutual Feature Interactions},
  booktitle = {2025 International Conference on Wireless and Mobile Computing, Networking and Communications (WiMob)},
  year      = {2025},
  pages     = {1--6},
  publisher = {IEEE},
  doi       = {10.1109/WiMob66857.2025.11257542}
}
```
