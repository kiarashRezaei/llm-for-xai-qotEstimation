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

[The experiment runner](run_genxnet.py) trains the XGBoost BER regressor, computes local SHAP contributions and interactions, and generates paired baseline and proposed explanations. Table A-I in the paper describes the proposed prompt structure, and Table A-II gives examples of both prompts and their outputs.

Download the lightpath NetCDF file for Dataset 03 (TSNN, 1,321,452 samples) from the [Fraunhofer HHI QoT Dataset Collection](https://www.hhi.fraunhofer.de/en/pn-software/qot-dataset-collection.html). The experiment notebook used `lightpath_dataset_1.nc` as its local filename. Use Python 3.11 or 3.12 and install the dependencies in a virtual environment.

```bash
python -m pip install -r requirements.txt
python run_genxnet.py /path/to/lightpath_dataset_1.nc --prompts-only --samples 2 --output prompts.csv
```

To generate the 40 paired explanations, start [Ollama](https://ollama.com/) and make a 4-bit DeepSeek-R1 32B model available.

```bash
ollama pull deepseek-r1:32b
python run_genxnet.py /path/to/lightpath_dataset_1.nc --output explanations.csv
```

Each output row includes the source row ID, predicted BER, and both prompts. A full run also includes both generated explanations. The runner computes each BER prediction on the same sampled lightpath used for its SHAP values. This fixes a row alignment issue in the research notebook. The table above reports the published human evaluation of the original outputs.

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
