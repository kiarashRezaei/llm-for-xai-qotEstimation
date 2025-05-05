# LLM-Augmented Explainable AI with Mutual Feature Interactions: a Quality-of-Transmission Estimation Use Case

This repository contains the implementation and evaluation of the proposed LLM-Augmented XAI framework that combines Large Language Models (LLMs) with Explainable AI (XAI) techniques, specifically focusing on mutual feature interactions for Quality of Transmission (QoT) estimation use case in optical networks. This work is part of our paper submitted to the European Conference on Optical Communication (ECOC) 2025.

> **Note:** The content of this repository will be made available upon paper acceptance.

## Project Structure

The repository is organized into the following key directories:

- **`data/`**: Contains the datasets used for training and evaluating the machine learning model.
- **`ml_models/`**: Stores the trained machine learning models in `.pkl` format.
- **`results/`**: Organizes LLM responses and token usage statistics in separate folders for different prompt variations.
- **`shap_interactions/`**: Contains local SHAP interaction matrices for individual samples, saved as `.png` files.
- **`shap_explanations/`**: Stores local SHAP feature importance visualizations for each sample in `.png` format.
- **`Evaluations/`**: Contains expert evaluations of the generated explanations.

**Core Files:**
- **`requirements.txt`**: Lists all Python package dependencies required to run the project.
- **`LLM-Augmented XAI.ipynb`**: Main Jupyter notebook containing the implementation and analysis of the LLM-augmented XAI framework.

## Directory Tree

```
qotEstimation-xai-llm/
│
├── data/                      
│   └── lightpath_dataset_1.nc 
│
├── ml_models/ 
    └── trained_{ml_model_name}.pkl 
│                
├── results/                   
│   └── {model_name}/         
│       ├── {model_name}_responses_{format}.csv
│       ├── token_counts.csv
│       └── {model_name}_responses_{format}_postprocessed.csv
│
├── Evaluations/              
│   └── {model_name}/        
│       └── {model_name}_responses_{format}_{expert_name}.xlsx
│
├── shap_explanations/        
├── shap_interactions/        
├── requirements.txt          
└── LLM-Augmented XAI.ipynb  
```

## Setup

1. Create a new virtual environment (do not use the one in the repository):
```bash
python -m venv your-env-name
source your-env-name/bin/activate  # On Unix/macOS
# or
.\your-env-name\Scripts\activate  # On Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running Experiments

1. Open `LLM-Augmented XAI.ipynb` in your preferred Jupyter environment.

2. Configure your experiment by setting the parameters in the first cell (Experiment Configuration):
   ```python
   load_ml_model = True/False      # Load existing or train new ML model
   load_existing_shap = True/False # Load existing or compute new SHAP values
   instances_to_explain_subset = N  # Number of instances to explain (max 40)
   
   # Prompt Configuration
   use_txt_baseline = True/False
   use_xml = True/False           # Use XML format for explanations
   use_json = True/False          # Use JSON format for explanations

   # Suplementary Information
   use_shap_interaction = True/False # Set to True to use shap interaction in the prompt, False to not use it
   use_corr_matrix = False/False # Set to True to use correlation matrix in the prompt, False to not use it
   
   # An Example of LLM Configuration
   model_params = {
    'model_name': 'deepseek-r1:32b',
    'temperature': 0.1,
    'do_sample': True,
    'top_p': 0.95,
    'repetition_penalty': 1.1,
    'quantization': 'Q4_K_M'
   }

   use_shap_interaction = True/False # Set to True to use shap interaction in the prompt, False to not use it
   use_corr_matrix = False/False # Set to True to use correlation matrix in the prompt, False to not use it
   
   # Save results parameters
   postprocess = True/False       # Postprocess responses (for deepseek models)
   experiment_name = "your-experiment-name"
   model_name = f"your-model-name_{experiment_name}"

   plot = True/False # Plotting
   ```

3. Run all cells in the notebook to execute the experiment.
