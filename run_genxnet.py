"""Generate paired GenXNet prompts and explanations for optical lightpaths."""

import argparse
import csv
import json
import re
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


FEATURES = (
    "Path Len",
    "Avg Link Len",
    "Num Links",
    "Num Spans",
    "Freq",
    "Mod Order",
    "Sum Link Occ",
    "Std Link Occ",
    "Min Mod Order Left",
    "Max Mod Order Left",
    "Max Mod Order Right",
    "Min Mod Order Right",
)


def create_prompt_txt_baseline(formatted_explanations):

    return (
        "Below is the output of a SHAP explanation from a machine learning model that predicts "
        "Bit Error Rate (BER) in optical networks.\n\n"
        "Please write a short explanation of what this means and which features seem most important.\n\n"
        f"{formatted_explanations.strip()}"
    )


def create_prompt_txt(formatted_explanations, use_shap_interaction=False):
    base_intro = (
        "## Context\n"
        "You are an interpreter of a SHAP explanation from a machine learning model predicting "
        "**Bit Error Rate (BER)** for optical lightpaths.\n\n"
        "### Format\n"
        "- **Input**: Plain Text SHAP explanation\n"
        "- **Includes for each feature**:\n"
        "  - Min/Max Value\n"
        "  - Actual Value\n"
        "  - SHAP Value (signed)\n\n"
        "### SHAP Value Interpretation\n"
        "- **Positive SHAP** → Increases BER (undesired)\n"
        "- **Negative SHAP** → Decreases BER (desired)\n\n"
        "### Example\n"
        "ML Model Prediction: <value>\n"
        "Feature-wise breakdown:\n"
        "- Feature A: Min Value = x, Max Value = y, Value = z, SHAP Value= s\n"
        "- Feature B: Min Value = x, Max Value = y, Value = z, SHAP Value= s\n"
        "..."
    )

    base_task = (
        "## Task\n"
        "1. **Identify** the top 2–3 features with the highest absolute SHAP values.\n"
        "2. **For each feature**, describe:\n"
        "   - Its actual value\n"
        "   - Whether it increases or decreases BER (based on SHAP sign)\n"
        "3. **Summarize** with a concise explanation covering:\n"
        "   - Most impactful features\n"
        "   - Their influence on BER\n"
        "   - Practical insights to reduce BER\n"
    )

    interaction_task = (
        "## Interaction Handling (If Applicable)\n"
        "If a feature seems to reduce BER in a way that contradicts domain expectations:\n\n"
        "1. Refer to the **SHAP interaction values**.\n"
        "2. Identify the top 2–3 feature pairs with the highest absolute SHAP interaction values.\n"
        "3. For each pair:\n"
        "   - Name both features\n"
        "   - Describe the interaction effect on BER\n"
        "   - Clarify if the unexpected behavior results from the interaction, not individual features\n"
        "4. **Revise your conclusion**:\n"
        "   - Clearly state if an effect is interaction-driven\n"
        "   - Avoid misleading conclusions from individual SHAP values alone\n"
    )

    summary_instruction = (
        "## Input Explanation\n"
        f"```\n{formatted_explanations.strip()}\n```\n\n"
        "## Response Structure\n"
        "### 1. Interpretation\n"
        "- Key features, values, and SHAP values\n"
        "- Their effect on BER\n\n"
        "### 2. Conclusion\n"
        "- Actionable insights\n"
    )

    return (base_intro + "\n\n" + base_task + ("\n\n" + interaction_task if use_shap_interaction else "") + "\n\n" + summary_instruction).strip()


def format_shap_input(prediction, row, minima, maxima, influences, interactions=None):
    """Format local SHAP data in the same layout as the experiment notebook."""
    lines = [f"ML Model Prediction: {prediction}", "", "Feature-wise breakdown:"]
    for i, (feature, value) in enumerate(row.items()):
        minimum = minima[feature] if value >= minima[feature] else value
        maximum = maxima[feature] if value <= maxima[feature] else value
        lines.append(
            f"- {feature}: Min Value = {minimum}, Max Value = {maximum}, "
            f"Value = {value}, SHAP Value = {influences[i]:.4f}"
        )

    if interactions is not None:
        lines.extend(("", "Feature Interaction Breakdown:"))
        features = tuple(row)
        for i, feature in enumerate(features):
            for j in range(i + 1, len(features)):
                interaction = interactions[i][j]
                if abs(interaction) > 1e-5:
                    lines.append(
                        f"  - {feature} & {features[j]}: SHAP Interaction Value = {interaction:.6f}"
                    )
    return "\n".join(lines)


def prepare_cases(dataset, samples):
    import numpy as np
    import shap
    import xarray as xr
    from sklearn.model_selection import train_test_split
    from xgboost import XGBRegressor

    with xr.open_dataset(dataset) as source:
        data = source["data"].to_pandas()
        target = source["target"].to_pandas()["ber"]

    data.columns = [" ".join(word.capitalize() for word in name.split("_")) for name in data.columns]
    data = data.loc[:, list(FEATURES)]
    train_rows, test_rows, train_ber, _ = train_test_split(
        data, target, test_size=0.1, random_state=42, shuffle=True
    )
    if len(test_rows) < samples:
        raise ValueError(f"Requested {samples} explanations for a test set of {len(test_rows)} rows")

    model = XGBRegressor(eta=0.2, max_depth=9, subsample=0.9, random_state=42)
    model.fit(train_rows, train_ber)

    # The research notebook drew a pool of 100 test rows and used its first 40.
    rng = np.random.RandomState(42)
    indices = rng.choice(len(test_rows), size=min(100, len(test_rows)), replace=False)[:samples]
    explained_rows = test_rows.iloc[indices]
    predictions = model.predict(explained_rows)
    influences = shap.Explainer(model, train_rows)(explained_rows).values
    interactions = shap.TreeExplainer(model).shap_interaction_values(explained_rows)
    minima = train_rows.min()
    maxima = train_rows.max()

    for i, (row_id, row) in enumerate(explained_rows.iterrows()):
        feature_values = row.to_dict()
        baseline_input = format_shap_input(predictions[i], feature_values, minima, maxima, influences[i])
        proposed_input = format_shap_input(
            predictions[i], feature_values, minima, maxima, influences[i], interactions[i]
        )
        yield {
            "row_id": row_id,
            "predicted_ber": predictions[i],
            "baseline_prompt": create_prompt_txt_baseline(baseline_input),
            "proposed_prompt": create_prompt_txt(proposed_input, use_shap_interaction=True),
        }


def generate_explanation(prompt, model, ollama_url):
    wrapped_prompt = f"### Instruction: {prompt}\n\n### Response:"
    body = json.dumps({
        "model": model,
        "prompt": wrapped_prompt,
        "stream": False,
        # These were the effective defaults of the original Ollama call.
        "options": {"temperature": 0.7, "top_p": 0.95},
    }).encode("utf-8")
    request = Request(
        f"{ollama_url.rstrip('/')}/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=600) as response:
            result = json.load(response)
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"Ollama returned HTTP {error.code}: {detail}") from error
    except (URLError, TimeoutError) as error:
        raise RuntimeError(f"Could not get a response from Ollama at {ollama_url}: {error}") from error

    reply = result.get("response")
    if not isinstance(reply, str):
        raise RuntimeError(f"Ollama returned no explanation text: {result.get('error', 'missing response')}")
    reply = re.sub(r"<think>.*?</think>", "", reply, flags=re.DOTALL).replace(wrapped_prompt, "").strip()
    if not reply:
        raise RuntimeError("Ollama returned an empty explanation")
    return reply


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Path to the Dataset 03 lightpath NetCDF file")
    parser.add_argument("--output", type=Path, default=Path("genxnet_explanations.csv"))
    parser.add_argument("--samples", type=int, choices=range(1, 41), default=40, metavar="1–40")
    parser.add_argument("--model", default="deepseek-r1:32b", help="Ollama model tag")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--prompts-only", action="store_true", help="Export the prompts without calling Ollama")
    args = parser.parse_args()

    if not args.dataset.is_file():
        parser.error(f"dataset file does not exist: {args.dataset}")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")

    fields = ["row_id", "predicted_ber", "baseline_prompt", "proposed_prompt"]
    if not args.prompts_only:
        fields.extend(("baseline_explanation", "proposed_explanation"))

    try:
        cases = list(prepare_cases(args.dataset, args.samples))
        with args.output.open("x", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            for case in cases:
                if not args.prompts_only:
                    case["baseline_explanation"] = generate_explanation(
                        case["baseline_prompt"], args.model, args.ollama_url
                    )
                    case["proposed_explanation"] = generate_explanation(
                        case["proposed_prompt"], args.model, args.ollama_url
                    )
                writer.writerow(case)
                output.flush()
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"{error}\n")

    print(f"Wrote {args.samples} paired cases to {args.output}")


if __name__ == "__main__":
    main()
