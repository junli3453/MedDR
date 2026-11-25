<div align="center">

<h1>Causality-Guided Proactive Research Agents<br>for Personalized Medication Recommendation</h1>

<p><strong>
  Xiaoyan Zhao<sup>1</sup>, Xiang Li<sup>2</sup>, Zipeng Zhan<sup>2</sup>, Chunyuan Zheng<sup>2</sup>,<br>
  Yang Zhang<sup>1,&dagger;</sup>, Weihao Liu<sup>3</sup>, Haoxuan Li<sup>2,&dagger;</sup>, Tat-Seng Chua<sup>1</sup>
</strong></p>

<p>
  <sup>1</sup>National University of Singapore &nbsp;&nbsp;
  <sup>2</sup>Peking University &nbsp;&nbsp;
  <sup>3</sup>Renmin University of China
</p>

<p><sup>&dagger;</sup>Corresponding authors</p>

</div>

## Abstract

Medication recommendation aims to support accurate, safe, and personalized prescribing from electronic health records. Existing retrieval-augmented approaches often select evidence by semantic relevance rather than its utility for resolving medication uncertainty. MedDR reformulates this task as proactive deep research: a Planner develops patient-specific research objectives, a Questioner iteratively requests decision-relevant evidence from a frozen Retriever, and a Reporter updates uncertain medication recommendations under drug-drug interaction constraints. Counterfactual evidence interventions estimate the contribution of each research step through a necessity-and-sufficiency-inspired reward, which trains the Questioner with group relative policy optimization. The framework combines frozen deep-model predictions with evidence-grounded, personalized decision refinement.

MedDR operates on frozen deep-model medication probabilities. A SafeDrug baseline implementation is included. Datasets and pretrained weights are not distributed. For research use only.

## Framework

![MedDR framework](framework.png)

MedDR retains confident deep-model decisions and researches medications in the closed probability interval `[0.3, 0.7]`. The Planner uses same-patient history and up to three similar visits from distinct training patients to construct at most five research objectives. The Questioner performs up to five research rounds. The Reporter revises boundary-medication probabilities using the accumulated evidence and the aligned DDI graph.

The online training loop samples eight rollouts per visit. It combines the global medication-accuracy/DDI reward with step-wise PNS rewards estimated from ten Reporter samples per counterfactual condition. Group-normalized rewards update only the Questioner LoRA parameters before the next visit; the other agents and the deep recommender remain frozen. Prompts and output templates are in `LLM/meddr/prompts.py`.

## Full Pipeline

The project contains data preprocessing, SafeDrug, and the MedDR agent pipeline. Agent orchestration, prompts, counterfactual rewards, and online training are implemented in `LLM/meddr/`.

```text
MedDR/
├── data/
│   └── process.py
├── Deep_Model/
│   ├── model/
│   │   └── safedrug.py
│   └── main_SafeDrug.py
├── LLM/
│   ├── meddr/
│   │   ├── agents.py
│   │   ├── prompts.py
│   │   ├── retrieval.py
│   │   ├── pipeline.py
│   │   ├── rewards.py
│   │   ├── train.py
│   │   └── infer.py
│   ├── prepare_inputs.py
│   └── train_online.sh
├── framework.png
├── pyproject.toml
└── README.md
```

Run all commands from the repository root.

### Step 1. Install Dependencies

Use Python 3.9 or later and install the package:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

The Planner, Questioner, and Reporter use Qwen3-8B. The example commands assume local weights at `models/qwen3-8b`.

Configure the frozen Retriever with three environment variables: `MEDDR_RETRIEVER_MODEL` for its model identifier, `MEDDR_RETRIEVER_ENDPOINT` for the complete HTTPS chat-completions endpoint, and `MEDDR_RETRIEVER_KEY` for authentication. The endpoint must accept JSON chat-completions requests and return `choices[0].message.content`. Keep credentials outside version control. Patient-specific questions are sent to this service; use a deployment approved for the data being processed.

### Step 2. Preprocess EHR Data

The preprocessing script supports MIMIC-III and adapts a workflow similar to GiantMed: diagnosis/procedure encoding, NDC-to-RxCUI-to-ATC-3 mapping, molecular filtering, and DDI construction. Obtain authorized access to the required data and place the following files in `data/raw/`:

```text
DIAGNOSES_ICD.csv
PROCEDURES_ICD.csv
PRESCRIPTIONS.csv
D_ICD_DIAGNOSES.csv
D_ICD_PROCEDURES.csv
ndc2RXCUI.txt
ndc2atc_level4.csv
idx2SMILES.pkl
drug-atc.csv
drug-DDI.csv
```

`ndc2RXCUI.txt` contains a dictionary literal, `ndc2atc_level4.csv` supplies `RXCUI` and `ATC4`, and `idx2SMILES.pkl` maps ATC-3 codes to molecular SMILES strings. Load pickle files only from trusted sources.

```bash
python data/process.py \
  --raw-dir data/raw \
  --output-dir data/ready
```

The script produces patient-level train/validation/test records, vocabularies, molecular structures, the DDI adjacency matrix, a BRICS substructure mask, and clinical-name mappings. Each visit is represented as `[diagnosis_indices, procedure_indices, medication_indices]`. The first two-thirds of patients form the training split; the remaining patients are divided into test and validation splits in that order.

DDI constraints and DDI-rate computation use `data/ready/ddi_A_final.pkl`, with rows and columns aligned to the medication vocabulary. Set `MEDDR_DDI_MATRIX` or `--ddi-matrix` to use a different path.

Visits are ordered by the earliest retained prescription time. Patients must have at least two visits after modality filtering. Vocabulary construction and frequency filtering use the full cohort; model fitting uses only the training split. Processed records omit source patient and admission identifiers. MIMIC-IV raw tables require a separate adapter to this processed schema.

### Step 3. Generate Deep-Model Probabilities

Select a deep medication recommender, such as SafeDrug, MoleRec, DEPOT, or MedAlign, and run visit-level inference with a frozen checkpoint. MedDR uses the resulting probabilities to identify boundary medications; the downstream agent pipeline is independent of the recommender architecture.

Use the same medication vocabulary, patient splits, and visit order throughout preprocessing, deep-model inference, and MedDR. Export one probability for every medication, not only the predicted medication set. Save the training and test predictions as `outputs/base_model/train_predictions.json` and `outputs/base_model/test_predictions.json`.

Prediction files use the patient/visit organization of GiantMed: each patient has a positional `patient_id` and a `visits` list; each visit has a positional `visit_id` and `predicted` pairs of `[medication_id, probability]`. Medication IDs index `voc_final.pkl`. Pair order may vary, but IDs must uniquely cover the full medication vocabulary. Target labels are read separately from the training records.

The following example uses the included SafeDrug implementation. Other recommenders require their own training/inference implementation and an export in the same format.

<details>
<summary><strong>SafeDrug example</strong></summary>

Train SafeDrug and select the checkpoint with the highest validation Jaccard:

```bash
python -m Deep_Model.main_SafeDrug \
  --mode train \
  --data-dir data/ready \
  --checkpoint outputs/safedrug/best.pt
```

Training defaults to Adam, an embedding dimension of 64, and batches of 32 visits. Set the execution device with `--device`.

Export probabilities from the selected checkpoint:

```bash
for split in train test; do
  python -m Deep_Model.main_SafeDrug \
    --mode predict \
    --data-dir data/ready \
    --checkpoint outputs/safedrug/best.pt \
    --split "$split" \
    --output "outputs/base_model/${split}_predictions.json"
done
```

</details>

### Step 4. Prepare Visit Context and DDI Constraints

Convert the frozen probabilities into MedDR inputs using the processed records, vocabularies, and clinical-name mappings from Step 2. A medication-name mapping is also required at `data/ready/medication_names.json`: its JSON keys are ATC-3 vocabulary codes, and its values are unique readable medication-class names. It must cover the complete medication vocabulary.

```bash
for split in train test; do
  python LLM/prepare_inputs.py \
    --data-dir data/ready \
    --predictions "outputs/base_model/${split}_predictions.json" \
    --medication-names data/ready/medication_names.json \
    --split "$split" \
    --output "data/LLM-data/${split}.jsonl"
done
```

`prepare_inputs.py` joins each visit's probabilities with its current diagnoses/procedures and earlier same-patient history. It retrieves similar visits from training patients by diagnosis-set Jaccard similarity, excludes the target patient, and keeps up to three visits from distinct patients. The resulting JSONL files are the inputs to the Planner and the online research pipeline.

Only training JSONL files include target medication labels. The current visit contains no medication outcomes, and agent prompts omit labels and patient/sample keys. During validation, training, and inference, the loader derives DDI pairs for boundary and confidently accepted medications from `ddi_A_final.pkl`.

Visits without boundary medications are counted as `baseline_only_visits` and omitted from research inputs. Full-cohort evaluation combines MedDR predictions for research visits with the chosen recommender's unmodified decisions at threshold `0.5` for baseline-only visits.

Validate the prepared inputs:

```bash
python -m meddr.validate_data \
  --data data/LLM-data/train.jsonl \
  --mode training

python -m meddr.validate_data \
  --data data/LLM-data/test.jsonl \
  --mode inference
```

### Step 5. Run Online Research and GRPO Training

Run the Planner-Questioner-Retriever-Reporter pipeline on the prepared training visits and update the Questioner with online GRPO:

```bash
bash LLM/train_online.sh \
  --train-data data/LLM-data/train.jsonl \
  --questioner-model models/qwen3-8b \
  --planner-model models/qwen3-8b \
  --reporter-model models/qwen3-8b \
  --max-updates 100 \
  --output-dir outputs/meddr
```

The online loop alternates rollout collection and Questioner updates. `--max-updates` sets the number of updates. Run `bash LLM/train_online.sh --help` for all options.

For inference, use the saved Questioner adapter with the same frozen backbone, deep-model checkpoint, vocabulary, and DDI matrix:

```bash
python -m meddr.infer \
  --input-data data/LLM-data/test.jsonl \
  --questioner-model models/qwen3-8b \
  --planner-model models/qwen3-8b \
  --reporter-model models/qwen3-8b \
  --adapter-path outputs/meddr/adapter-final \
  --output-file outputs/meddr/predictions.jsonl
```

## Citation

```bibtex
@misc{zhao2027meddr,
  title={Causality-Guided Proactive Research Agents for Personalized Medication Recommendation},
  author={Zhao, Xiaoyan and Li, Xiang and Zhan, Zipeng and Zheng, Chunyuan and Zhang, Yang and Liu, Weihao and Li, Haoxuan and Chua, Tat-Seng},
  year={2027}
}
```

## Acknowledgements

We follow a data preprocessing approach similar to [GiantMed](https://github.com/lvhangkenn/GiantMed), adapt it for MedDR, and thank its authors for making their code publicly available.
