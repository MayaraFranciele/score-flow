# ScoreFlow

## Team members
- Allan Reis da Conceição
- Gabriel Noriler Souza
- Daniel Barbosa Alves
- Mayara Franciele Santos da Silva

## Project overview
ScoreFlow is a machine learning application designed to estimate the default risk of a credit applicant based on financial and personal profile data. The project uses a synthetic dataset, trains and compares classification models, selects the best model, and exposes a lightweight web dashboard where a user can insert customer information and obtain a risk prediction in real time.

## Project report
A complete technical report for this project is available in [docs/relatorio.md](docs/relatorio.md).

To find it quickly, open the repository folder named `docs` and access the file `relatorio.md`. The document contains the academic and technical explanation of the project, including:
- project introduction and objectives
- problem definition and context
- dataset description and preprocessing
- methodology and feature engineering
- model comparison and training process
- evaluation metrics and results
- architecture and implementation details
- conclusions and final considerations

## Objective
The main goal of this project is to build a complete end-to-end ML workflow that includes:
- data generation and preparation
- model training and validation
- feature engineering
- threshold tuning
- deployment in a simple web interface

The final application classifies the applicant as either low risk or high risk and shows an estimated risk percentage based on the trained model.

## Machine learning pipeline
The workflow was implemented as follows:
1. Generation of a synthetic credit dataset with realistic financial behaviors.
2. Data cleaning and preprocessing.
3. Calculation of derived variables such as income commitment ratio.
4. Comparison of multiple classification models.
5. Selection of the best model and threshold tuning.
6. Export of the trained artifact for use in the Flask application.

## Tech stack
- Python
- Flask
- Pandas
- NumPy
- scikit-learn
- Matplotlib
- Seaborn
- HTML, CSS and JavaScript

## Project structure
- `app.py`: Flask application and prediction endpoint.
- `treinar.py`: dataset generation, model training, validation and export.
- `config.py`: feature configuration used by the dashboard.
- `static/`: CSS and JavaScript files for the interface.
- `templates/`: HTML templates rendered by Flask.
- `data/`: generated dataset used in training.
- `requirements.txt`: project dependencies.
- `scoreflow_model.pkl`: serialized trained model created after running the training script.

## Features
- dark corporate dashboard interface
- dynamic risk gauge
- form with financial and profile fields
- prediction endpoint for model inference
- model evaluation metrics for comparison during training
- clean separation between training logic and web application logic

## Installation
Clone the repository and create a virtual environment:

```bash
git clone <repository-url>
cd scoreflow
python -m venv .venv
```

Activate the environment:

On Windows:
```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS/Linux:
```bash
source .venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Run the project
1. Generate the dataset and train the model:
```bash
python treinar.py
```

2. Start the web app:
```bash
python app.py
```

3. Open the app in the browser:
```text
http://127.0.0.1:5000
```

## Model behavior
The model uses structured financial features to estimate the probability of default. The prediction endpoint returns:
- risk level (`Low` or `High`)
- probability value
- threshold used for classification
- explanatory message

## Data and model notes
This project uses a synthetic dataset designed for academic and demonstrative purposes. The data generation process aims to approximate common credit assessment patterns without using real customer data.

The trained artifact is saved in the project root as `scoreflow_model.pkl` and is loaded by `app.py` when the application starts.