# Sentiment Analysis project workflow using DVC

## Preparation

### 1. Download and Install Python and Anaconda

### 2. Create and activate conda env with anaconda prompt
```bash
conda create -n env_name python=3.12
conda activate env_name
```

### 3. Install python libraries
```bash
pip install -r requirements.txt
```

### 4. Download Spacy en_core_web_sm
```bash
python -m download spacy en_core_web_sm
```

### 5. Make the src folder a python module
```bash
set PYTHONPATH=C:\Users\path_to_directory\dvc_stt;%PYTHONPATH%
```