# LLM-MTLN

## Large Language Model Augmented Multi-Task Learning Network with Inter-Clause Modeling for Emotion-Cause Pair Extraction
This repository contains the code for the official implementation of the paper: Large Language Model Augmented Multi-Task Learning Network with Inter-Clause Modeling for Emotion-Cause Pair Extraction. 

## How to use it
### Step 1: Prepare your environments
Reference environment settings:
```
torch               1.12.0+cu11.3
torch_geometric     2.3.1
transformers        4.30.2
sentencepiece       0.1.99
scikit-learn        0.20.0
scipy               1.7.3
pandas              1.1.5
```

The code has been tested on Ubuntu 18.04 operating system using a piece of Tesla-V100-SXM2-32GB GPU.

### Step 2: Download the GLM-Large-Chinese

Please download the GLM-Large-Chinese from [here](https://huggingface.co/THUDM/glm-large-chinese/tree/main). And put the files to
```
./src/glm-large-chinese/
```

### Step 3: Train the model
Please change the directory to ./src/
```
cd ./src/
```
and run the following command
```
python main.py
```
You can use different model configurations, such as running the following command
```
python main.py --emotion_enhanced false
```