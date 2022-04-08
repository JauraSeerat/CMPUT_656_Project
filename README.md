# CMPUT 656 Project:  Entity linking by jointly encoding all candidate entity descriptions

### Participants:
|Student name      | CCID       |
|------------------|------------|
|Seeratpal Jaura   |  seeratpa  |
|Talgat Omarov     |  omarov    |


### Installation instructions
Create a virtual environment
`python3 -m venv venv`

Activate the virtual environment
`source venv/bin/activate`

Install pytorch following the instruction on the official website ([link](https://pytorch.org/get-started/locally/)). For example,

`pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113`

Install the remaining dependencies:
`pip install requirements.txt`

### Download artificats
Run the following command to download required artifacts:
`python3 download_data.py`

### Train a model
`python3 -m src.models.escher.train --max_steps <max steps> --gpus <number of gpus> --top_k_candidates <select top k candidates> --entity_length <specify the entity length> --batch_size <batch size>`

It will generate the fine tuned checkpoint in src/models/escher/checkpoint

### Evaluate the model
`python3 -m src.models.escher.evaluate --device 1 --ckpt_path <checkpoint path> --filename <val.json or test.json> --top_k_candidates <select top k candidates> --entity_length <specify the entity length> --output <prediction output path>`

The script dipslays the normalized and unnormalized scores.
