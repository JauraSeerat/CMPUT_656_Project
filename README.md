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
`python3 -m src.models.escher.train --max_steps 10000 --gpus 2 --top_k_candidates 64 --entity_length 16 --save_top_k_ckpts 3 --batch_size 16 --wandb_project cmput656`

It will generate the fine tuned checkpoint in src/models/escher/checkpoint

### Evaluate the model
Find the checkpoint in src/models/escher/checkpoint folder. It could look like 
"src/models/escher/checkpoint/cmput656/2d40ip8t/checkpoints/epoch=1.ckpt"

`python3 -m src.models.escher.evaluate --device 1 --ckpt_path src/models/escher/checkpoint/cmput656_64cand_16entt/2d40ip8t/checkpoints/epoch=1.ckpt --filename test.json --top_k_candidates 64 --entity_length 11 --output output/predictions.txt`

The script dipslays the normalized and unnormalized scores.
