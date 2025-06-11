# Far Terrain Diffusion

Using a custom diffusion model to generate terrain heightmaps.

## Training
Place `.tif` heightmaps in the `raw` directory and run the `process.py` script to process the data.

See the `dataset_exp.ipynb` notebook for some examples from the resulting dataset.

See the `train.ipynb` notebook for training the diffusion model.

## Inference
Run the `model.py` script to generate terrain heightmaps using the trained diffusion model.

## Game Server
Run the `server.py` script to start a game server that serves the generated terrain heightmaps. This will use the `model.pth` provided in the repository.