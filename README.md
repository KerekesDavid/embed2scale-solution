# A winning solution to the Embed2Scale challenge

This solution achieved the highest q-mean value of 15.22 on the [evaluation leaderboard](https://eval.ai/web/challenges/challenge-page/2465/leaderboard/6117) of the [Embed2Scale](https://eval.ai/web/challenges/challenge-page/2465/overview) challenge.

## Usage

Clone the repo:

```
git clone git@github.com:KerekesDavid/embed2scale-solution.git
cd embed2scale-solution
```

Clone Copernicus-FM, a dependency for generating the foundation model embeddings:

```
git clone https://github.com/zhu-xlab/Copernicus-FM.git ./foundation_embeddings/Copernicus-FM
```

Install dependencies:

```
mamba env create --file environment.yaml
mamba activate embed2scale-solution
```

Download the foundation models under `foundation_embeddings/pretrained_models`:

```
wget https://huggingface.co/wangyi111/Copernicus-FM/resolve/main/CopernicusFM_ViT_base_varlang_e100.pth -P foundation_embeddings/pretrained_models/

wget https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-600M/resolve/main/Prithvi_EO_V2_600M.pt-P foundation_embeddings/pretrained_models/

wget https://github.com/bair-climate-initiative/scale-mae/releases/download/base-800/scalemae-vitlarge-800.pth -P foundation_embeddings/pretrained_models/

wget https://huggingface.co/antofuller/CROMA/resolve/main/CROMA_large.pt -P foundation_embeddings/pretrained_models/
```

Download the Embed2Scale dataset to `./data`, or symlink it from elsewhere if you already have it:

```
git lfs install
git clone https://huggingface.co/datasets/embed2scale/SSL4EO-S12-downstream ./data/SSL4EO-S12-downstream
```

Run the foundation feature extraction:

```
cd foundation_embeddings
bash ./generate_foundation_embeddings.sh
```

Train the autoencoder to generate the evaluation embeddings:

```
python3 train_encoder.py
```

Point `embeddings_to_csv.py` to the newly generated experiment folder to generate a submission file:

```
python3 embeddings_to_csv.py experiments/<experiment folder> submission.csv
```

