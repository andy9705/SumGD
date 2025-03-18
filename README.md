## Setup

We implement our Summary-Guided Decoding in `transformers-4.29.2/src/transformers/generation/utils.py` followed by OPERA, summary_guided_decoding.py, and summary_guided_decoding_distill.py

Follow below steps to use our code.
```
# Create and activate a new conda environment
conda create -n sumgd 
conda activate sumgd

# Install the modified transformers library
python -m pip install -e transformers-4.29.2

# Install dependencies
pip install -r requirements.txt
```
