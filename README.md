# MEFold: A Memory-Efficient Approach for Protein Language Model Using Chunk and Quantization

MEFold is based on ESMFold with inference memory optimization.

The main optimization techniques are as follows:
1. weight-only quantization
2. look-up table chunk

## Usage

### How to install
You must have PyTorch installed to use this repository, then follow these commands to configure the environment.

```bash
git clone https://github.com/llwx593/MEFold.git
cd MEFold
pip install "fair-esm[esmfold]"
pip install 'dllogger @ git+https://github.com/NVIDIA/dllogger.git'
pip install 'openfold @ git+https://github.com/aqlaboratory/openfold.git@4b41059694619831a7db195b7e0988fc4ff3a307'
git clone https://github.com/qwopqwop200/GPTQ-for-LLaMa
cd GPTQ-for-LLaMa
pip install -r requirements.txt
cd ..
```

### Quick start
For quick use, you can use our pre-computed chunk size configuration. 

First we need to quantize the model to get the quantized weights. After storing the quantized weights, we don't need to do this step again afterwards and can load the quantized weights directly.

```bash
python weight_only.py --wbits 4 --save "esmfold_rtn_quant.pt"
```

Next, we get the chunk size configuration for the input sequence with the following command.

```bash
python lookuptable_chunk.py --task real_inference --input_seqlen 922
```

We get the corresponding chunk size configuration from the output of the above program, and choose the appropriate scheme according to the actual memory.

```bash
python test.py --wbits 4 --load "esmfold_rtn_quant.pt" --cs1 15 --cs2 461 --cs3 231
```

## Acknowledgements
This code is based on [ESM](https://github.com/facebookresearch/esm) and [GPTQ for LLaMA](https://github.com/qwopqwop200/GPTQ-for-LLaMa)