# A General Noisy Environment Adaptation Framework (NEAF) for Neural Speech Codecs (NSCs)

## 📢 News

[June 4, 2026] - Our paper "Noisy Environment Adaptation of Neural Speech Codec via Focal Mask and Noise Feature Separation" has been **accepted** for **INTERSPEECH 2026**!

[June 6, 2026] - :wave: Hi, we’d like to invite you to listen to our <a href='https://shaokai1209.github.io/FocalSE/' target='_blank' style='text-decoration: none;'><img src='https://img.shields.io/badge/🎧_Audio_Demo-1DB954?style=flat' alt='Audio Demo' style='border-radius: 14px; padding: 8px 16px; background-color: #1DB954;'></a>.


## 🚀 Training Guides

Select your target codec below for detailed training instructions:

<!-- ============================================================ -->
<!-- WavTokenizer 板块 -->
<!-- ============================================================ -->
<details>
<summary><b>WavTokenizer_wNEAF</b> — Single-stage VQ speech codec</summary>

### WavTokenizer Noise-Robust Adaptation Training Guide

This repository contains the noise-robust adaptation source code for WavTokenizer, located in the `wavtokenizer_wNEAF` directory. The training pipeline follows a two-stage workflow: clean speech pre-training, followed by noisy scenario adaptation.

#### Stage 1: Clean Pre-training Stage

This stage trains the baseline WavTokenizer model on a clean speech dataset, with the runtime environment fully aligned with the official WavTokenizer release.

1. Navigate to the adaptation source directory
    ```bash
    cd wavtokenizer_wNEAF
    ```

2. Create and activate the conda environment
    ```bash
    conda create -n wavtokenizer python=3.9
    conda activate wavtokenizer
    ```

3. Install project dependencies
    ```bash
    pip install -r requirements.txt
    ```

4. Modify the training configuration
    - Configuration file path: `your_path/.../wavtokenizer_wNEAF/configs/wavtokenizer_24khz_900bps.yaml`
    - Replace all paths containing `shaokai` with your local absolute paths
    - Adjust training parameters (batch size, dataset path, save directory, etc.) according to your needs

5. Start pre-training
    ```bash
    python train.py fit --config your_path/.../wavtokenizer_wNEAF/configs/wavtokenizer_24khz_900bps.yaml
    ```

#### Stage 2: Noisy Adaptation Stage

This stage fine-tunes the Focal-SE denoising module based on the pre-trained weights from Stage 1, to improve the model's encoding and reconstruction quality under noisy environments.

1. Return to the parent directory
    ```bash
    cd ..
    ```

2. Create the conda environment for adaptation training
    ```bash
    conda create -n neaf_nsc python=3.12
    ```

3. Full environment dependency setup
    ```bash
    cd NoiseRobustVRVQ-main
    ```
    | Item                 | Command / Details                                                                                                          |
    |----------------------|----------------------------------------------------------------------------------------------------------------------------|
    | **Python**           | 3.12                                                                                                                       |
    | **Create Conda env** | `conda create -n neaf_nsc python=3.12`                                                                                    |
    | **Activate env**     | `conda activate neaf_nsc`                                                                                                  |
    | **PyTorch**          | `conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia`          |
    | **Mamba-SSM**        | `pip install mamba-ssm==1.2.0.post1 --no-build-isolation`                                                               |
    | **Other dependencies** | `pip install -r requirements.txt`                                                                                        |

5. Activate the environment and navigate to the source directory
    ```bash
    conda activate neaf_nsc
    cd ..
    cd wavtokenizer_wNEAF
    ```

6. Modify the adaptation training configuration
    - Configuration file path: `your_path/.../wavtokenizer_wNEAF/configs/wavtokenizer_focalse.yaml`
    - Replace all paths containing `shaokai` with your local absolute paths
    - Fill in the pre-trained weight path generated in Stage 1
    - Adjust denoising training parameters according to your needs

7. Start noisy adaptation training
    ```bash
    python train.py fit --config your_path/.../wavtokenizer_wNEAF/configs/wavtokenizer_focalse.yaml
    ```

#### Training Output

Trained model checkpoints and training logs will be saved to the `save_dir` specified in the corresponding configuration file.

</details>

<!-- ============================================================ -->
<!-- DAC 板块 -->
<!-- ============================================================ -->
<details>
<summary><b>DAC_wNEAF</b> — High-fidelity neural audio codec</summary>

### DAC Noise-Robust Adaptation Training Guide

This repository contains the noise-robust adaptation source code for DAC, located in the `NoiseRobustVRVQ-main` directory. The training pipeline follows a two-stage workflow: clean speech pre-training, followed by noisy scenario adaptation.

#### Environment Setup

Navigate to the project directory and set up the runtime environment:

```bash
cd NoiseRobustVRVQ-main
```

Full environment dependency setup:

| Item                 | Command / Details                                                                                                          |
|----------------------|----------------------------------------------------------------------------------------------------------------------------|
| **Python**           | 3.12                                                                                                                       |
| **Create Conda env** | `conda create -n neaf_nsc python=3.12`                                                                                     |
| **Activate env**     | `conda activate neaf_nsc`                                                                                                  |
| **PyTorch**          | `conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia`                |
| **Mamba-SSM**        | `pip install mamba-ssm==1.2.0.post1 --no-build-isolation`                                                                 |
| **Other dependencies** | `pip install -r requirements.txt`                                                                                        |

#### Stage 1: Clean Pre-training Stage

This stage trains the DAC model on a clean speech dataset.

1. Modify the training configuration

    - Configuration file path: `./conf/stage1_clean_recon/CBR_16k.yml`
    - Replace all paths containing `shaokai` with your local absolute paths
    - Adjust dataset paths, save directory, and other training parameters according to your needs

2. Start pre-training

    ```bash
    CUDA_VISIBLE_DEVICES=5 taskset -c 24-39 python ./scripts/stage1_train_clean.py \
        --args.load ./conf/stage1_clean_recon/CBR_16k.yml \
        --save_path ./stage1/CBR_16k_625bps \
        --batch_size 16 \
        --val_batch_size 4
    ```

#### Stage 2: Noisy Adaptation Stage

This stage fine-tunes the NEAF denoising method based on the pre-trained DAC weights from Stage 1, to improve the model's encoding and reconstruction quality under noisy environments.

1. Modify the adaptation training configuration

    - Configuration file path: `./conf/stage2_denoising/CBR_focalSE.yml`
    - Replace all paths containing `shaokai` with your local absolute paths
    - Fill in the pre-trained weight path generated in Stage 1
    - Adjust denoising training parameters according to your needs

2. Start noisy adaptation training

    ```bash
    CUDA_VISIBLE_DEVICES=6,7 taskset -c 24-39 torchrun \
        --nproc_per_node=2 \
        --master_port=38650 \
        ./scripts/stage2_train_NANSC.py \
        --args.load ./conf/stage2_denoising/CBR_focalSE.yml \
        --save_path ./stage2/CBR_feat_denoise_16k \
        --batch_size 24 \
        --val_batch_size 4
    ```

</details>

<!-- ============================================================ -->
<!-- HiFiCodec 板块 -->
<!-- ============================================================ -->
<details>
<summary><b>HiFiCodec_wNEAF</b> — High-fidelity neural speech codec</summary>

### HiFiCodec Noise-Robust Adaptation Training Guide

*To be updated.*

</details>

## 🔮 Acknowledgments

We thank the authors for their excellent works:

[1] Kumar R, Seetharaman P, Luebs A, et al. High-fidelity audio compression with improved rvqgan[J].
    Advances in Neural Information Processing Systems, 2023, 36: 27980-27993.
    [DAC Code](https://github.com/descriptinc/descript-audio-codec)

[2] Yang D, Liu S, Huang R, et al. Hifi-codec: Group-residual vector quantization for high fidelity audio
    codec[J]. arXiv preprint arXiv:2305.02765, 2023.
    [HiFiCodec Code](https://github.com/yangdongchao/AcademiCodec)

[3] Chae Y, Lee K. Towards Bitrate-Efficient and Noise-Robust Speech Coding with Variable Bitrate RVQ[C]//
    Proc. Interspeech 2025. 2025: 609-613.
    [FDCBR Code](https://github.com/yoongi43/NoiseRobustVRVQ)

[4] Ji S, Jiang Z, Wang W, et al. Wavtokenizer: an efficient acoustic discrete codec tokenizer for audio
    language modeling[C]//International Conference on Learning Representations. 2025, 2025: 93809-93826.
    [WavTokenizer Code](https://github.com/jishengpeng/WavTokenizer)

[5] Della Libera L, Paissan F, Subakan C, et al. FocalCodec: Low-bitrate speech coding via focal modulation
    networks[J]. Advances in Neural Information Processing Systems, 2026, 38: 23742-23767.
    [FocalCodec Code](https://github.com/lucadellalib/focalcodec)

