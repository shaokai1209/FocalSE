# FocalSE
### :wave: Hi, we’d like to invite you to listen to our <a href='https://shaokai1209.github.io/FocalSE/' target='_blank' style='text-decoration: none;'><img src='https://img.shields.io/badge/🎧_Audio_Demo-1DB954?style=flat' alt='Audio Demo' style='border-radius: 14px; padding: 8px 16px; background-color: #1DB954;'></a>.

- Our paper "Noisy Environment Adaptation of Neural Speech Codec via Focal Mask and Noise Feature Separation" has been **accepted** for **INTERSPEECH 2026** (June 4, 2026).

:stuck_out_tongue_winking_eye: The code will be made publicly available after the extended version of the work is published.


## 🚀 Training Guides by Codec

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
    | Item                 | Command / Details                                                                                                          |
    |----------------------|----------------------------------------------------------------------------------------------------------------------------|
    | **Python**           | 3.12                                                                                                                       |
    | **Create Conda env** | `conda create -n neaf_nsc python=3.12`                                                                                    |
    | **Activate env**     | `conda activate neaf_nsc`                                                                                                  |
    | **PyTorch**          | `conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia`          |
    | **Mamba-SSM**        | `pip install mamba-ssm==1.2.0.post1 --no-build-isolation`                                                               |
    | **Other dependencies** | `pip install -r requirements.txt`                                                                                        |

4. Activate the environment and navigate to the source directory
    ```bash
    conda activate neaf_nsc
    cd wavtokenizer_wNEAF
    ```

5. Modify the adaptation training configuration
    - Configuration file path: `your_path/.../wavtokenizer_wNEAF/configs/wavtokenizer_focalse.yaml`
    - Replace all paths containing `shaokai` with your local absolute paths
    - Fill in the pre-trained weight path generated in Stage 1
    - Adjust denoising training parameters according to your needs

6. Start noisy adaptation training
    ```bash
    python train.py fit --config your_path/.../wavtokenizer_wNEAF/configs/wavtokenizer_focalse.yaml
    ```

#### Training Output

Trained model checkpoints and training logs will be saved to the `save_dir` specified in the corresponding configuration file.

</details>

<!-- ============================================================ -->
<!-- DAC 板块（模板，后续填充） -->
<!-- ============================================================ -->
<details>
<summary><b>DAC_wNEAF</b> — High-fidelity neural audio codec</summary>

### DAC Noise-Robust Adaptation Training Guide

*To be updated.*

</details>

<!-- ============================================================ -->
<!-- HiFiCodec 板块（模板，后续填充） -->
<!-- ============================================================ -->
<details>
<summary><b>HiFiCodec_wNEAF</b> — High-fidelity neural speech codec</summary>

### HiFiCodec Noise-Robust Adaptation Training Guide

*To be updated.*

</details>
