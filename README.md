# TAIDE-ContinuePretrain

## 建立環境

1. 安裝 [`conda`](https://docs.conda.io/projects/miniconda/en/latest) 或 [`mamba`](https://github.com/conda-forge/miniforge)，建議使用 `mamba`
2. 執行 `environment/setup.sh`
3. 設定預設的計畫、分區

## 自訂 SLURM 指令

### 修改預設的計畫、分區

```sh
conda env config vars set \
    SLURM_DEFAULT_ACCOUNT="GOVXXXXXX" \
    SLURM_DEFAULT_PARTITION="XXXXXXX"
```

### [`sbatchx`](src/taide_cp/cli/slurm/sbatchx.py)

簡化 `sbatch` 使用流程，自動選擇預設的計畫和分區，並自動 `sattach` 以便觀察輸出

```sh
COMMAND="..."
sbatchx -j "任務名稱" --nodes 4 "$COMMAND"
```

### [`srunx`](src/taide_cp/cli/slurm/srunx.py)

開啟一個分配了 SLURM 資源的互動式 Shell，以便執行各種指令，同樣會自動選擇預設的計畫和分區

註：CPU 和 RAM 等資源是根據 GPU 數量來決定的，因此當你不需要 GPU 卻需要大量 CPU 或 RAM 時，你還是必須分配 GPU 才行

```sh
srunx -g 2 # GPU=2、CPU=8、RAM=180GB
srunx -g 8 # GPU=8、CPU=32、RAM=720GB
```

### `si`

顯示預設分區的節點狀態

### `sq`

顯示當前使用者的所有任務

### `sqp`

顯示預設分區的所有任務

### `sqa`

顯示預設計畫的所有任務

## 訓練

訓練腳本使用 [LightningCLI](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.cli.LightningCLI.html) 實作，建議閱讀 [此教學](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html) 來瞭解更詳細的使用方法

### 設定檔

使用 YAML 檔案來設定訓練參數，可以參考 [config](config) 資料夾下的檔案，但因程式碼改動，有些設定檔可能會報錯

### [`scripts/main.py`](scripts/main.py)

訓練腳本

#### 使用 `config/XXX.yaml` 進行訓練

```sh
python scripts/main.py fit --config "config/XXX.yaml"
```

#### 從存檔點恢復訓練

```sh
python scripts/main.py fit \
    --config "config/XXX.yaml" \
    --trainer.logger.version "XXX" \
    --ckpt_path "XXX.ckpt"
```

#### 多節點訓練

```sh
sbatchx -j train --nodes 4 "python scripts/main.py fit --config \"config/XXX.yaml\""
```

參考 [`shell_scripts/train.sh`](shell_scripts/train.sh)

### [`scripts/cp/prepare_data_for_pre_training.py`](scripts/cp/prepare_data_for_pre_training.py)

因為 TWCC 在多節點運算的時候，每個程序只能有 4 個 CPU 核心，不預先處理的話速度會非常慢，我們可以先利用 `srunx -g 8` 分配 32 顆 CPU 核心，再執行此腳本進行資料預處理

```sh
srunx -g 8

python scripts/cp/prepare_data_for_pre_training.py \
    --dataset_kwargs="{'path': '...', 'data_dir': '...'}" \
    --tokenizer_path="..." \
    --dataset_path="..." \
    --max_length=4096 \
    --num_proc=32
```

### [`scripts/cp/convert_pl_to_hf.py`](scripts/cp/convert_pl_to_hf.py)

把 Lightning 的存檔點轉換成 HuggingFace Transformers 的格式

```sh
# 根據 CPU RAM 的需求來選擇 GPU 的數量
# 7B 的存檔點需要 90GB 以上的 RAM，因此通常會分配 2 顆 GPU
srunx -g 2 

python scripts/cp/convert_pl_to_hf.py \
    --checkpoint_path="XXX.ckpt" \
    --output_path="checkpoints/XXX"
```
