param(
    [string]$PythonExe = "..\\..\\.venv312\\Scripts\\python.exe",
    [string]$RepoDir = "PaddleOCR",
    [int]$Epochs = 20,
    [int]$BatchSize = 128,
    [string]$PretrainedModel = "",
    [string]$LogDir = "artifacts\\logs"
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Resolve-Path (Join-Path $scriptDir "..\\..")
$python = Resolve-Path (Join-Path $scriptDir $PythonExe)
$repoPath = Join-Path $scriptDir $RepoDir
$labelDir = Join-Path $scriptDir "labels"
$dataRoot = Join-Path $projectRoot "ML\\data\\IIIT-HW-Telugu_v1"
$trainManifest = Join-Path $projectRoot "ML\\data\\manifests\\cvit_train.jsonl"
$valManifest = Join-Path $projectRoot "ML\\data\\manifests\\cvit_val.jsonl"
$saveDir = Join-Path $scriptDir "artifacts\\te_finetune"
$logDirPath = Join-Path $scriptDir $LogDir
$runTs = Get-Date -Format "yyyyMMdd_HHmmss"
$runLog = Join-Path $logDirPath "finetune_$runTs.log"
$trainLog = Join-Path $logDirPath "train_$runTs.log"

function Write-Log {
    param([string]$Message)
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "$ts | $Message"
    Write-Host $line
    Add-Content -Path $runLog -Value $line
}

New-Item -ItemType Directory -Force -Path $logDirPath | Out-Null
"" | Set-Content -Path $runLog

Write-Log "Fine-tune run started"
Write-Log "Project root: $projectRoot"
Write-Log "Python: $python"
Write-Log "Repo path: $repoPath"
Write-Log "Save dir: $saveDir"
Write-Log "Run log: $runLog"
Write-Log "Train log: $trainLog"

if (!(Test-Path $repoPath)) {
    Write-Log "Cloning PaddleOCR repo..."
    git clone https://github.com/PaddlePaddle/PaddleOCR.git $repoPath
}
else {
    Write-Log "PaddleOCR repo already exists, skipping clone"
}

Write-Log "Installing PaddleOCR training dependencies..."
& $python -m pip install -r (Join-Path $repoPath "requirements.txt")
Write-Log "Dependency install step complete"

Write-Log "Preparing PaddleOCR label files from manifests..."
& $python (Join-Path $scriptDir "prepare_paddleocr_labels.py") `
    --train-manifest $trainManifest `
    --val-manifest $valManifest `
    --data-root $dataRoot `
    --out-dir $labelDir

$trainLabelPath = Join-Path $labelDir "rec_gt_train.txt"
$valLabelPath = Join-Path $labelDir "rec_gt_val.txt"
if (!(Test-Path $trainLabelPath) -or !(Test-Path $valLabelPath)) {
    throw "Label files missing after preparation step"
}
$trainRows = (Get-Content $trainLabelPath | Measure-Object -Line).Lines
$valRows = (Get-Content $valLabelPath | Measure-Object -Line).Lines
Write-Log "Prepared labels: train_rows=$trainRows val_rows=$valRows"

$configPath = Join-Path $repoPath "configs\\rec\\PP-OCRv3\\multi_language\\te_PP-OCRv3_mobile_rec.yml"
if (!(Test-Path $configPath)) {
    throw "Missing config: $configPath"
}
Write-Log "Using config: $configPath"

Push-Location $repoPath
try {
    $opts = @(
        "Global.use_gpu=True",
        "Global.epoch_num=$Epochs",
        "Global.save_model_dir=$saveDir",
        "Train.dataset.data_dir=$dataRoot",
        "Train.dataset.label_file_list=['$labelDir/rec_gt_train.txt']",
        "Eval.dataset.data_dir=$dataRoot",
        "Eval.dataset.label_file_list=['$labelDir/rec_gt_val.txt']",
        "Train.loader.batch_size_per_card=$BatchSize"
    )

    if ($PretrainedModel -ne "") {
        $opts += "Global.pretrained_model=$PretrainedModel"
        Write-Log "Using pretrained model: $PretrainedModel"
    }
    else {
        Write-Log "No pretrained model override provided"
    }

    Write-Log "Training options: Epochs=$Epochs BatchSize=$BatchSize"
    Write-Log "Starting fine-tuning (live output follows)"

    & $python tools/train.py -c $configPath -o $opts 2>&1 | Tee-Object -FilePath $trainLog -Append
    $trainExit = $LASTEXITCODE
    if ($trainExit -ne 0) {
        throw "Training failed with exit code $trainExit"
    }
    Write-Log "Training command completed successfully"
}
finally {
    Pop-Location
}

Write-Log "Fine-tuning finished. Artifacts: $saveDir"
Write-Log "Run complete"
