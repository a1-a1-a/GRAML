if (-not $env:OPENAI_API_KEY) { throw "Set OPENAI_API_KEY before running." }
if (-not $env:BASE_URL) { throw "Set BASE_URL before running." }

$BASE_URL = $env:BASE_URL
$MODEL = if ($env:MODEL) { $env:MODEL } else { "gpt-5" }
$NUM_SHOTS = if ($env:NUM_SHOTS) { [int]$env:NUM_SHOTS } else { 4 }
$SCRIPT = "$PSScriptRoot\infer_openai_baseline.py"
$EXAMPLES = "$PSScriptRoot\few_shot_examples.json"
$DATASET_DIR = "$PSScriptRoot\..\..\Dataset\eval"
$MODEL_TAG = $MODEL -replace '[^A-Za-z0-9._-]', '_'
$OUTPUT_DIR = "$PSScriptRoot\..\..\predictions\few_shot\$MODEL_TAG"
$SUMMARY_CSV = "$OUTPUT_DIR\summary.csv"
New-Item -ItemType Directory -Force -Path $OUTPUT_DIR | Out-Null

$datasets = @(
    @{name="CVEfixes"; path="$DATASET_DIR\CVEfixes_test.json"},
    @{name="Devign"; path="$DATASET_DIR\Devign_test.json"},
    @{name="DiverseVul"; path="$DATASET_DIR\DiverseVul_test.json"},
    @{name="Juliet"; path="$DATASET_DIR\Juliet_test.json"},
    @{name="PrimeVul"; path="$DATASET_DIR\PrimeVul_test.json"},
    @{name="ReVeal"; path="$DATASET_DIR\ReVeal_test.json"},
    @{name="ID_Test"; path="$DATASET_DIR\ID_test.json"}
)

foreach ($ds in $datasets) {
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "Running: $($ds.name) ($NUM_SHOTS-shot)" -ForegroundColor Cyan
    Write-Host "========================================"
    $output = "$OUTPUT_DIR\$($ds.name).jsonl"
    $log = "$OUTPUT_DIR\$($ds.name)_log.txt"

    python $SCRIPT `
        --test_path $ds.path `
        --output_path $output `
        --summary_csv_path $SUMMARY_CSV `
        --model $MODEL `
        --base_url $BASE_URL `
        --api_key_env "OPENAI_API_KEY" `
        --examples_path $EXAMPLES `
        --num_shots $NUM_SHOTS `
        --max_example_code_chars 2000 `
        --num_workers 16 `
        --resume `
        --disable_thinking `
        --max_output_tokens 4 `
        --temperature 0.0 `
        --progress_every 20 `
        2>&1 | Tee-Object -FilePath $log

    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR on $($ds.name)! Check log: $log" -ForegroundColor Red
        break
    }
    Write-Host "DONE: $($ds.name)" -ForegroundColor Green
}

Write-Host ""
Write-Host "==============================" -ForegroundColor Green
Write-Host "ALL DATASETS COMPLETED" -ForegroundColor Green
Write-Host "=============================="
