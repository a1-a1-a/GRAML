# Generic zero-shot LLM baseline runner.
# Pair with infer_openai_baseline.py in the same folder.
# To run a different model, set MODEL / BASE_URL / OPENAI_API_KEY below.
# Outputs are written to <repo>/predictions/zero_shot/.

$env:OPENAI_API_KEY = "sk-XXX"

$BASE_URL = "http://127.0.0.1:8317/v1"
$MODEL = "gpt-5"
$SCRIPT = "$PSScriptRoot\infer_openai_baseline.py"
$DATASET_DIR = "$PSScriptRoot\..\..\..\Dataset\ood"
$OUTPUT_DIR = "$PSScriptRoot\..\..\..\predictions\zero_shot"

$datasets = @(
    @{name="CVEfixes"; path="$DATASET_DIR\CVEfixes\test.json"},
    @{name="Devign"; path="$DATASET_DIR\Devign\test.json"},
    @{name="DiverseVul"; path="$DATASET_DIR\DiverseVul\test.json"},
    @{name="Juliet"; path="$DATASET_DIR\Juliet\test.json"},
    @{name="PrimeVul"; path="$DATASET_DIR\PrimeVul\test.json"},
    @{name="ReVeal"; path="$DATASET_DIR\ReVeal\test.json"}
)

foreach ($ds in $datasets) {
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "Running: $($ds.name)" -ForegroundColor Cyan
    Write-Host "========================================"
    $output = "$OUTPUT_DIR\$($ds.name).jsonl"
    $log = "$OUTPUT_DIR\$($ds.name)_log.txt"
    
    python $SCRIPT `
        --test_path $ds.path `
        --output_path $output `
        --model $MODEL `
        --base_url $BASE_URL `
        --api_key_env "OPENAI_API_KEY" `
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
