param(
    [string]$CondaEnv = "pyjoern-env",
    [int]$Limit = 0,
    [switch]$ReuseCpg
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

where.exe joern | Out-Null
where.exe joern-parse | Out-Null
where.exe java | Out-Null

$EvidencePath = "Dataset\cpg_evidence\train_joern_cpg_description_only.json"
$WorkDir = "Dataset\cpg_evidence\joern_work_description_only"
$OutputTrain = "Dataset\ablations\direct_cpg\Ultimate_train_cpg_description_only.json"

$GenerateArgs = @(
    "Dataset\generate_joern_cpg_evidence.py",
    "--input_path", "Dataset\id\Ultimate_train.json",
    "--output_path", $EvidencePath,
    "--work_dir", $WorkDir,
    "--task_filter", "description",
    "--skip_benign_description",
    "--include_label_lines"
)

if ($Limit -gt 0) {
    $GenerateArgs += @("--limit", "$Limit")
}

if ($ReuseCpg) {
    $GenerateArgs += "--reuse_cpg"
}

conda run -n $CondaEnv python @GenerateArgs

conda run -n $CondaEnv python Dataset\scripts\build_evidence_dataset.py `
    --input_path Dataset\id\Ultimate_train.json `
    --output_path $OutputTrain `
    --variant cpg_evidence `
    --evidence_path $EvidencePath `
    --evidence_task_policy description_only `
    --preserve_raw_when_no_evidence

Write-Host ""
Write-Host "Done."
Write-Host "Evidence: $EvidencePath"
Write-Host "Training JSON: $OutputTrain"
Write-Host "Use raw valid/test JSON files for evaluation."
