param(
    [string]$CondaEnv = "pyjoern-env",
    [int]$Limit = 0,
    [switch]$ReuseCpg,
    [switch]$EnableDataFlow
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $RepoRoot

where.exe joern | Out-Null
where.exe joern-parse | Out-Null
where.exe java | Out-Null

$EvidencePath = "Dataset\graph_context\cpg_evidence\joern_cpg_evidence.json"
if ($Limit -gt 0) {
    $EvidencePath = "Dataset\graph_context\cpg_evidence\joern_cpg_evidence_limit$Limit.json"
}
$WorkDir = "Dataset\graph_context\cpg_evidence\joern_work_description_only"
$OutputTrain = "Dataset\ablations\direct_cpg\Ultimate_train_direct_cpg.json"

$GenerateArgs = @(
    "Dataset\scripts\generate_joern_cpg_evidence.py",
    "--input_path", "Dataset\id\expansion\Ultimate_train_expanded.json",
    "--output_path", $EvidencePath,
    "--work_dir", $WorkDir,
    "--task_filter", "description",
    "--skip_benign_description"
)

if ($Limit -gt 0) {
    $GenerateArgs += @("--limit", "$Limit")
}
if ($ReuseCpg) {
    $GenerateArgs += "--reuse_cpg"
}
if ($EnableDataFlow) {
    $GenerateArgs += "--enable_joern_slice_data_flow"
}

conda run -n $CondaEnv python @GenerateArgs

if ($Limit -gt 0) {
    Write-Host "Limited run: evidence written to $EvidencePath; Direct CPG training file not rebuilt."
    exit 0
}

conda run -n $CondaEnv python Dataset\scripts\build_direct_cpg_target.py `
    --train_path Dataset\id\expansion\Ultimate_train_expanded.json `
    --joern_path $EvidencePath `
    --output_path $OutputTrain

Write-Host ""
Write-Host "Done."
Write-Host "Evidence: $EvidencePath"
Write-Host "Training JSON: $OutputTrain"
