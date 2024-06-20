# src/start.ps1
param (
    [string]$hf_token,
    [string]$flock_api_key,
    [string]$task_id,
    [string]$validation_args_file,
    [string[]]$other_args
)

function Show-Usage {
    Write-Host "Usage: .\start.ps1 -hf_token <token> -flock_api_key <key> -task_id <id> [-validation_args_file <file>] [-other_args <args>]"
    exit 1
}

# Check required parameters
if (-not $hf_token -or -not $flock_api_key -or -not $task_id) {
    Show-Usage
}

$env:HF_TOKEN = $hf_token
$env:FLOCK_API_KEY = $flock_api_key

while ($true) {
    $args = @("--task_id", $task_id)
    if ($validation_args_file) {
        $args += @("--validation_args_file", $validation_args_file)
    }
    if ($other_args) {
        $args += $other_args
    }

    python validate.py loop @args
    $exit_code = $LASTEXITCODE

    if ($exit_code -eq 100) {
        Write-Host "CUDA error detected, restarting the process..."
        continue
    } elseif ($exit_code -ne 0) {
        Write-Host "Validation failed with exit code $exit_code"
        exit $exit_code
    }
    break
}