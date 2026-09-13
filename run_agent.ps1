<#
Launches the harness CLI using the ml-env conda Python interpreter whose
path is recorded in python_path.txt (kept next to this script).

Usage:
    .\run_agent.ps1 "list every python file in this project and summarize what each does"
    .\run_agent.ps1                      # interactive session
    .\run_agent.ps1 --root "C:\some\other\project" "do the thing"
#>
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Args
)

$root = $PSScriptRoot
$pythonPathFile = Join-Path $root "python_path.txt"

if (-not (Test-Path $pythonPathFile)) {
    Write-Error "python_path.txt not found next to run_agent.ps1 at: $pythonPathFile"
    exit 1
}

$pythonPath = (Get-Content $pythonPathFile -Raw).Trim()

if (-not (Test-Path $pythonPath)) {
    Write-Error "Python interpreter not found at: $pythonPath (check python_path.txt)"
    exit 1
}

& $pythonPath -m harness @Args
exit $LASTEXITCODE
