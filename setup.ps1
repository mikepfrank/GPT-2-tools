[CmdletBinding()]
param(
    [switch] $DownloadModel
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = $PSScriptRoot
$LinuxProjectRoot = (& wsl.exe -d Ubuntu -- wslpath -a $ProjectRoot).Trim()
if ($LASTEXITCODE -ne 0 -or -not $LinuxProjectRoot) {
    throw 'Could not translate the project path for Ubuntu under WSL.'
}

& wsl.exe -d Ubuntu --cd $LinuxProjectRoot -- bash scripts/setup-wsl.sh
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

if ($DownloadModel) {
    $LinuxHome = (& wsl.exe -d Ubuntu -- bash -lc 'printf %s "$HOME"').Trim()
    if ($LASTEXITCODE -ne 0 -or -not $LinuxHome) {
        throw 'Could not locate the WSL home directory.'
    }
    & wsl.exe -d Ubuntu -- "$LinuxHome/.venvs/gpt2-xl/bin/gpt2-xl" --download-only
    exit $LASTEXITCODE
}
