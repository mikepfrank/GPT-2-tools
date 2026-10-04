[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$ProjectRoot = $PSScriptRoot
$LinuxProjectRoot = (& wsl.exe -d Ubuntu -- wslpath -a $ProjectRoot).Trim()
if ($LASTEXITCODE -ne 0 -or -not $LinuxProjectRoot) {
    throw 'Could not translate the project path for Ubuntu under WSL.'
}

$LinuxHome = (& wsl.exe -d Ubuntu -- printenv HOME).Trim()
if ($LASTEXITCODE -ne 0 -or -not $LinuxHome) {
    throw 'Could not locate the WSL home directory.'
}

$Runner = "$LinuxHome/.venvs/gpt2-xl/bin/python"
$WslArguments = @(
    '-d', 'Ubuntu',
    '--cd', $LinuxProjectRoot,
    '--', 'env', "PYTHONPATH=$LinuxProjectRoot/src",
    $Runner,
    '-m', 'gpt2_local.public_chat',
    '--host', '127.0.0.1',
    '--port', '8891',
    '--offline',
    '--threads', '2',
    '--enable-streaming',
    '--max-reply-tokens', '256',
    '--minimum-available-memory-gib', '8'
)

Write-Host 'Open http://localhost:8891/ after the ready message.'
Write-Host 'Keep this terminal open. Press Ctrl+C to stop the local preview.'

& wsl.exe @WslArguments
exit $LASTEXITCODE
