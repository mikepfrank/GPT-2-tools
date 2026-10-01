[CmdletBinding()]
param(
    [ValidateRange(1, 65535)]
    [int] $Port = 8765,

    [switch] $Offline,

    [ValidateRange(1, 2147483647)]
    [Nullable[int]] $Threads
)

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
    '-m', 'gpt2_local.chat',
    '--port', $Port.ToString([Globalization.CultureInfo]::InvariantCulture)
)

if ($Offline) {
    $WslArguments += '--offline'
}
if ($null -ne $Threads) {
    $WslArguments += @(
        '--threads',
        $Threads.ToString([Globalization.CultureInfo]::InvariantCulture)
    )
}

Write-Host "Open http://localhost:$Port/ in your browser after the server starts."
Write-Host 'Keep this terminal open. Press Ctrl+C to stop the chat server.'

& wsl.exe @WslArguments
exit $LASTEXITCODE
