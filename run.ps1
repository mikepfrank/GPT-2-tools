[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string] $Prompt,

    [ValidateRange(1, 1023)]
    [int] $MaxNewTokens = 80,

    [double] $Temperature = 0.8,

    [Nullable[int]] $Seed,

    [switch] $Offline,

    [switch] $IncludePrompt
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = $PSScriptRoot
$LinuxProjectRoot = (& wsl.exe -d Ubuntu -- wslpath -a $ProjectRoot).Trim()
if ($LASTEXITCODE -ne 0 -or -not $LinuxProjectRoot) {
    throw 'Could not translate the project path for Ubuntu under WSL.'
}

$LinuxHome = (& wsl.exe -d Ubuntu -- bash -lc 'printf %s "$HOME"').Trim()
if ($LASTEXITCODE -ne 0 -or -not $LinuxHome) {
    throw 'Could not locate the WSL home directory.'
}

$Runner = "$LinuxHome/.venvs/gpt2-xl/bin/python"
$WslArguments = @(
    '-d', 'Ubuntu',
    '--cd', $LinuxProjectRoot,
    '--', $Runner,
    '-m', 'gpt2_local.cli',
    '--max-new-tokens', $MaxNewTokens.ToString(),
    '--temperature', $Temperature.ToString([Globalization.CultureInfo]::InvariantCulture)
)

if ($Offline) {
    $WslArguments += '--offline'
}
if ($IncludePrompt) {
    $WslArguments += '--include-prompt'
}
if ($null -ne $Seed) {
    $WslArguments += @(
        '--seed',
        $Seed.ToString([Globalization.CultureInfo]::InvariantCulture)
    )
}
if ($Prompt) {
    $WslArguments += @('--', $Prompt)
}

& wsl.exe @WslArguments
exit $LASTEXITCODE
