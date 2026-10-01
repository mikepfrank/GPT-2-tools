[CmdletBinding()]
param(
    [string] $Config = (Join-Path $PSScriptRoot 'experiments\identity-context.json'),

    [string] $OutputRoot = (Join-Path $PSScriptRoot 'outputs\experiments'),

    [switch] $Offline,

    [switch] $DryRun,

    [ValidateRange(1, 100)]
    [Nullable[int]] $Serendipity,

    [Nullable[double]] $Temperature,

    [ValidateRange(1, 1024)]
    [Nullable[int]] $Threads
)

$ErrorActionPreference = 'Stop'
if ($null -ne $Temperature -and $null -eq $Serendipity) {
    throw '-Temperature is only valid together with -Serendipity.'
}

$ProjectRoot = $PSScriptRoot
$LinuxProjectRoot = (& wsl.exe -d Ubuntu -- wslpath -a $ProjectRoot).Trim()
if ($LASTEXITCODE -ne 0 -or -not $LinuxProjectRoot) {
    throw 'Could not translate the project path for Ubuntu under WSL.'
}

$ResolvedConfig = (Resolve-Path -LiteralPath $Config).Path
$ResolvedOutputRoot = [IO.Path]::GetFullPath($OutputRoot)
$LinuxConfig = (& wsl.exe -d Ubuntu -- wslpath -a $ResolvedConfig).Trim()
if ($LASTEXITCODE -ne 0 -or -not $LinuxConfig) {
    throw 'Could not translate the experiment config path for Ubuntu under WSL.'
}
$LinuxOutputRoot = (& wsl.exe -d Ubuntu -- wslpath -a $ResolvedOutputRoot).Trim()
if ($LASTEXITCODE -ne 0 -or -not $LinuxOutputRoot) {
    throw 'Could not translate the experiment output path for Ubuntu under WSL.'
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
    '-m', 'gpt2_local.experiment',
    $LinuxConfig,
    '--output-root', $LinuxOutputRoot
)

if ($Offline) {
    $WslArguments += '--offline'
}
if ($DryRun) {
    $WslArguments += '--dry-run'
}
if ($null -ne $Serendipity) {
    $WslArguments += @(
        '--serendipity',
        $Serendipity.ToString([Globalization.CultureInfo]::InvariantCulture)
    )
    if ($null -ne $Temperature) {
        $WslArguments += @(
            '--temperature',
            $Temperature.ToString([Globalization.CultureInfo]::InvariantCulture)
        )
    }
}
if ($null -ne $Threads) {
    $WslArguments += @(
        '--threads',
        $Threads.ToString([Globalization.CultureInfo]::InvariantCulture)
    )
}

& wsl.exe @WslArguments
exit $LASTEXITCODE
