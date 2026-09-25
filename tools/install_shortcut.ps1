<#
.SYNOPSIS
    Create Desktop and Start Menu shortcuts for the injector sizing tool.

.DESCRIPTION
    Makes the tool launchable like a conventional Windows application: a
    double-clickable icon on the Desktop and an entry in the Start Menu (so it
    also turns up in Windows search and can be pinned to the taskbar).

    The shortcut targets pythonw.exe rather than python.exe, so no console
    window appears. Working directory is pinned to the project root so the
    package imports no matter where the shortcut is invoked from.

.PARAMETER Uninstall
    Remove the shortcuts instead of creating them.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File tools\install_shortcut.ps1
    powershell -ExecutionPolicy Bypass -File tools\install_shortcut.ps1 -Uninstall
#>

param(
    [switch]$Uninstall,
    # Point the shortcuts at the built standalone executable instead of the
    # Python launcher. Use this if you want the shortcut to keep working
    # without a Python installation.
    [switch]$UseExe
)

$ErrorActionPreference = 'Stop'

$Root = Split-Path -Parent $PSScriptRoot
$Launcher = Join-Path $Root 'run_app.pyw'
$IconPath = Join-Path $Root 'assets\n2o_injector.ico'
$Name = 'HASTE'

$Desktop = [Environment]::GetFolderPath('Desktop')
$StartMenu = Join-Path ([Environment]::GetFolderPath('StartMenu')) 'Programs'
$Targets = @(
    (Join-Path $Desktop   "$Name.lnk"),
    (Join-Path $StartMenu "$Name.lnk")
)

if ($Uninstall) {
    foreach ($t in $Targets) {
        if (Test-Path $t) { Remove-Item $t -Force; Write-Host "removed  $t" }
        else { Write-Host "not present  $t" }
    }
    return
}

# --- standalone-executable mode --------------------------------------------
if ($UseExe) {
    # Prefer the one-folder build: it launches in a few seconds, where the
    # one-file build must unpack ~150 MB to temp on every start.
    $candidates = @(
        (Join-Path $Root "dist\$Name\$Name.exe"),
        (Join-Path $Root "dist\$Name.exe")
    )
    $AppExe = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
    if (-not $AppExe) {
        throw "no built executable found. Run: python -m PyInstaller tools\n2o_injector.spec --noconfirm"
    }

    $Shell = New-Object -ComObject WScript.Shell
    foreach ($t in $Targets) {
        $dir = Split-Path -Parent $t
        if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
        $sc = $Shell.CreateShortcut($t)
        $sc.TargetPath       = $AppExe
        $sc.WorkingDirectory = (Split-Path -Parent $AppExe)
        $sc.IconLocation     = "$IconPath,0"
        $sc.Description      = 'Size a multi-hole N2O injector orifice plate for a hybrid rocket motor'
        $sc.WindowStyle      = 1
        $sc.Save()
        Write-Host "created  $t"
    }
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($Shell) | Out-Null
    Write-Host ""
    Write-Host "Target : $AppExe"
    Write-Host "(standalone build - no Python needed)"
    return
}

# --- locate pythonw.exe -----------------------------------------------------
# Ask the interpreter where it actually lives rather than trusting the PATH
# entry. On Windows `python` often resolves to the Microsoft Store execution
# alias in WindowsApps, which is a stub that can redirect to the Store and does
# not necessarily correspond to the interpreter holding your installed
# packages. sys.executable gives the real binary.
$PythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $PythonCmd) { throw "python was not found on PATH; cannot build a shortcut." }

$RealPython = & python -c "import sys; print(sys.executable)" 2>$null
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($RealPython)) {
    throw "could not determine the Python interpreter path (python -c failed)."
}
$RealPython = $RealPython.Trim()
$PythonDir = Split-Path -Parent $RealPython

$PythonW = Join-Path $PythonDir 'pythonw.exe'
if (-not (Test-Path $PythonW)) {
    Write-Warning "pythonw.exe not found beside $RealPython; falling back to python.exe (a console window will appear)."
    $PythonW = $RealPython
}

# Verify this interpreter can actually import the GUI's dependencies, so a
# broken shortcut is caught here rather than as a silent no-op on double-click.
& $RealPython -c "import numpy, scipy, matplotlib, tkinter" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Warning "$RealPython cannot import numpy/scipy/matplotlib/tkinter - the shortcut will show an error dialog until those are installed."
}

foreach ($p in @($Launcher, $IconPath)) {
    if (-not (Test-Path $p)) { throw "missing required file: $p" }
}

# --- create the shortcuts ---------------------------------------------------
$Shell = New-Object -ComObject WScript.Shell
foreach ($t in $Targets) {
    $dir = Split-Path -Parent $t
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }

    $sc = $Shell.CreateShortcut($t)
    $sc.TargetPath       = $PythonW
    $sc.Arguments        = '"' + $Launcher + '"'
    $sc.WorkingDirectory = $Root
    $sc.IconLocation     = "$IconPath,0"
    $sc.Description      = 'Size a multi-hole N2O injector orifice plate for a hybrid rocket motor'
    $sc.WindowStyle      = 1
    $sc.Save()
    Write-Host "created  $t"
}

[System.Runtime.InteropServices.Marshal]::ReleaseComObject($Shell) | Out-Null

Write-Host ""
Write-Host "Interpreter : $PythonW"
Write-Host "Launcher    : $Launcher"
Write-Host "Start in    : $Root"
Write-Host ""
Write-Host "Double-click the Desktop icon, or press Start and type 'N2O'."
