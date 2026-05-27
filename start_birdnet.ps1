#!/usr/bin/env pwsh
<#
.SYNOPSIS
  BirdNET Bioacoustic Detection System - production startup script.
.DESCRIPTION
  Launches birdnet_recorder_enhanced.py with auto-restart, crash logging,
  health checks, and graceful shutdown.
#>

$ScriptPath = Split-Path -Parent $PSCommandPath
Set-Location -LiteralPath $ScriptPath

# -- Config --
$RecorderScript  = "birdnet_recorder_enhanced.py"
$LogDir          = ".\logs"
$CrashLogFile    = ".\crash.log"
$MaxRestarts     = 10
$RestartWindow   = 300   # seconds - resets counter after this window
$BackoffInitial  = 2     # seconds
$BackoffMax      = 60    # seconds

# -- Logging helpers --
if (-not (Test-Path -LiteralPath $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}
function Write-CrashLog {
    param([string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "$timestamp [CRASH] $Message" | Out-File -FilePath $CrashLogFile -Append
    Write-Host "[$timestamp CRASH] $Message" -ForegroundColor Red
}

# -- Pre-flight checks --
Write-Host "=== BirdNET Detection System ===" -ForegroundColor Cyan
Write-Host "Starting at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

# Check Python availability
$python = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $python) {
    Write-CrashLog "Python not found in PATH"
    exit 1
}
Write-Host " Python: $python" -ForegroundColor Green

# Check dependencies
$missing = @()
@("pyaudio","numpy","tensorflow","librosa","flask","birdnetlib") | ForEach-Object {
    $null = python -c "import $_" 2>$null
    if (-not $?) { $missing += $_ }
}
if ($missing.Count -gt 0) {
    Write-Host " Missing packages: $($missing -join ', ')" -ForegroundColor Yellow
    Write-Host " Run: pip install $($missing -join ' ')" -ForegroundColor Yellow
}

# -- Main loop with auto-restart --
$restartCount = 0
$firstCrashTime = $null
$backoff = $BackoffInitial

while ($restartCount -lt $MaxRestarts) {
    # Reset counter if window elapsed
    if ($firstCrashTime -and ((Get-Date) - $firstCrashTime).TotalSeconds -gt $RestartWindow) {
        $restartCount = 0
        $backoff = $BackoffInitial
        $firstCrashTime = $null
        Write-Host " Restart window elapsed - counter reset." -ForegroundColor Cyan
    }

    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $logFile = Join-Path -Path $LogDir -ChildPath "birdnet_$timestamp.log"

    Write-Host " Launching recorder (attempt $($restartCount+1)/$MaxRestarts)..." -ForegroundColor Green
    Write-Host " Log: $logFile" -ForegroundColor Gray

    # Launch recorder
    $proc = Start-Process -FilePath python -ArgumentList "-u", $RecorderScript -NoNewWindow -PassThru -RedirectStandardOutput $logFile -RedirectStandardError "${logFile}.err"

    try {
        $proc.WaitForExit()
        $exitCode = $proc.ExitCode
    } catch {
        Write-CrashLog "Process monitor error: $_"
        break
    }

    if ($exitCode -eq 0) {
        Write-Host " Recorder exited cleanly (code 0). Stopping." -ForegroundColor Green
        break
    }

    # Crash detected
    $restartCount++
    if (-not $firstCrashTime) { $firstCrashTime = Get-Date }

    Write-CrashLog "Exit code $exitCode - restart #$restartCount/$MaxRestarts"

    # Apply exponential backoff
    Write-Host " Waiting ${backoff}s before restart..." -ForegroundColor Yellow
    Start-Sleep -Seconds $backoff
    $backoff = [Math]::Min($backoff * 2, $BackoffMax)
}

if ($restartCount -ge $MaxRestarts) {
    Write-CrashLog "Exceeded $MaxRestarts restarts in ${RestartWindow}s - giving up"
    exit 1
}

Write-Host "=== System stopped ===" -ForegroundColor Cyan
