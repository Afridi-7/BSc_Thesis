# ============================================================================
# snap.ps1 — capture a clipboard image and save it to figures/webapp/
# ============================================================================
# Usage:
#   1. Press Win+Shift+S, drag a rectangle around the panel you want.
#      (The snip is now on your clipboard.)
#   2. From the repo root, run:
#         .\scripts\snap.ps1 01_detection
#      (the .png extension is added automatically)
# ============================================================================

param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Name
)

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

$img = [System.Windows.Forms.Clipboard]::GetImage()
if ($null -eq $img) {
    Write-Host "[snap] Clipboard does not contain an image. Use Win+Shift+S first." -ForegroundColor Red
    exit 1
}

$repoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$outDir = Join-Path $repoRoot "figures\webapp"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

if ($Name -notmatch '\.png$') { $Name = "$Name.png" }
$outPath = Join-Path $outDir $Name

$img.Save($outPath, [System.Drawing.Imaging.ImageFormat]::Png)
Write-Host "[snap] Saved $outPath ($($img.Width)x$($img.Height))" -ForegroundColor Green
