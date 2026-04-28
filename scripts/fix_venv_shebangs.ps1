# Patch the embedded shebang in every .venv\Scripts\*.exe launcher so they
# point at the current venv's python.exe. Needed when a venv folder is
# renamed/moved on Windows (the .exe shims have the old path baked in).
param(
    [string]$Venv = (Join-Path $PSScriptRoot '..\.venv')
)

$Venv = (Resolve-Path $Venv).Path
$newPy = Join-Path $Venv 'Scripts\python.exe'
if (-not (Test-Path $newPy)) { throw "python.exe not found at $newPy" }

$newShebang = "#!`"$newPy`"`r`n"
$newBytes = [System.Text.Encoding]::ASCII.GetBytes($newShebang)
$zipMagic = [byte[]](0x50,0x4B,0x03,0x04)

function Find-Bytes([byte[]]$hay, [byte[]]$needle) {
    for ($i = 0; $i -le $hay.Length - $needle.Length; $i++) {
        $ok = $true
        for ($j = 0; $j -lt $needle.Length; $j++) {
            if ($hay[$i + $j] -ne $needle[$j]) { $ok = $false; break }
        }
        if ($ok) { return $i }
    }
    return -1
}

$exes = Get-ChildItem (Join-Path $Venv 'Scripts\*.exe')
$fixed = 0; $skipped = 0
foreach ($e in $exes) {
    $bytes = [System.IO.File]::ReadAllBytes($e.FullName)
    $zipAt = Find-Bytes $bytes $zipMagic
    if ($zipAt -lt 0) { Write-Host "SKIP (no zip): $($e.Name)"; $skipped++; continue }
    $searchStart = [Math]::Max(0, $zipAt - 2048)
    $shAt = -1
    for ($i = $zipAt - 2; $i -ge $searchStart; $i--) {
        if ($bytes[$i] -eq 0x23 -and $bytes[$i+1] -eq 0x21) { $shAt = $i; break }
    }
    if ($shAt -lt 0) { Write-Host "SKIP (no shebang): $($e.Name)"; $skipped++; continue }
    $oldShebang = [System.Text.Encoding]::ASCII.GetString($bytes, $shAt, $zipAt - $shAt)
    $head = New-Object byte[] $shAt
    [Array]::Copy($bytes, 0, $head, 0, $shAt)
    $tailLen = $bytes.Length - $zipAt
    $tail = New-Object byte[] $tailLen
    [Array]::Copy($bytes, $zipAt, $tail, 0, $tailLen)
    $out = New-Object byte[] ($head.Length + $newBytes.Length + $tail.Length)
    [Array]::Copy($head, 0, $out, 0, $head.Length)
    [Array]::Copy($newBytes, 0, $out, $head.Length, $newBytes.Length)
    [Array]::Copy($tail, 0, $out, $head.Length + $newBytes.Length, $tail.Length)
    [System.IO.File]::WriteAllBytes($e.FullName, $out)
    Write-Host "FIX  $($e.Name): $($oldShebang.Trim())"
    $fixed++
}
Write-Host ""
Write-Host "Done. Fixed: $fixed  Skipped: $skipped"
