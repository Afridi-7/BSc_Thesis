param(
    [string]$Path = (Join-Path $PSScriptRoot '..\README.md')
)

$Path = (Resolve-Path $Path).Path
$utf8 = New-Object System.Text.UTF8Encoding($false)
$content = [System.IO.File]::ReadAllText($Path, $utf8)
$before = $content.Length

# Multiplication sign in "~10x"
$content = $content.Replace([char]0xEF + [char]0xBF + [char]0xBD, '-')   # naive fallback (used if all else fails)

# Reload original (we want targeted replacements first, then a generic fallback).
$content = [System.IO.File]::ReadAllText($Path, $utf8)

$mojibake = "$([char]0xC3)$([char]0xAF)$([char]0xC2)$([char]0xBF)$([char]0xC2)$([char]0xBD)"

# Targeted patterns (apply BEFORE generic fallback)
$content = $content -replace ([regex]::Escape("~10$mojibake")), '~10x'
$content = $content -replace ("(\d)$([regex]::Escape($mojibake))(\d)"), '$1-$2'
$content = $content -replace ("\|\s*$([regex]::Escape($mojibake))\s*\|"), '| - |'
$content = $content -replace (" $([regex]::Escape($mojibake)) "), ' - '

# Generic fallback for any remaining instances
$content = $content.Replace($mojibake, '-')

# Arrow placeholders that earlier corruption left as literal '?'
$content = $content -replace 'Stage 1 \? 2 \? 3', 'Stage 1 -> 2 -> 3'
$content = $content -replace 'Thought \? Action \? Observation', 'Thought -> Action -> Observation'
$content = $content -replace '/api \? http', '/api -> http'
$content = $content -replace 'adapter \? merged', 'adapter -> merged'
$content = $content -replace 'Deploy \? Inference', 'Deploy -> Inference'
$content = $content -replace 'dashboard \? click', 'dashboard -> click'
$content = $content -replace 'bucket \? \{', 'bucket -> {'
$content = $content -replace '(\d{2,4})\s+pages\s+\?\s+(\d+)\s+chunks', '$1 pages -> $2 chunks'
$content = $content -replace '(`[^`]+`)\s+\?\s+(`)', '$1 -> $2'
$content = $content -replace '(\.yaml)\s+\?\s+(`)', '$1 -> $2'
$content = $content -replace 'Edit `config\.yaml` \? `', 'Edit `config.yaml` -> `'
$content = $content -replace '(\d+)\s+chunks\s+\?\s+MiniLM', '$1 chunks - MiniLM'

[System.IO.File]::WriteAllText($Path, $content, $utf8)

$after = [System.IO.File]::ReadAllText($Path, $utf8)
$remaining = ([regex]::Matches($after, [regex]::Escape($mojibake))).Count
Write-Host ("Bytes before: {0}  after: {1}" -f $before, $after.Length)
Write-Host ("Remaining mojibake sequences: {0}" -f $remaining)
