$ErrorActionPreference = 'Stop'
$workspace = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path.TrimEnd('\')
$trackedNames = @(& git -C $workspace ls-files)
$tracked = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
foreach ($name in $trackedNames) { [void]$tracked.Add($name.Replace('/', '\')) }
$candidates = [System.Collections.Generic.List[object]]::new()
$directories = [System.Collections.Generic.Stack[string]]::new()
$directories.Push($workspace)
while ($directories.Count -gt 0) {
    $currentDirectory = $directories.Pop()
    foreach ($item in Get-ChildItem -LiteralPath $currentDirectory -Force) {
        if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) { continue }
        if ($item.PSIsContainer) {
            if ($item.Name -notin @('.git', '.venv', 'venv', '.idea', 'node_modules')) {
                $directories.Push($item.FullName)
            }
            continue
        }
        if ($item.Extension -notin @('.pyc', '.pyo')) { continue }
        $absolute = [IO.Path]::GetFullPath($item.FullName)
        if (-not $absolute.StartsWith($workspace + '\', [StringComparison]::OrdinalIgnoreCase)) {
            throw "Cleanup path escaped workspace: $absolute"
        }
        $relative = $absolute.Substring($workspace.Length + 1)
        if ($tracked.Contains($relative)) { continue }
        $candidates.Add([PSCustomObject]@{ path = $absolute; relative = $relative; bytes = $item.Length; sha256 = (Get-FileHash -LiteralPath $absolute -Algorithm SHA256).Hash })
    }
}
$manifestPath = Join-Path $PSScriptRoot 'cleanup_manifest.json'
if (Test-Path -LiteralPath $manifestPath) { throw 'Cleanup receipt exists; refusing to overwrite' }
$receipt = [ordered]@{ scope = 'Untracked regenerable Python bytecode outside virtual environments, Git, IDE and node_modules only'; deleted_files = @(); bytes_removed = 0; excluded = @('historical source', 'weights', 'datasets', 'results', 'logs', 'snapshots', 'archives', 'virtual environment', 'Git objects') }
foreach ($candidate in $candidates) {
    $absolute = [IO.Path]::GetFullPath($candidate.path)
    if (-not $absolute.StartsWith($workspace + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Unsafe target' }
    if ((Get-FileHash -LiteralPath $absolute -Algorithm SHA256).Hash -ne $candidate.sha256) { throw 'Cache changed during cleanup' }
    Remove-Item -LiteralPath $absolute -Force
    $receipt.deleted_files += $candidate
    $receipt.bytes_removed += $candidate.bytes
}
# Only remove empty bytecode directories, each checked inside this workspace.
$emptyDirectories = @(Get-ChildItem -LiteralPath $workspace -Directory -Filter '__pycache__' -Recurse -ErrorAction SilentlyContinue | Where-Object { $_.FullName -notlike '*\.venv\*' -and $_.FullName -notlike '*\.git\*' })
foreach ($directory in $emptyDirectories) {
    $absolute = [IO.Path]::GetFullPath($directory.FullName)
    if (-not $absolute.StartsWith($workspace + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Unsafe directory' }
    if (@(Get-ChildItem -LiteralPath $absolute -Force).Count -eq 0) { Remove-Item -LiteralPath $absolute }
}
$receipt | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $manifestPath -Encoding utf8
[PSCustomObject]@{ deleted_cache_files = $receipt.deleted_files.Count; bytes_removed = $receipt.bytes_removed } | ConvertTo-Json
