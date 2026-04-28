param(
  [Parameter(Mandatory=$true)]
  [string]$ProjectDir,

  [string]$Entry = "main.tex",

  [int]$Runs = 2
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$xelatex = Join-Path $repoRoot "TinyTeX\bin\windows\xelatex.exe"

if (!(Test-Path $xelatex)) {
  throw "Local TinyTeX not found: $xelatex"
}

$projectPath = Resolve-Path $ProjectDir
Push-Location $projectPath
try {
  for ($i = 1; $i -le $Runs; $i++) {
    & $xelatex -interaction=nonstopmode -halt-on-error $Entry
    if ($LASTEXITCODE -ne 0) {
      throw "xelatex failed on run $i"
    }
  }
}
finally {
  Pop-Location
}
