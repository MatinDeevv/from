$ErrorActionPreference = "Continue"
$Python = "C:\Users\marti\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
$Root = "C:\Users\marti\from"
$Work = "C:\Users\marti\from\from"
$Items = @(
  @{Symbol="XAGUSD"; Scale="1000"},
  @{Symbol="EURUSD"; Scale="100000"},
  @{Symbol="USDJPY"; Scale="1000"},
  @{Symbol="GBPUSD"; Scale="100000"},
  @{Symbol="USDCHF"; Scale="100000"},
  @{Symbol="AUDUSD"; Scale="100000"},
  @{Symbol="NZDUSD"; Scale="100000"},
  @{Symbol="USDCAD"; Scale="100000"},
  @{Symbol="EURGBP"; Scale="100000"},
  @{Symbol="EURJPY"; Scale="1000"},
  @{Symbol="GBPJPY"; Scale="1000"}
)

Set-Location $Work
foreach ($Item in $Items) {
  $Symbol = $Item.Symbol
  $Scale = $Item.Scale
  $Out = Join-Path $Root "data\derived\leader_${Symbol}_5m_20230101_20260521.parquet"
  $Report = Join-Path $Root "reports\leader_${Symbol}_5m_20230101_20260521_redownload_report.json"
  $Log = Join-Path $Root "reports\leader_${Symbol}_5m_20230101_20260521_redownload.log"
  $Args = @(
    "tools/fetch_dukascopy_5m.py",
    "--instrument", $Symbol,
    "--start", "2023-01-01",
    "--end", "2026-05-22",
    "--out", $Out,
    "--report", $Report,
    "--price-scale", $Scale,
    "--workers", "3"
  )
  "Starting $Symbol at $(Get-Date -Format o)" | Out-File -FilePath $Log -Encoding utf8
  & $Python @Args *>> $Log
  "Finished $Symbol at $(Get-Date -Format o)" | Out-File -FilePath $Log -Append -Encoding utf8
}
