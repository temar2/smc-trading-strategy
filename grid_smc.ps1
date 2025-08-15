# Grid Search SMC — GBPUSD M15/4H
# Sauve les résultats dans results_grid.csv + best_params.txt

$ErrorActionPreference = "Stop"

# ===== Constantes (adapte si besoin) =====
$PY = "python"
$SCRIPT = "smc_backtest_v2.5.py"
$API = "8af42105d7754290bc090dfb3a6ca6d4"
$SYMBOL = "GBP/USD"
$LTF = "15min"
$HTF = "4h"
$START = "2025-07-16"
$END = "2025-08-16"
$CAPITAL = 100000
$RISK = 0.0045
$SPREAD = 0.2
$SLIP_IN = 0.1
$SLIP_OUT = 0.1
$COMMM = 7
$LEV = 30
$LOT_MIN = 1000
$LOT_STEP = 1000
$MAX_LOT = 2.0
$ATR_PER = 14
$ATR_ALPHA = 0.4
$MIN_STOP = 3.0
$HARD_MIN_STOP = 3.0
$BUF_SL = 0.25
$PARTIAL_R = 1.0
$BE_R = 0.7
$TRAIL_WIN = 3
$TTL = 60
$FALLBACK = 24
$PULLBACK_BARS = 2
$NEWS_MIN = 15
$SESSION_START = 6
$SESSION_END = 20

# ===== Grilles à tester =====
$RR_TARGETS = @(1.3, 1.4, 1.5)
$RR_ALT     = @(1.1, 1.2)
$ATR_MINs   = @(0.9, 1.0, 1.2)
$MOMs       = @(0.40, 0.45, 0.50)
$FORBIDDENs = @(
  "0,1,2,3,4,5,6,7,12,20,21,22,23",           # focus 08–19
  "0,1,2,3,4,5,6,7,12,19,20,21,22,23",        # focus 08–18
  "0,1,2,3,4,5,6,7,12,18,19,20,21,22,23"      # focus 08–17
)

$results = @()

# ===== Boucle grid =====
foreach ($rr in $RR_TARGETS) {
  foreach ($rrAlt in $RR_ALT) {
    foreach ($atrMin in $ATR_MINs) {
      foreach ($mom in $MOMs) {
        foreach ($forb in $FORBIDDENs) {

          $cmd = @(
            $PY, $SCRIPT,
            "--api-key", $API,
            "--symbol", "$SYMBOL",
            "--ltf", $LTF, "--htf", $HTF,
            "--start", $START, "--end", $END,
            "--capital", $CAPITAL, "--risk_per_trade", $RISK,
            "--spread_pips", $SPREAD, "--slip_entry_pips", $SLIP_IN, "--slip_exit_pips", $SLIP_OUT,
            "--commission_per_million", $COMMM,
            "--leverage_max", $LEV, "--lot_min", $LOT_MIN, "--lot_step", $LOT_STEP, "--max_lot", $MAX_LOT,
            "--atr_period", $ATR_PER, "--atr_alpha", $ATR_ALPHA,
            "--auto-bias",
            "--require_confluence", "--confluence_min", 2,
            "--session_start_utc", $SESSION_START, "--session_end_utc", $SESSION_END,
            "--forbidden_hours_utc", $forb,
            "--atr_min_pips", $atrMin,
            "--min_stop_pips", $MIN_STOP, "--hard_min_stop_pips", $HARD_MIN_STOP,
            "--buffer_sl_pips", $BUF_SL,
            "--rr_target", $rr, "--rr_target_alt", $rrAlt, "--atr_rr_switch_pips", 9,
            "--use_partials", "--partial_take_r", $PARTIAL_R, "--move_be_r", $BE_R,
            "--trail_struct_window", $TRAIL_WIN,
            "--pending_ttl_bars", $TTL, "--fallback_market_bars", $FALLBACK,
            "--pullback_bars", $PULLBACK_BARS, "--pullback_optional",
            "--momentum_min_body_atr", $mom,
            "--news-free-minutes", $NEWS_MIN
          )

          Write-Host ("==> Test rr={0} alt={1} atrMin={2} mom={3} forb=[{4}]" -f $rr,$rrAlt,$atrMin,$mom,$forb)

          $out = & $cmd 2>&1 | Out-String

          # Parsers
          $fe   = [regex]::Match($out, "final_equity:\s*([0-9]+\.[0-9]+)").Groups[1].Value
          $ret  = [regex]::Match($out, "return_pct:\s*([+-]?[0-9]+\.[0-9]+)%").Groups[1].Value
          $tr   = [regex]::Match($out, "trades:\s*([0-9]+)").Groups[1].Value
          $avgR = [regex]::Match($out, "avg_R:\s*([+-]?[0-9]+\.[0-9]+)").Groups[1].Value
          $dd   = [regex]::Match($out, "max_drawdown_pct:\s*([0-9]+\.[0-9]+)%").Groups[1].Value

          if (-not $fe) { $fe = "0" }
          if (-not $ret) { $ret = "0" }
          if (-not $tr) { $tr = "0" }
          if (-not $avgR) { $avgR = "0" }
          if (-not $dd) { $dd = "0" }

          $row = [pscustomobject]@{
            rr_target              = $rr
            rr_target_alt          = $rrAlt
            atr_min_pips           = $atrMin
            momentum_min_body_atr  = $mom
            forbidden_hours_utc    = $forb
            final_equity           = [double]$fe
            return_pct             = [double]$ret
            trades                 = [int]$tr
            avg_R                  = [double]$avgR
            max_drawdown_pct       = [double]$dd
            raw_log_short          = ($out -split "`n" | Select-String "RÉSULTATS|JOURNAL DES TRADES" -Context 0,6 | ForEach-Object { $_.ToString() }) -join " "
          }

          $results += $row
        }
      }
    }
  }
}

# Export CSV
$csvPath = Join-Path (Get-Location) "results_grid.csv"
$results | Export-Csv -NoTypeInformation -Path $csvPath
Write-Host "💾 Résultats: $csvPath"

# Sélection du "meilleur" (règles simples)
# 1) DD <= 1.5%
# 2) 6 <= trades <= 30
# 3) avg_R >= 0.10
# Classement par return_pct desc, puis avg_R desc, puis DD asc
$filtered = $results | Where-Object {
  $_.max_drawdown_pct -le 1.5 -and
  $_.trades -ge 6 -and $_.trades -le 30 -and
  $_.avg_R -ge 0.10
}

if ($filtered.Count -eq 0) {
  Write-Host "⚠️ Aucun set ne respecte tous les critères. Affichage du top 5 par return_pct / avg_R / DD…"
  $top = $results | Sort-Object `
    @{Expression="return_pct";Descending=$true}, `
    @{Expression="avg_R";Descending=$true}, `
    @{Expression="max_drawdown_pct";Descending=$false} | Select-Object -First 5
} else {
  $top = $filtered | Sort-Object `
    @{Expression="return_pct";Descending=$true}, `
    @{Expression="avg_R";Descending=$true}, `
    @{Expression="max_drawdown_pct";Descending=$false} | Select-Object -First 5
}

$best = $top | Select-Object -First 1
$bestPath = Join-Path (Get-Location) "best_params.txt"

"BEST PARAMS:
rr_target=$($best.rr_target)
rr_target_alt=$($best.rr_target_alt)
atr_min_pips=$($best.atr_min_pips)
momentum_min_body_atr=$($best.momentum_min_body_atr)
forbidden_hours_utc=$($best.forbidden_hours_utc)

Metrics:
return_pct=$($best.return_pct)%
trades=$($best.trades)
avg_R=$($best.avg_R)
max_drawdown_pct=$($best.max_drawdown_pct)%
final_equity=$($best.final_equity)
" | Set-Content -Path $bestPath -Encoding UTF8

Write-Host "🏆 Best params → $bestPath"
$best
