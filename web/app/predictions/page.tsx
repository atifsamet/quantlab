"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { RefreshCw } from "lucide-react";
import { api, HistoryRecord, Prediction } from "@/lib/api";
import { EmptyState, LoadingState, OutcomeBadge, PageHeader, Panel, PredictionBadge } from "@/components/ui";

const SYMBOLS = ["BTC-USDT", "ETH-USDT", "SOL-USDT", "XRP-USDT"];
const TFS = ["15m", "1h", "4h"];
const AUTO_OPTIONS = [
  { label: "Off", ms: 0 },
  { label: "15m", ms: 15 * 60 * 1000 },
  { label: "1h", ms: 60 * 60 * 1000 },
  { label: "4h", ms: 4 * 60 * 60 * 1000 },
];

function outcomeClass(outcome: string | null, evaluated: boolean) {
  if (!evaluated) return "text-slate-400";
  if (outcome === "WIN") return "text-long";
  if (outcome === "LOSS") return "text-short";
  if (outcome === "TIMEOUT") return "text-amber-300";
  return "text-slate-400";
}

function StatusPill({ label, tone }: { label: string; tone: "live" | "stale" | "error" | "neutral" }) {
  const cls =
    tone === "live"
      ? "border-long/40 bg-long/10 text-long"
      : tone === "stale"
        ? "border-amber-700/50 bg-amber-950/40 text-amber-300"
        : tone === "error"
          ? "border-red-800/50 bg-red-950/40 text-red-300"
          : "border-ink-600 bg-ink-800 text-slate-400";
  return (
    <span className={`rounded border px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide ${cls}`}>
      {label}
    </span>
  );
}

export default function PredictionsPage() {
  const [symbol, setSymbol] = useState("BTC-USDT");
  const [tf, setTf] = useState("4h");
  const [pred, setPred] = useState<Prediction | null>(null);
  const [market, setMarket] = useState<any>(null);
  const [history, setHistory] = useState<HistoryRecord[]>([]);
  const [pending, setPending] = useState<HistoryRecord[]>([]);
  const [histCtx, setHistCtx] = useState<any>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [detail, setDetail] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [autoMs, setAutoMs] = useState(0);
  const [staleLabel, setStaleLabel] = useState(false);
  const inFlight = useRef(false);

  const loadHistory = useCallback(async () => {
    try {
      const [h, p, st] = await Promise.all([
        api.predictionHistory({ symbol, timeframe: tf, limit: 40 }),
        api.predictionHistory({ symbol, timeframe: tf, evaluated: "false", limit: 20 }),
        api.predictionStats(symbol, tf),
      ]);
      setHistory(h.records || []);
      setPending(p.records || []);
      setHistCtx(st?.overall || null);
    } catch {
      setHistory([]);
      setPending([]);
    }
  }, [symbol, tf]);

  const load = useCallback(
    async (refresh = false) => {
      if (inFlight.current) return;
      inFlight.current = true;
      setLoading(true);
      setError(null);
      try {
        if (refresh) {
          const payload = await api.refreshPrediction(symbol, tf);
          setPred(payload.prediction);
          setHistCtx(payload.historical_context || null);
          setPending(payload.pending_predictions || []);
          setStaleLabel(false);
        } else {
          const p = await api.prediction(symbol, tf, false);
          setPred(p);
          setStaleLabel(false);
        }
        const m = await api.market(symbol, tf);
        setMarket(m);
        await loadHistory();
      } catch (e: any) {
        setError(e.message || "Failed to load prediction");
        setStaleLabel(true);
      } finally {
        setLoading(false);
        inFlight.current = false;
      }
    },
    [symbol, tf, loadHistory]
  );

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    if (params.get("symbol")) setSymbol(params.get("symbol")!);
    if (params.get("tf")) setTf(params.get("tf")!);
  }, []);

  useEffect(() => {
    load(false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [symbol, tf]);

  useEffect(() => {
    if (!autoMs) return;
    const id = window.setInterval(() => {
      if (!inFlight.current) load(true);
    }, autoMs);
    return () => window.clearInterval(id);
  }, [autoMs, load]);

  useEffect(() => {
    if (!selectedId) {
      setDetail(null);
      return;
    }
    api.predictionDetail(selectedId).then(setDetail).catch(() => setDetail(null));
  }, [selectedId]);

  const chartData = useMemo(() => {
    return (market?.candles || [])
      .filter((c: any) => c.ema_20 != null)
      .map((c: any) => ({
        t: c.timestamp.slice(5, 16),
        close: c.close,
        ema20: c.ema_20,
        ema50: c.ema_50,
        ema200: c.ema_200,
      }));
  }, [market]);

  const detailChart = useMemo(() => {
    if (!detail) return [];
    const hist = (detail.chart_history || []).map((p: any) => ({
      t: String(p.timestamp).slice(5, 16),
      close: p.close,
    }));
    const fut = (detail.chart_future || []).map((p: any) => ({
      t: String(p.timestamp).slice(5, 16),
      close: p.close,
      future: p.close,
    }));
    return [...hist, ...fut];
  }, [detail]);

  const marketStatus = pred?.live?.market_status || market?.live?.market_status || (error ? "ERROR" : "—");
  const predStatus = staleLabel ? "STALE" : pred?.live?.prediction_status || "—";

  return (
    <div className="space-y-6">
      <PageHeader
        eyebrow="Live Market Analysis"
        title="Predictions"
        description="Closed-candle analytical signals with historical context — not trade execution."
        actions={
          <div className="flex flex-wrap items-center gap-2">
            <select
              className="rounded border border-ink-600 bg-ink-800 px-3 py-2 text-sm"
              value={symbol}
              onChange={(e) => setSymbol(e.target.value)}
              aria-label="Asset"
            >
              {SYMBOLS.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
            <select
              className="rounded border border-ink-600 bg-ink-800 px-3 py-2 text-sm"
              value={tf}
              onChange={(e) => setTf(e.target.value)}
              aria-label="Timeframe"
            >
              {TFS.map((t) => (
                <option key={t} value={t}>
                  {t}
                </option>
              ))}
            </select>
            <select
              className="rounded border border-ink-600 bg-ink-800 px-3 py-2 text-sm"
              value={autoMs}
              onChange={(e) => setAutoMs(Number(e.target.value))}
              aria-label="Auto refresh interval"
            >
              {AUTO_OPTIONS.map((o) => (
                <option key={o.label} value={o.ms}>
                  Auto: {o.label}
                </option>
              ))}
            </select>
            <button
              type="button"
              onClick={() => load(true)}
              disabled={loading}
              className="inline-flex items-center gap-2 rounded border border-accent/40 bg-accent/10 px-3 py-2 text-sm text-accent hover:bg-accent/20 disabled:opacity-50"
              aria-busy={loading}
            >
              <RefreshCw size={14} className={loading ? "animate-spin" : ""} aria-hidden />
              {loading ? "Analyzing market…" : "Refresh Analysis"}
            </button>
          </div>
        }
      />

      <div className="flex flex-wrap items-center gap-2 text-xs">
        <StatusPill
          label={`Market ${marketStatus}`}
          tone={marketStatus === "LIVE" ? "live" : marketStatus === "STALE" ? "stale" : marketStatus === "ERROR" ? "error" : "neutral"}
        />
        <StatusPill label={`Prediction ${predStatus}`} tone={predStatus === "CURRENT" ? "live" : predStatus === "STALE" ? "stale" : "neutral"} />
        {pred?.live?.last_market_update && (
          <span className="text-slate-500">Last market update: {String(pred.live.last_market_update).replace("T", " ").slice(0, 19)} UTC</span>
        )}
        {pred?.timestamp && (
          <span className="text-slate-500">Prediction candle: {String(pred.timestamp).replace("T", " ").slice(0, 19)} UTC</span>
        )}
      </div>

      {loading && !pred && <LoadingState label="Analyzing market…" />}
      {error && (
        <EmptyState
          title="Market data unavailable"
          detail={`${error}. Last valid prediction is kept and labeled STALE when possible.`}
        />
      )}

      {pred && (
        <>
          <div className="grid gap-4 lg:grid-cols-3">
            <Panel title={`${pred.symbol} · ${pred.timeframe.toUpperCase()}`}>
              <div className="flex items-center gap-3">
                <PredictionBadge value={pred.prediction} />
                <span className="text-xs capitalize text-slate-500">{pred.trend}</span>
              </div>
              <div className="mt-4 font-mono text-3xl text-white">
                ${pred.current_price.toLocaleString(undefined, { maximumFractionDigits: 2 })}
              </div>
              <div className="mt-4" title={pred.signal_strength_tooltip || "Analytical score, not probability"}>
                <div className="text-xs uppercase tracking-wide text-slate-500">Signal strength</div>
                <div className="mt-1 font-mono text-2xl text-accent">{pred.signal_strength} / 100</div>
                <p className="mt-1 text-[11px] text-slate-500">
                  Analytical signal strength — not a probability of future return
                </p>
              </div>
            </Panel>

            <Panel title="Model levels">
              <dl className="space-y-3 text-sm">
                <div className="flex justify-between gap-4">
                  <dt className="text-slate-500">Entry zone</dt>
                  <dd className="font-mono text-slate-200">
                    {pred.levels.entry_low != null && pred.levels.entry_high != null
                      ? `$${pred.levels.entry_low.toFixed(2)} – $${pred.levels.entry_high.toFixed(2)}`
                      : "—"}
                  </dd>
                </div>
                <div className="flex justify-between gap-4">
                  <dt className="text-slate-500">Stop loss</dt>
                  <dd className="font-mono text-short">{pred.levels.stop_loss != null ? `$${pred.levels.stop_loss.toFixed(2)}` : "—"}</dd>
                </div>
                <div className="flex justify-between gap-4">
                  <dt className="text-slate-500">Take profit</dt>
                  <dd className="font-mono text-long">{pred.levels.take_profit != null ? `$${pred.levels.take_profit.toFixed(2)}` : "—"}</dd>
                </div>
                <div className="flex justify-between gap-4">
                  <dt className="text-slate-500">Risk / reward</dt>
                  <dd className="font-mono">{pred.levels.risk_reward ?? "—"}</dd>
                </div>
              </dl>
            </Panel>

            <Panel title="Technical context">
              <div className="grid grid-cols-2 gap-3 text-xs">
                {[
                  ["EMA20", pred.indicators.ema_20],
                  ["EMA50", pred.indicators.ema_50],
                  ["EMA200", pred.indicators.ema_200],
                  ["RSI", pred.indicators.rsi_14],
                  ["MACD hist", pred.indicators.macd_hist],
                  ["ATR", pred.indicators.atr_14],
                  ["Volume", pred.indicators.volume],
                  ["Vol ratio", pred.indicators.volume_ratio],
                ].map(([k, v]) => (
                  <div key={String(k)} className="rounded border border-ink-600 bg-ink-900/50 px-2 py-2">
                    <div className="text-slate-500">{k}</div>
                    <div className="mt-0.5 font-mono text-slate-200">{v == null ? "—" : Number(v).toFixed(3)}</div>
                  </div>
                ))}
              </div>
            </Panel>
          </div>

          <Panel title="Price & EMAs">
            {chartData.length === 0 ? (
              <EmptyState title="No chartable candles" />
            ) : (
              <div className="h-64 w-full">
                <ResponsiveContainer>
                  <LineChart data={chartData}>
                    <CartesianGrid stroke="#1c2430" strokeDasharray="3 3" />
                    <XAxis dataKey="t" tick={{ fill: "#64748b", fontSize: 10 }} minTickGap={30} />
                    <YAxis domain={["auto", "auto"]} tick={{ fill: "#64748b", fontSize: 10 }} width={70} />
                    <Tooltip contentStyle={{ background: "#151b24", border: "1px solid #243041" }} />
                    <Legend />
                    {pred.levels.entry_low != null && <ReferenceLine y={pred.current_price} stroke="#94a3b8" strokeDasharray="4 4" />}
                    {pred.levels.stop_loss != null && <ReferenceLine y={pred.levels.stop_loss} stroke="#e74c3c" strokeDasharray="3 3" />}
                    {pred.levels.take_profit != null && <ReferenceLine y={pred.levels.take_profit} stroke="#2ecc71" strokeDasharray="3 3" />}
                    <Line type="monotone" dataKey="close" stroke="#e2e8f0" dot={false} strokeWidth={1.5} />
                    <Line type="monotone" dataKey="ema20" stroke="#3d8bfd" dot={false} strokeWidth={1} />
                    <Line type="monotone" dataKey="ema50" stroke="#f59e0b" dot={false} strokeWidth={1} />
                    <Line type="monotone" dataKey="ema200" stroke="#a78bfa" dot={false} strokeWidth={1} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            )}
          </Panel>

          <div className="grid gap-4 lg:grid-cols-2">
            <Panel title="Why this signal?">
              <div className="grid gap-4 sm:grid-cols-2">
                <div>
                  <h3 className="text-xs font-semibold uppercase text-long">Positive</h3>
                  <ul className="mt-2 space-y-2 text-xs text-slate-400">
                    {pred.reasons.positive.map((r) => <li key={r}>• {r}</li>)}
                  </ul>
                </div>
                <div>
                  <h3 className="text-xs font-semibold uppercase text-short">Negative</h3>
                  <ul className="mt-2 space-y-2 text-xs text-slate-400">
                    {pred.reasons.negative.map((r) => <li key={r}>• {r}</li>)}
                  </ul>
                </div>
              </div>
            </Panel>
            <Panel title="Historical context">
              <div className="mb-2 text-[10px] font-semibold uppercase tracking-wide text-amber-400">Historical performance</div>
              <p className="mb-3 text-[11px] text-slate-500">Past outcomes for this asset/timeframe — not future probability.</p>
              {histCtx ? (
                <dl className="grid grid-cols-2 gap-3 text-xs">
                  <div><div className="text-slate-500">Sample</div><div className="font-mono">{histCtx.scored_n ?? histCtx.sample ?? "—"}</div></div>
                  <div><div className="text-slate-500">WIN rate</div><div className="font-mono">{histCtx.win_rate == null ? "—" : `${histCtx.win_rate}%`}</div></div>
                  <div><div className="text-slate-500">LOSS rate</div><div className="font-mono">{histCtx.loss_rate == null ? "—" : `${histCtx.loss_rate}%`}</div></div>
                  <div><div className="text-slate-500">TIMEOUT rate</div><div className="font-mono">{histCtx.timeout_rate == null ? "—" : `${histCtx.timeout_rate}%`}</div></div>
                </dl>
              ) : (
                <EmptyState title="No historical sample for this pair yet" />
              )}
              <Link href="/prediction-performance" className="mt-3 inline-block text-xs text-accent hover:underline">Full performance →</Link>
            </Panel>
          </div>

          <Panel title="Research status">
            <p className="font-mono text-sm uppercase text-amber-300">{pred.research_conclusion}</p>
            <p className="mt-2 text-xs text-slate-500">Historical testing has not demonstrated a durable predictive edge.</p>
            <p className="mt-3 text-[11px] text-slate-500">{pred.disclaimer}</p>
          </Panel>
        </>
      )}

      <Panel title="Pending predictions">
        {pending.length === 0 ? (
          <EmptyState title="No pending predictions" detail="Recent snapshots will appear here until evaluated." />
        ) : (
          <ul className="space-y-2 text-xs">
            {pending.slice(0, 12).map((r) => (
              <li key={r.id} className="flex flex-wrap items-center justify-between gap-2 border-b border-ink-700 py-2">
                <span className="font-mono text-slate-300">{r.symbol} · {r.timeframe} · {r.prediction} · {r.signal_strength}/100</span>
                <span className="text-slate-500">Waiting for evaluation</span>
              </li>
            ))}
          </ul>
        )}
      </Panel>

      <Panel title="Prediction history">
        {history.length === 0 ? (
          <EmptyState title="No stored snapshots yet" detail="Press Refresh Analysis to save a live snapshot." />
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[800px] text-left text-xs">
              <thead className="text-slate-500">
                <tr>
                  {["Timestamp", "Asset", "TF", "Prediction", "Strength", "Entry", "Outcome", "Return", "Status"].map((h) => (
                    <th key={h} className="border-b border-ink-600 px-2 py-2 font-medium">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {history.map((r) => (
                  <tr key={r.id} className="cursor-pointer border-b border-ink-700/70 hover:bg-ink-700/40" onClick={() => setSelectedId(r.id)}>
                    <td className="px-2 py-2 font-mono text-slate-400">{String(r.candle_timestamp).slice(0, 16).replace("T", " ")}</td>
                    <td className="px-2 py-2 font-mono">{r.symbol.replace("-USDT", "")}</td>
                    <td className="px-2 py-2">{r.timeframe}</td>
                    <td className="px-2 py-2"><PredictionBadge value={r.prediction} /></td>
                    <td className="px-2 py-2 font-mono">{r.signal_strength}</td>
                    <td className="px-2 py-2 font-mono">${Number(r.entry_price).toFixed(2)}</td>
                    <td className="px-2 py-2">
                      <OutcomeBadge outcome={r.outcome} evaluated={r.evaluated} />
                    </td>
                    <td className={`px-2 py-2 font-mono ${outcomeClass(r.outcome, r.evaluated)}`}>{r.return_pct == null ? "—" : `${Number(r.return_pct).toFixed(2)}%`}</td>
                    <td className="px-2 py-2 text-slate-500">{r.evaluated ? "Evaluated" : "Pending"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Panel>

      {detail?.record && (
        <Panel title="Prediction detail" action={<button type="button" className="text-xs text-slate-500" onClick={() => setSelectedId(null)}>Close</button>}>
          <div className="grid gap-4 lg:grid-cols-2">
            <div className="space-y-2 text-xs text-slate-400">
              <div><PredictionBadge value={detail.record.prediction} /> <span className="font-mono text-slate-300">{detail.record.symbol}</span></div>
              <div>Outcome: {detail.record.evaluated ? detail.record.outcome : "Pending"}</div>
              <p className="text-[11px] text-amber-200/70">{detail.note}</p>
            </div>
            <div className="h-56">
              <ResponsiveContainer>
                <LineChart data={detailChart}>
                  <CartesianGrid stroke="#1c2430" strokeDasharray="3 3" />
                  <XAxis dataKey="t" tick={{ fill: "#64748b", fontSize: 10 }} minTickGap={28} />
                  <YAxis domain={["auto", "auto"]} tick={{ fill: "#64748b", fontSize: 10 }} width={60} />
                  <Tooltip contentStyle={{ background: "#151b24", border: "1px solid #243041" }} />
                  <Line type="monotone" dataKey="close" stroke="#e2e8f0" dot={false} />
                  <Line type="monotone" dataKey="future" stroke="#f59e0b" strokeDasharray="4 2" dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </Panel>
      )}

      <p className="text-[11px] text-slate-600">
        Market data is retrieved from public exchange endpoints and may be delayed or temporarily unavailable.
      </p>
    </div>
  );
}
