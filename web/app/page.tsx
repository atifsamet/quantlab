"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api } from "@/lib/api";
import { EmptyState, LoadingState, Panel, PredictionBadge } from "@/components/ui";

export default function OverviewPage() {
  const [data, setData] = useState<any>(null);
  const [analytics, setAnalytics] = useState<any>(null);
  const [market, setMarket] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      api.overview("4h"),
      api.analytics().catch(() => null),
      api.market("BTC-USDT", "4h").catch(() => null),
    ])
      .then(([ov, an, mkt]) => {
        setData(ov);
        setAnalytics(an);
        setMarket(mkt);
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  const chartData = useMemo(() => {
    return (market?.candles || [])
      .filter((c: any) => c.ema_20 != null)
      .slice(-80)
      .map((c: any) => ({
        t: String(c.timestamp).slice(5, 16),
        close: c.close,
        ema20: c.ema_20,
        ema50: c.ema_50,
      }));
  }, [market]);

  const perf = data?.prediction_performance;

  return (
    <div className="space-y-8">
      <section className="rounded-lg border border-ink-600 bg-ink-900/50 p-6 sm:p-8">
        <div className="font-mono text-xs uppercase tracking-[0.22em] text-accent">QuantLab</div>
        <h1 className="mt-3 text-3xl font-semibold tracking-tight text-white sm:text-4xl">
          Quantitative Crypto Market Intelligence
        </h1>
        <p className="mt-3 max-w-2xl text-sm leading-relaxed text-slate-400">
          Real-time market analysis. Historical prediction evaluation. Quantitative research —
          without trade execution.
        </p>
        <div className="mt-5 flex flex-wrap gap-2 text-[11px]">
          <span className="rounded border border-amber-700/60 bg-amber-950/40 px-2.5 py-1 font-semibold uppercase tracking-wide text-amber-300">
            Research Mode
          </span>
          <span className="rounded border border-red-800/50 bg-red-950/40 px-2.5 py-1 font-semibold uppercase tracking-wide text-red-300">
            Live Trading Disabled
          </span>
        </div>
        <dl className="mt-6 grid gap-3 text-xs text-slate-500 sm:grid-cols-3">
          <div className="rounded border border-ink-600 bg-ink-800/40 p-3">
            <dt className="uppercase tracking-wide text-slate-600">Data</dt>
            <dd className="mt-1 text-slate-300">Public OKX market candles</dd>
          </div>
          <div className="rounded border border-ink-600 bg-ink-800/40 p-3">
            <dt className="uppercase tracking-wide text-slate-600">Outputs</dt>
            <dd className="mt-1 text-slate-300">LONG / SHORT / NEUTRAL + signal strength</dd>
          </div>
          <div className="rounded border border-ink-600 bg-ink-800/40 p-3">
            <dt className="uppercase tracking-wide text-slate-600">Finding</dt>
            <dd className="mt-1 font-mono text-amber-300">NO ROBUST EDGE DETECTED</dd>
          </div>
        </dl>
      </section>

      {loading && <LoadingState label="Loading market overview…" />}
      {error && (
        <EmptyState
          title="API unavailable"
          detail={`${error}. Start the backend with uvicorn, then refresh this page.`}
        />
      )}

      {data && (
        <>
          <section>
            <div className="mb-3 flex items-end justify-between gap-3">
              <div>
                <h2 className="text-sm font-semibold text-slate-200">Market overview</h2>
                <p className="mt-0.5 text-xs text-slate-500">Closed-candle predictions · 4H default</p>
              </div>
              <Link href="/predictions" className="text-xs text-accent hover:underline">
                Open live analysis →
              </Link>
            </div>
            <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
              {data.cards?.map((card: any) => (
                <Link
                  key={card.symbol}
                  href={`/predictions?symbol=${card.symbol}&tf=${card.timeframe}`}
                  className="rounded-lg border border-ink-600 bg-ink-800/70 p-4 transition hover:border-accent/40 focus:outline-none focus-visible:ring-2 focus-visible:ring-accent"
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="font-mono text-sm text-slate-300">{card.symbol}</span>
                    <div className="flex items-center gap-2">
                      {card.market_status && (
                        <span
                          className={`text-[10px] font-semibold uppercase ${
                            card.market_status === "LIVE"
                              ? "text-long"
                              : card.market_status === "STALE"
                                ? "text-amber-300"
                                : "text-red-300"
                          }`}
                        >
                          {card.market_status}
                        </span>
                      )}
                      {card.prediction ? <PredictionBadge value={card.prediction} /> : null}
                    </div>
                  </div>
                  {card.error ? (
                    <p className="mt-3 text-xs text-red-400">Market data temporarily unavailable</p>
                  ) : (
                    <>
                      <div className="mt-3 font-mono text-xl text-white">
                        ${Number(card.current_price).toLocaleString(undefined, { maximumFractionDigits: 2 })}
                      </div>
                      <div className="mt-2 flex justify-between text-xs text-slate-500">
                        <span>Strength {card.signal_strength} / 100</span>
                        <span className="capitalize">{card.trend}</span>
                      </div>
                      <div className="mt-2 text-[10px] text-slate-600">
                        {card.prediction_status ? `Prediction ${card.prediction_status}` : "—"}
                      </div>
                    </>
                  )}
                </Link>
              ))}
            </div>
          </section>

          <Panel title="BTC-USDT 4H — recent price">
            {chartData.length === 0 ? (
              <EmptyState
                title="Market chart unavailable"
                detail="Public market data could not be loaded for this chart."
              />
            ) : (
              <div className="h-56">
                <ResponsiveContainer>
                  <LineChart data={chartData}>
                    <CartesianGrid stroke="#1c2430" strokeDasharray="3 3" />
                    <XAxis dataKey="t" tick={{ fill: "#64748b", fontSize: 10 }} minTickGap={40} />
                    <YAxis domain={["auto", "auto"]} tick={{ fill: "#64748b", fontSize: 10 }} width={70} />
                    <Tooltip contentStyle={{ background: "#151b24", border: "1px solid #243041" }} />
                    <Line type="monotone" dataKey="close" stroke="#e2e8f0" dot={false} strokeWidth={1.5} />
                    <Line type="monotone" dataKey="ema20" stroke="#3d8bfd" dot={false} strokeWidth={1} />
                    <Line type="monotone" dataKey="ema50" stroke="#f59e0b" dot={false} strokeWidth={1} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            )}
          </Panel>

          <div className="grid gap-4 lg:grid-cols-3">
            <Panel title="Historical prediction performance">
              {!perf || !perf.total ? (
                <EmptyState
                  title="No evaluated prediction history"
                  detail="Run a historical backfill or refresh live analysis to accumulate snapshots."
                />
              ) : (
                <dl className="space-y-2 text-xs text-slate-400">
                  <div className="flex justify-between border-b border-ink-700 py-2">
                    <dt>Total evaluated</dt>
                    <dd className="font-mono text-slate-200">{perf.evaluated}</dd>
                  </div>
                  <div className="flex justify-between border-b border-ink-700 py-2">
                    <dt>WIN rate</dt>
                    <dd className="font-mono text-slate-200">
                      {perf.insufficient_sample
                        ? "Insufficient sample"
                        : perf.win_rate == null
                          ? "—"
                          : `${perf.win_rate}%`}
                    </dd>
                  </div>
                  <div className="flex justify-between border-b border-ink-700 py-2">
                    <dt>LONG / SHORT WIN</dt>
                    <dd className="font-mono text-slate-200">
                      {perf.long_win_rate ?? "—"}% / {perf.short_win_rate ?? "—"}%
                    </dd>
                  </div>
                  <div className="flex justify-between border-b border-ink-700 py-2">
                    <dt>Pending</dt>
                    <dd className="font-mono text-slate-200">{perf.pending}</dd>
                  </div>
                  <p className="pt-1 text-[10px] text-slate-600">
                    Historical outcomes only — not future probability. Do not treat as accuracy.
                  </p>
                  <Link href="/prediction-performance" className="inline-block text-xs text-accent hover:underline">
                    View performance detail →
                  </Link>
                </dl>
              )}
            </Panel>

            <Panel title="Research status">
              <div className="text-[10px] font-semibold uppercase tracking-wide text-amber-400">
                Final research conclusion
              </div>
              <p className="mt-2 font-mono text-lg uppercase tracking-wide text-amber-300">
                {data.research_conclusion}
              </p>
              <p className="mt-3 text-xs leading-relaxed text-slate-500">
                Historical validation across strategies, ML filters, Monte Carlo tests, and prediction
                snapshots has not demonstrated a durable predictive edge.
              </p>
              <Link href="/research" className="mt-4 inline-block text-xs text-accent hover:underline">
                Research timeline →
              </Link>
            </Panel>

            <Panel title="Latest analysis">
              <p className="text-xs leading-relaxed text-slate-400">
                Generate a closed-candle prediction for any supported asset and timeframe, store a
                snapshot, and compare it later against historical outcomes.
              </p>
              <div className="mt-4 flex flex-col gap-2">
                <Link
                  href="/predictions"
                  className="inline-flex justify-center rounded border border-accent/40 bg-accent/10 px-3 py-2 text-sm text-accent hover:bg-accent/20"
                >
                  Open Predictions
                </Link>
                <Link href="/methodology" className="text-center text-xs text-slate-500 hover:text-slate-300">
                  How QuantLab works
                </Link>
              </div>
              {analytics?.verdict && (
                <p className="mt-4 text-[10px] text-slate-600">
                  Phase 11 artifact verdict: {analytics.verdict}
                </p>
              )}
            </Panel>
          </div>
        </>
      )}
    </div>
  );
}
