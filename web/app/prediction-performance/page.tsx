"use client";

import { useEffect, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api } from "@/lib/api";
import { EmptyState, LoadingState, PageHeader, Panel } from "@/components/ui";

function fmtRate(v: number | null | undefined, n?: number, insufficient?: boolean) {
  if (v == null) return "—";
  if (insufficient) return `${v.toFixed(1)}% (n=${n ?? 0}, insufficient)`;
  return `${v.toFixed(1)}%${n != null ? ` (n=${n})` : ""}`;
}

export default function PredictionPerformancePage() {
  const [data, setData] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    api
      .predictionStats()
      .then(setData)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  const assetChart = useMemo(() => {
    if (!data?.by_asset) return [];
    return Object.entries(data.by_asset).map(([symbol, m]: any) => ({
      symbol: symbol.replace("-USDT", ""),
      win_rate: m.win_rate ?? 0,
      n: m.scored_n ?? 0,
    }));
  }, [data]);

  const tfChart = useMemo(() => {
    if (!data?.by_timeframe) return [];
    return Object.entries(data.by_timeframe).map(([tf, m]: any) => ({
      timeframe: tf,
      win_rate: m.win_rate ?? 0,
      n: m.scored_n ?? 0,
    }));
  }, [data]);

  const strengthChart = useMemo(() => {
    return (data?.by_signal_strength || []).map((b: any) => ({
      bucket: b.bucket,
      win_rate: b.win_rate ?? 0,
      n: b.n,
    }));
  }, [data]);

  const outcomeChart = useMemo(() => {
    const d = data?.outcome_distribution || {};
    return Object.entries(d).map(([k, v]) => ({ outcome: k, count: v as number }));
  }, [data]);

  const o = data?.overall;

  return (
    <div className="space-y-6">
      <PageHeader
        eyebrow="Historical evaluation"
        title="Prediction Performance"
        description="Measured outcomes of stored prediction snapshots — not future win probability."
      />

      {loading && <LoadingState label="Loading historical performance…" />}
      {error && <EmptyState title="Stats unavailable" detail={error} />}
      {!loading && !error && !data && <EmptyState title="No prediction history yet" detail="Run: python -m app.prediction.backfill" />}
      {data && (o?.total_predictions ?? 0) === 0 && (
        <EmptyState
          title="No prediction history yet"
          detail="Run: python -m app.prediction.backfill"
        />
      )}

      {data && o && (o.total_predictions ?? 0) > 0 && (
        <>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            {[
              ["Total", o.total_predictions],
              ["Evaluated", o.evaluated_predictions],
              ["Pending", o.pending_predictions],
              ["Win rate", o.insufficient_sample ? "Insufficient sample" : fmtRate(o.win_rate, o.scored_n)],
              ["Loss rate", fmtRate(o.loss_rate)],
              ["Timeout rate", fmtRate(o.timeout_rate)],
              ["LONG win rate", fmtRate(o.long?.win_rate, o.long?.n, o.long?.insufficient_sample)],
              ["SHORT win rate", fmtRate(o.short?.win_rate, o.short?.n, o.short?.insufficient_sample)],
            ].map(([k, v]) => (
              <div key={String(k)} className="rounded-lg border border-ink-600 bg-ink-800/60 p-3">
                <div className="text-[11px] uppercase text-slate-500">{k}</div>
                <div className="mt-1 font-mono text-sm text-slate-100">{v}</div>
              </div>
            ))}
          </div>

          <Panel title="Signal strength vs historical win rate">
            <p className="mb-3 text-[11px] text-slate-500">
              Strength is an analytical score. Bars show measured historical success for each band.
            </p>
            <div className="h-56">
              <ResponsiveContainer>
                <BarChart data={strengthChart}>
                  <CartesianGrid stroke="#1c2430" strokeDasharray="3 3" />
                  <XAxis dataKey="bucket" tick={{ fill: "#94a3b8", fontSize: 11 }} />
                  <YAxis tick={{ fill: "#64748b", fontSize: 10 }} domain={[0, 100]} />
                  <Tooltip
                    contentStyle={{ background: "#151b24", border: "1px solid #243041" }}
                    formatter={(value: any, _n: any, item: any) => [
                      `${Number(value).toFixed(1)}% (n=${item?.payload?.n ?? 0})`,
                      "Historical win rate",
                    ]}
                  />
                  <Bar dataKey="win_rate" fill="#3d8bfd" />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <ul className="mt-3 grid gap-2 sm:grid-cols-4 text-xs text-slate-400">
              {(data.by_signal_strength || []).map((b: any) => (
                <li key={b.bucket} className="rounded border border-ink-600 px-2 py-2">
                  <div className="font-mono text-slate-200">{b.bucket}</div>
                  <div>
                    {b.win_rate == null
                      ? "—"
                      : b.insufficient_sample
                        ? `Insufficient (n=${b.n})`
                        : `${b.win_rate}% · n=${b.n}`}
                  </div>
                </li>
              ))}
            </ul>
          </Panel>

          <div className="grid gap-4 lg:grid-cols-2">
            <Panel title="Performance by asset">
              <div className="h-52">
                <ResponsiveContainer>
                  <BarChart data={assetChart}>
                    <CartesianGrid stroke="#1c2430" strokeDasharray="3 3" />
                    <XAxis dataKey="symbol" tick={{ fill: "#94a3b8", fontSize: 11 }} />
                    <YAxis tick={{ fill: "#64748b", fontSize: 10 }} domain={[0, 100]} />
                    <Tooltip contentStyle={{ background: "#151b24", border: "1px solid #243041" }} />
                    <Bar dataKey="win_rate" fill="#2ecc71" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </Panel>
            <Panel title="Performance by timeframe">
              <div className="h-52">
                <ResponsiveContainer>
                  <BarChart data={tfChart}>
                    <CartesianGrid stroke="#1c2430" strokeDasharray="3 3" />
                    <XAxis dataKey="timeframe" tick={{ fill: "#94a3b8", fontSize: 11 }} />
                    <YAxis tick={{ fill: "#64748b", fontSize: 10 }} domain={[0, 100]} />
                    <Tooltip contentStyle={{ background: "#151b24", border: "1px solid #243041" }} />
                    <Bar dataKey="win_rate" fill="#f59e0b" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </Panel>
          </div>

          <div className="grid gap-4 lg:grid-cols-2">
            <Panel title="Outcome distribution">
              <div className="h-52">
                <ResponsiveContainer>
                  <BarChart data={outcomeChart}>
                    <CartesianGrid stroke="#1c2430" strokeDasharray="3 3" />
                    <XAxis dataKey="outcome" tick={{ fill: "#94a3b8", fontSize: 11 }} />
                    <YAxis tick={{ fill: "#64748b", fontSize: 10 }} />
                    <Tooltip contentStyle={{ background: "#151b24", border: "1px solid #243041" }} />
                    <Bar dataKey="count" fill="#94a3b8" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </Panel>
            <Panel title="LONG vs SHORT">
              <dl className="space-y-3 text-xs">
                <div className="flex justify-between border-b border-ink-700 py-2">
                  <dt className="text-slate-500">LONG</dt>
                  <dd className="font-mono">
                    {fmtRate(o.long?.win_rate, o.long?.n, o.long?.insufficient_sample)}
                  </dd>
                </div>
                <div className="flex justify-between border-b border-ink-700 py-2">
                  <dt className="text-slate-500">SHORT</dt>
                  <dd className="font-mono">
                    {fmtRate(o.short?.win_rate, o.short?.n, o.short?.insufficient_sample)}
                  </dd>
                </div>
                <div className="flex justify-between border-b border-ink-700 py-2">
                  <dt className="text-slate-500">Avg return</dt>
                  <dd className="font-mono">{o.average_return_pct == null ? "—" : `${o.average_return_pct}%`}</dd>
                </div>
                <div className="flex justify-between border-b border-ink-700 py-2">
                  <dt className="text-slate-500">Avg MFE / MAE</dt>
                  <dd className="font-mono">
                    {o.average_mfe_pct ?? "—"}% / {o.average_mae_pct ?? "—"}%
                  </dd>
                </div>
              </dl>
            </Panel>
          </div>

          <p className="text-[11px] text-slate-500">{data.disclaimer}</p>
        </>
      )}
    </div>
  );
}
