"use client";

import { useEffect, useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "@/lib/api";
import { EmptyState, PageHeader, Panel } from "@/components/ui";

export default function MonteCarloPage() {
  const [data, setData] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.monteCarlo().then(setData).catch((e) => setError(e.message));
  }, []);

  const pooled = data?.pooled || data?.summary?.find((r: any) => r.scope === "pooled");
  const dd = useMemo(() => data?.drawdown_distribution || [], [data]);

  return (
    <div className="space-y-6">
      <PageHeader
        eyebrow="Uncertainty analysis"
        title="Monte Carlo"
        description="Phase 11 trade-resampling artifacts — model-based uncertainty, not a forecast."
      />

      {error && <EmptyState title="Monte Carlo unavailable" detail={error} />}
      {data && !data.available && <EmptyState title="No Monte Carlo artifacts" detail="Run Phase 11 first" />}

      {data?.available && pooled && (
        <>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            {[
              ["Simulations", pooled.n_sims],
              ["Median return", `${Number(pooled.median_return).toFixed(2)}%`],
              ["P05 / P95", `${Number(pooled.p05_return).toFixed(2)}% / ${Number(pooled.p95_return).toFixed(2)}%`],
              ["P(loss)", `${(Number(pooled.prob_losing) * 100).toFixed(1)}%`],
              ["P(DD>5%)", `${(Number(pooled.prob_dd_gt_5) * 100).toFixed(1)}%`],
              ["P(DD>10%)", `${(Number(pooled.prob_dd_gt_10) * 100).toFixed(1)}%`],
              ["P(DD>20%)", `${(Number(pooled.prob_dd_gt_20) * 100).toFixed(1)}%`],
              ["Worst DD", `${Number(pooled.worst_max_dd).toFixed(2)}%`],
            ].map(([k, v]) => (
              <div key={String(k)} className="rounded-lg border border-ink-600 bg-ink-800/60 p-3">
                <div className="text-[11px] uppercase text-slate-500">{k}</div>
                <div className="mt-1 font-mono text-sm text-slate-100">{v}</div>
              </div>
            ))}
          </div>

          <Panel title="Drawdown distribution percentiles">
            {dd.length === 0 ? (
              <EmptyState title="No drawdown distribution CSV" />
            ) : (
              <div className="h-56">
                <ResponsiveContainer>
                  <BarChart data={dd.filter((r: any) => r.scope === "pooled")}>
                    <CartesianGrid stroke="#1c2430" strokeDasharray="3 3" />
                    <XAxis dataKey="percentile" tick={{ fill: "#94a3b8", fontSize: 10 }} />
                    <YAxis tick={{ fill: "#64748b", fontSize: 10 }} />
                    <Tooltip contentStyle={{ background: "#151b24", border: "1px solid #243041" }} />
                    <Bar dataKey="max_dd" fill="#e74c3c" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}
          </Panel>

          {data.random_null && (
            <Panel title="Random benchmark (BTC)">
              <dl className="grid gap-3 sm:grid-cols-2 text-xs">
                <div className="flex justify-between border-b border-ink-700 py-2">
                  <dt className="text-slate-500">Strategy return</dt>
                  <dd className="font-mono">{Number(data.random_null.observed_btc_return).toFixed(2)}%</dd>
                </div>
                <div className="flex justify-between border-b border-ink-700 py-2">
                  <dt className="text-slate-500">Random median</dt>
                  <dd className="font-mono">{Number(data.random_null.random_median).toFixed(2)}%</dd>
                </div>
                <div className="flex justify-between border-b border-ink-700 py-2">
                  <dt className="text-slate-500">P(random &lt; strategy)</dt>
                  <dd className="font-mono">
                    {(Number(data.random_null.fraction_random_below_strategy) * 100).toFixed(1)}%
                  </dd>
                </div>
              </dl>
            </Panel>
          )}

          <p className="text-[11px] text-slate-500">{pooled.note}</p>
          {data.verdict && (
            <p className="font-mono text-xs uppercase text-amber-300">Verdict: {data.verdict}</p>
          )}
        </>
      )}
    </div>
  );
}
