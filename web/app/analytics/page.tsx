"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { EmptyState, PageHeader, Panel } from "@/components/ui";

export default function AnalyticsPage() {
  const [data, setData] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.analytics().then(setData).catch((e) => setError(e.message));
  }, []);

  return (
    <div className="space-y-6">
      <PageHeader
        eyebrow="Statistical validation"
        title="Analytics"
        description="Phase 11 historical statistical artifacts — not live prediction performance."
      />

      {error && <EmptyState title="Analytics unavailable" detail={error} />}
      {data && !data.available && (
        <EmptyState title="No Phase 11 report found" detail="Run: python -m app.stats.phase11" />
      )}

      {data?.available && (
        <>
          <Panel title="Verdict">
            <p className="font-mono text-sm uppercase text-amber-300">{data.verdict}</p>
            {data.eval_window && (
              <p className="mt-2 text-xs text-slate-500">
                Eval window: {String(data.eval_window.start || data.eval_window)} →{" "}
                {String(data.eval_window.end || "")}
              </p>
            )}
            <div className="mt-4 space-y-2 text-xs text-slate-400">
              {Object.entries(data.answers || {}).map(([k, v]) => (
                <div key={k}>
                  <span className="text-slate-500">{k}: </span>
                  <span className="text-slate-300">{String(v)}</span>
                </div>
              ))}
            </div>
          </Panel>

          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            {[
              ["Pooled mean return", data.pooled_mean_return_pct, "%"],
              ["Buy & hold mean", data.buy_and_hold_mean_return, "%"],
            ].map(([label, val, suffix]) => (
              <div key={String(label)} className="rounded-lg border border-ink-600 bg-ink-800/60 p-3">
                <div className="text-[11px] uppercase text-slate-500">{label}</div>
                <div className="mt-1 font-mono text-sm text-slate-100">
                  {val == null ? "—" : `${Number(val).toFixed(2)}${suffix}`}
                </div>
              </div>
            ))}
          </div>

          <Panel title="Per-asset breakout@4h (eval window)">
            <div className="overflow-x-auto">
              <table className="w-full min-w-[800px] text-left text-xs">
                <thead className="text-slate-500">
                  <tr>
                    {[
                      "symbol",
                      "return_pct",
                      "trades",
                      "win_rate",
                      "profit_factor",
                      "expectancy",
                      "max_dd",
                      "long_signal_pct",
                      "short_signal_pct",
                    ].map((h) => (
                      <th key={h} className="border-b border-ink-600 px-2 py-2">
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(data.per_asset || []).map((r: any) => (
                    <tr key={r.symbol} className="border-b border-ink-700/70">
                      <td className="px-2 py-2 font-mono">{r.symbol}</td>
                      <td className="px-2 py-2 font-mono">{Number(r.return_pct).toFixed(2)}</td>
                      <td className="px-2 py-2 font-mono">{r.trades}</td>
                      <td className="px-2 py-2 font-mono">{Number(r.win_rate).toFixed(1)}</td>
                      <td className="px-2 py-2 font-mono">
                        {r.profit_factor == null ? "—" : Number(r.profit_factor).toFixed(2)}
                      </td>
                      <td className="px-2 py-2 font-mono">{Number(r.expectancy).toFixed(3)}</td>
                      <td className="px-2 py-2 font-mono">{Number(r.max_dd).toFixed(2)}</td>
                      <td className="px-2 py-2 font-mono">{Number(r.long_signal_pct).toFixed(1)}</td>
                      <td className="px-2 py-2 font-mono">{Number(r.short_signal_pct).toFixed(1)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Panel>

          <div className="grid gap-4 lg:grid-cols-2">
            <Panel title="Bootstrap (pooled)">
              {(data.bootstrap_pooled || []).length === 0 ? (
                <EmptyState title="No bootstrap rows" />
              ) : (
                <ul className="space-y-2 text-xs">
                  {data.bootstrap_pooled.map((r: any) => (
                    <li key={r.metric} className="border-b border-ink-700 py-2">
                      <div className="flex justify-between">
                        <span className="text-slate-400">{r.metric}</span>
                        <span className="font-mono">{Number(r.point_estimate).toFixed(2)}</span>
                      </div>
                      <div className="mt-1 text-[11px] text-slate-600">
                        P05–P95: {Number(r.p05).toFixed(2)} … {Number(r.p95).toFixed(2)}
                        {r.reliable === false ? " · exploratory only" : ""}
                      </div>
                    </li>
                  ))}
                </ul>
              )}
            </Panel>
            <Panel title="Cost sensitivity (multi-asset mean)">
              {(data.cost_sensitivity || []).length === 0 ? (
                <EmptyState title="No cost sensitivity rows" />
              ) : (
                <ul className="space-y-2 text-xs">
                  {data.cost_sensitivity.map((r: any, i: number) => (
                    <li key={i} className="flex justify-between border-b border-ink-700 py-2">
                      <span className="text-slate-400">
                        {r.scenario} (fee {r.fee_rate}, slip {r.slippage_rate})
                      </span>
                      <span className="font-mono">{Number(r.mean_return_pct).toFixed(2)}%</span>
                    </li>
                  ))}
                </ul>
              )}
            </Panel>
          </div>

          <Panel title="Random / buy-hold benchmarks">
            {(data.benchmarks || []).length === 0 ? (
              <EmptyState title="No benchmark CSV" />
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full min-w-[600px] text-left text-xs">
                  <thead className="text-slate-500">
                    <tr>
                      {["name", "symbol", "return_pct", "trades", "win_rate", "max_dd"].map((h) => (
                        <th key={h} className="border-b border-ink-600 px-2 py-2">
                          {h}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {data.benchmarks.slice(0, 40).map((r: any, i: number) => (
                      <tr key={i} className="border-b border-ink-700/70">
                        <td className="px-2 py-2 font-mono">{r.name}</td>
                        <td className="px-2 py-2">{r.symbol}</td>
                        <td className="px-2 py-2 font-mono">{Number(r.return_pct).toFixed(2)}</td>
                        <td className="px-2 py-2 font-mono">{r.trades}</td>
                        <td className="px-2 py-2 font-mono">{Number(r.win_rate).toFixed(1)}</td>
                        <td className="px-2 py-2 font-mono">{Number(r.max_dd).toFixed(2)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Panel>

          <Panel title="Notes">
            <ul className="space-y-2 text-xs text-slate-400">
              {(data.notes || []).map((n: string) => (
                <li key={n}>• {n}</li>
              ))}
              <li>• Walk-forward and Phase 7–9 robustness details live under Strategies / Research.</li>
            </ul>
          </Panel>
        </>
      )}
    </div>
  );
}
