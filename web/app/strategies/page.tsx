"use client";

import { useEffect, useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "@/lib/api";
import { EmptyState, PageHeader, Panel } from "@/components/ui";

type Row = Record<string, any>;

export default function StrategiesPage() {
  const [rows, setRows] = useState<Row[]>([]);
  const [sortKey, setSortKey] = useState("val_return");
  const [asc, setAsc] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [best, setBest] = useState<any>(null);

  useEffect(() => {
    api
      .strategies()
      .then((d) => {
        setRows(d.rows || []);
        setBest(d.phase9_best);
      })
      .catch((e) => setError(e.message));
  }, []);

  const sorted = useMemo(() => {
    const copy = [...rows];
    copy.sort((a, b) => {
      const av = Number(a[sortKey] ?? -Infinity);
      const bv = Number(b[sortKey] ?? -Infinity);
      return asc ? av - bv : bv - av;
    });
    return copy;
  }, [rows, sortKey, asc]);

  const chart = useMemo(() => {
    const byVariant: Record<string, number[]> = {};
    for (const r of rows) {
      if (r.timeframe !== "4h") continue;
      const v = String(r.variant);
      byVariant[v] = byVariant[v] || [];
      if (r.val_return != null) byVariant[v].push(Number(r.val_return));
    }
    return Object.entries(byVariant).map(([variant, vals]) => ({
      variant,
      mean_val: vals.reduce((a, b) => a + b, 0) / Math.max(vals.length, 1),
    }));
  }, [rows]);

  function toggleSort(key: string) {
    if (sortKey === key) setAsc(!asc);
    else {
      setSortKey(key);
      setAsc(false);
    }
  }

  return (
    <div className="space-y-6">
      <PageHeader
        eyebrow="Strategy research"
        title="Strategies"
        description="Phase 9 multi-asset / multi-timeframe research cells — historical backtest metrics, not live predictions."
      />
      {best && (
        <p className="text-xs text-slate-500">
          Validation best: <span className="font-mono text-slate-300">{best.variant}@{best.timeframe}</span> — not claimed profitable
        </p>
      )}

      {error && <EmptyState title="Failed to load strategies" detail={error} />}
      {!error && rows.length === 0 && <EmptyState title="No strategy research CSV found" detail="Run Phase 9 research to generate cells_summary.csv" />}

      {chart.length > 0 && (
        <Panel title="Mean validation return by family (4h)">
          <div className="h-56">
            <ResponsiveContainer>
              <BarChart data={chart}>
                <CartesianGrid stroke="#1c2430" strokeDasharray="3 3" />
                <XAxis dataKey="variant" tick={{ fill: "#94a3b8", fontSize: 10 }} />
                <YAxis tick={{ fill: "#64748b", fontSize: 10 }} />
                <Tooltip contentStyle={{ background: "#151b24", border: "1px solid #243041" }} />
                <Bar dataKey="mean_val" fill="#3d8bfd" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Panel>
      )}

      {sorted.length > 0 && (
        <Panel title="Strategy comparison table">
          <div className="overflow-x-auto">
            <table className="w-full min-w-[900px] text-left text-xs">
              <thead className="text-slate-500">
                <tr>
                  {[
                    ["variant", "Strategy"],
                    ["symbol", "Asset"],
                    ["timeframe", "TF"],
                    ["val_return", "Val return"],
                    ["val_trades", "Trades"],
                    ["val_pf", "PF"],
                    ["val_dd", "Max DD"],
                    ["final_return", "Final test"],
                  ].map(([key, label]) => (
                    <th
                      key={key}
                      className="cursor-pointer border-b border-ink-600 px-2 py-2 font-medium hover:text-slate-300"
                      onClick={() => toggleSort(key)}
                    >
                      {label}
                      {sortKey === key ? (asc ? " ↑" : " ↓") : ""}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {sorted.slice(0, 200).map((r, i) => (
                  <tr key={i} className="border-b border-ink-700/70">
                    <td className="px-2 py-2 font-mono">{r.variant}</td>
                    <td className="px-2 py-2">{r.symbol}</td>
                    <td className="px-2 py-2">{r.timeframe}</td>
                    <td className="px-2 py-2 font-mono">{r.val_return == null ? "—" : Number(r.val_return).toFixed(2)}</td>
                    <td className="px-2 py-2 font-mono">{r.val_trades ?? "—"}</td>
                    <td className="px-2 py-2 font-mono">{r.val_pf == null ? "—" : Number(r.val_pf).toFixed(2)}</td>
                    <td className="px-2 py-2 font-mono">{r.val_dd == null ? "—" : Number(r.val_dd).toFixed(2)}</td>
                    <td className="px-2 py-2 font-mono">{r.final_return == null ? "—" : Number(r.final_return).toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>
      )}
    </div>
  );
}
