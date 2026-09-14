"use client";

import { useEffect, useState } from "react";
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
import { EmptyState, PageHeader, Panel } from "@/components/ui";

const SYMBOLS = ["BTC-USDT", "ETH-USDT", "SOL-USDT", "XRP-USDT"];
const TFS = ["15m", "1h", "4h"];

export default function BacktestsPage() {
  const [symbol, setSymbol] = useState("BTC-USDT");
  const [tf, setTf] = useState("1h");
  const [equity, setEquity] = useState<any>(null);
  const [trades, setTrades] = useState<any>(null);
  const [analytics, setAnalytics] = useState<any>(null);

  useEffect(() => {
    api.equity(symbol, tf).then(setEquity).catch(() => setEquity({ available: false, points: [] }));
    api.trades(symbol, tf).then(setTrades).catch(() => setTrades({ available: false, trades: [] }));
    api.analytics().then(setAnalytics).catch(() => setAnalytics(null));
  }, [symbol, tf]);

  const assetRow = analytics?.per_asset?.find((r: any) => r.symbol === symbol);

  const equityData = (equity?.points || []).map((p: any) => ({
    t: String(p.timestamp || p.Timestamp || "").slice(0, 16),
    equity: Number(p.equity ?? p.Equity ?? 0),
  }));

  return (
    <div className="space-y-6">
      <PageHeader
        eyebrow="Backtest artifacts"
        title="Backtests"
        description="Existing offline backtest equity and trade exports — not live predictions."
        actions={
          <div className="flex gap-2">
            <select className="rounded border border-ink-600 bg-ink-800 px-3 py-2 text-sm" value={symbol} onChange={(e) => setSymbol(e.target.value)} aria-label="Asset">
              {SYMBOLS.map((s) => (
                <option key={s}>{s}</option>
              ))}
            </select>
            <select className="rounded border border-ink-600 bg-ink-800 px-3 py-2 text-sm" value={tf} onChange={(e) => setTf(e.target.value)} aria-label="Timeframe">
              {TFS.map((t) => (
                <option key={t}>{t}</option>
              ))}
            </select>
          </div>
        }
      />

      {assetRow ? (
        <div className="grid gap-3 sm:grid-cols-3 lg:grid-cols-6">
          {[
            ["Return", `${Number(assetRow.return_pct).toFixed(2)}%`],
            ["Trades", assetRow.trades],
            ["Win rate", `${Number(assetRow.win_rate).toFixed(1)}%`],
            ["PF", assetRow.profit_factor == null ? "—" : Number(assetRow.profit_factor).toFixed(2)],
            ["Max DD", `${Number(assetRow.max_dd).toFixed(2)}%`],
            ["Expectancy", Number(assetRow.expectancy).toFixed(3)],
          ].map(([k, v]) => (
            <div key={String(k)} className="rounded-lg border border-ink-600 bg-ink-800/60 p-3">
              <div className="text-[11px] uppercase text-slate-500">{k}</div>
              <div className="mt-1 font-mono text-sm text-slate-100">{v}</div>
            </div>
          ))}
        </div>
      ) : (
        <EmptyState
          title="No Phase 11 per-asset stats for this selection"
          detail="Phase 11 reported breakout@4h. Switch to 4h or view Strategies / Analytics."
        />
      )}

      <Panel title="Equity curve">
        {!equity?.available || equityData.length === 0 ? (
          <EmptyState title="No equity CSV for this symbol/timeframe" detail="Export a backtest first or pick BTC-USDT 1h if available." />
        ) : (
          <div className="h-64">
            <ResponsiveContainer>
              <LineChart data={equityData}>
                <CartesianGrid stroke="#1c2430" strokeDasharray="3 3" />
                <XAxis dataKey="t" tick={{ fill: "#64748b", fontSize: 10 }} minTickGap={40} />
                <YAxis tick={{ fill: "#64748b", fontSize: 10 }} width={60} />
                <Tooltip contentStyle={{ background: "#151b24", border: "1px solid #243041" }} />
                <Line type="monotone" dataKey="equity" stroke="#3d8bfd" dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
      </Panel>

      <Panel title="Trades">
        {!trades?.available || !trades.trades?.length ? (
          <EmptyState title="No trades file found" />
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[640px] text-left text-xs">
              <thead className="text-slate-500">
                <tr>
                  {["side", "entry_timestamp", "exit_timestamp", "net_pnl", "return_pct"].map((h) => (
                    <th key={h} className="border-b border-ink-600 px-2 py-2 font-medium">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {trades.trades.slice(0, 50).map((t: any, i: number) => (
                  <tr key={i} className="border-b border-ink-700/80">
                    <td className="px-2 py-2 font-mono">{t.side}</td>
                    <td className="px-2 py-2 font-mono text-slate-400">{String(t.entry_timestamp).slice(0, 19)}</td>
                    <td className="px-2 py-2 font-mono text-slate-400">{String(t.exit_timestamp).slice(0, 19)}</td>
                    <td className="px-2 py-2 font-mono">{Number(t.net_pnl).toFixed(4)}</td>
                    <td className="px-2 py-2 font-mono">{Number(t.return_pct).toFixed(4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Panel>
    </div>
  );
}
