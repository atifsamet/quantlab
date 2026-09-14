import type { Metadata } from "next";
import Link from "next/link";
import { PageHeader, Panel, PipelineDiagram } from "@/components/ui";

export const metadata: Metadata = {
  title: "Methodology",
  description: "How QuantLab collects data, generates predictions, and evaluates historical outcomes.",
};

export default function MethodologyPage() {
  return (
    <div className="space-y-6">
      <PageHeader
        eyebrow="How it works"
        title="Methodology"
        description="Deterministic research pipeline using public market data. No private trading APIs. No order execution."
      />

      <Panel title="Data & model flow">
        <PipelineDiagram />
      </Panel>

      <div className="grid gap-4 lg:grid-cols-2">
        <Panel title="Market data">
          <ul className="space-y-2 text-xs text-slate-400">
            <li>• Source: public OKX candle endpoints only</li>
            <li>• Assets: BTC-USDT, ETH-USDT, SOL-USDT, XRP-USDT</li>
            <li>• Timeframes: 15m, 1H, 4H</li>
            <li>• Predictions use the latest closed candle when possible</li>
          </ul>
        </Panel>
        <Panel title="Indicators">
          <ul className="space-y-2 text-xs text-slate-400">
            <li>• EMA 20 / 50 / 200</li>
            <li>• RSI 14</li>
            <li>• MACD (line, signal, histogram)</li>
            <li>• ATR 14</li>
            <li>• Volume &amp; volume SMA</li>
          </ul>
        </Panel>
        <Panel title="Strategies researched">
          <ul className="space-y-2 text-xs text-slate-400">
            <li>• Baseline trend + momentum</li>
            <li>• Breakout</li>
            <li>• Trend following</li>
            <li>• Momentum</li>
            <li>• Mean reversion</li>
            <li>• Live prediction engine leans on breakout + indicator agreement</li>
          </ul>
        </Panel>
        <Panel title="ML & statistics">
          <ul className="space-y-2 text-xs text-slate-400">
            <li>• Phase 10: ML signal filter (no durable edge)</li>
            <li>• Monte Carlo trade resampling</li>
            <li>• Bootstrap intervals</li>
            <li>• Random benchmark comparison</li>
            <li>• Cost / slippage sensitivity</li>
            <li>• Walk-forward / robust validation in earlier phases</li>
          </ul>
        </Panel>
        <Panel title="Prediction outputs">
          <ul className="space-y-2 text-xs text-slate-400">
            <li>• Direction: LONG / SHORT / NEUTRAL</li>
            <li>• Signal strength: 0–100 analytical score (not probability)</li>
            <li>• Model levels: entry zone, stop, take-profit, R:R</li>
            <li>• Deterministic explanation factors</li>
          </ul>
        </Panel>
        <Panel title="Historical evaluation">
          <ul className="space-y-2 text-xs text-slate-400">
            <li>• WIN — take-profit before stop-loss</li>
            <li>• LOSS — stop-loss first (including same-candle SL-first)</li>
            <li>• TIMEOUT — neither level within horizon</li>
            <li>• SKIPPED — NEUTRAL / missing levels</li>
            <li>• Causal: predict on candles ≤ N; evaluate on N+1…</li>
          </ul>
        </Panel>
      </div>

      <Panel title="What QuantLab is not">
        <ul className="space-y-2 text-xs text-slate-400">
          <li>• Not a trading bot or exchange</li>
          <li>• Not an investment advisory service</li>
          <li>• Not a claim of profitable foresight</li>
        </ul>
        <p className="mt-4 font-mono text-sm uppercase text-amber-300">NO ROBUST EDGE DETECTED</p>
        <Link href="/research" className="mt-3 inline-block text-xs text-accent hover:underline">
          See research timeline →
        </Link>
      </Panel>
    </div>
  );
}
