"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { api } from "@/lib/api";
import { EmptyState, LoadingState, PageHeader, Panel, PipelineDiagram } from "@/components/ui";

export default function ResearchPage() {
  const [data, setData] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    api
      .research()
      .then(setData)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="space-y-6">
      <PageHeader
        eyebrow="Research pipeline"
        title="Research"
        description="Phases 1–14 of QuantLab’s quantitative investigation — and the honest conclusion."
      />

      {loading && <LoadingState label="Loading research results…" />}
      {error && <EmptyState title="Research results unavailable" detail={error} />}

      {data && (
        <>
          <div className="rounded-lg border border-amber-700/50 bg-amber-950/30 p-5 sm:p-6">
            <div className="text-[11px] font-semibold uppercase tracking-[0.2em] text-amber-400">
              Final research conclusion
            </div>
            <div className="mt-2 font-mono text-2xl font-semibold uppercase text-amber-200">
              {data.conclusion}
            </div>
            <p className="mt-3 max-w-3xl text-sm leading-relaxed text-amber-100/70">
              The system was evaluated across multiple assets, timeframes, strategies, ML filters,
              historical prediction snapshots, and statistical tests. Results did not demonstrate a
              durable predictive edge.
            </p>
            <div className="mt-4 flex flex-wrap gap-2 text-[11px]">
              <span className="rounded border border-amber-700/50 px-2 py-1 text-amber-300">{data.mode}</span>
              <span className="rounded border border-red-800/50 px-2 py-1 text-red-300">
                LIVE TRADING DISABLED
              </span>
            </div>
          </div>

          <Panel title="Methodology pipeline">
            <PipelineDiagram />
            <p className="mt-4 text-xs text-slate-500">
              Signal strength is an analytical score, not a probability of future return.{" "}
              <Link href="/methodology" className="text-accent hover:underline">
                Full methodology →
              </Link>
            </p>
          </Panel>

          <Panel title="Research timeline">
            <ol className="space-y-3">
              {(data.timeline || []).map((p: any) => (
                <li
                  key={p.phase}
                  className="grid gap-2 rounded-md border border-ink-600 bg-ink-900/40 p-4 sm:grid-cols-[4.5rem_1fr]"
                >
                  <div className="font-mono text-xs text-accent">Phase {p.phase}</div>
                  <div>
                    <div className="text-sm font-medium text-slate-200">{p.title}</div>
                    <div className="mt-1 text-xs text-slate-400">Built / tested: {p.result}</div>
                    <div className="mt-0.5 text-xs text-slate-500">Result: {p.conclusion}</div>
                  </div>
                </li>
              ))}
            </ol>
          </Panel>
        </>
      )}
    </div>
  );
}
