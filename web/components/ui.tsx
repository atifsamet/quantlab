"use client";

export function StatusBanner() {
  return (
    <div
      className="flex flex-wrap items-center gap-2 border-b border-ink-600 bg-ink-900/90 px-4 py-2.5 text-xs lg:px-8"
      role="status"
      aria-label="QuantLab operating mode"
    >
      <span className="rounded border border-amber-700/60 bg-amber-950/50 px-2 py-0.5 font-semibold uppercase tracking-wide text-amber-300">
        Research Mode
      </span>
      <span className="rounded border border-red-800/50 bg-red-950/40 px-2 py-0.5 font-semibold uppercase tracking-wide text-red-300">
        Live Trading Disabled
      </span>
      <span className="hidden text-slate-500 sm:inline">
        Quantitative research &amp; market analysis — no trade execution
      </span>
    </div>
  );
}

export function PredictionBadge({ value }: { value: string }) {
  const color =
    value === "LONG"
      ? "text-long border-long/40 bg-long/10"
      : value === "SHORT"
        ? "text-short border-short/40 bg-short/10"
        : "text-neutral border-slate-600 bg-slate-800/50";
  return (
    <span
      className={`rounded border px-2 py-0.5 font-mono text-sm font-semibold ${color}`}
      aria-label={`Prediction ${value}`}
    >
      {value}
    </span>
  );
}

export function OutcomeBadge({
  outcome,
  evaluated,
}: {
  outcome: string | null;
  evaluated: boolean;
}) {
  if (!evaluated) {
    return (
      <span className="rounded border border-ink-600 px-2 py-0.5 font-mono text-[11px] text-slate-400">
        Pending
      </span>
    );
  }
  const label = outcome || "—";
  const tone =
    outcome === "WIN"
      ? "border-long/40 text-long bg-long/10"
      : outcome === "LOSS"
        ? "border-short/40 text-short bg-short/10"
        : outcome === "TIMEOUT"
          ? "border-amber-700/40 text-amber-300 bg-amber-950/30"
          : "border-ink-600 text-slate-400";
  return (
    <span className={`rounded border px-2 py-0.5 font-mono text-[11px] ${tone}`} aria-label={`Outcome ${label}`}>
      {label}
    </span>
  );
}

export function EmptyState({ title, detail }: { title: string; detail?: string }) {
  return (
    <div
      className="rounded-lg border border-dashed border-ink-600 bg-ink-800/40 px-6 py-10 text-center"
      role="status"
    >
      <div className="text-sm font-medium text-slate-300">{title}</div>
      {detail && <div className="mx-auto mt-2 max-w-md text-xs leading-relaxed text-slate-500">{detail}</div>}
    </div>
  );
}

export function LoadingState({ label = "Loading…" }: { label?: string }) {
  return (
    <div className="rounded-lg border border-ink-600 bg-ink-800/40 px-6 py-10 text-center" role="status" aria-live="polite">
      <div className="mx-auto h-1.5 w-24 overflow-hidden rounded bg-ink-700">
        <div className="h-full w-1/2 animate-pulse rounded bg-accent/60" />
      </div>
      <div className="mt-3 text-sm text-slate-400">{label}</div>
    </div>
  );
}

export function Panel({
  title,
  children,
  action,
  id,
}: {
  title: string;
  children: React.ReactNode;
  action?: React.ReactNode;
  id?: string;
}) {
  return (
    <section id={id} className="rounded-lg border border-ink-600 bg-ink-800/60">
      <div className="flex items-center justify-between gap-3 border-b border-ink-600 px-4 py-3">
        <h2 className="text-sm font-semibold tracking-tight text-slate-200">{title}</h2>
        {action}
      </div>
      <div className="p-4 sm:p-5">{children}</div>
    </section>
  );
}

export function PageHeader({
  eyebrow,
  title,
  description,
  actions,
}: {
  eyebrow?: string;
  title: string;
  description?: string;
  actions?: React.ReactNode;
}) {
  return (
    <header className="flex flex-wrap items-end justify-between gap-4">
      <div>
        {eyebrow && (
          <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-accent">{eyebrow}</div>
        )}
        <h1 className={`text-2xl font-semibold tracking-tight text-white ${eyebrow ? "mt-1" : ""}`}>
          {title}
        </h1>
        {description && <p className="mt-1 max-w-2xl text-sm text-slate-400">{description}</p>}
      </div>
      {actions}
    </header>
  );
}

export function DisclaimerFooter() {
  return (
    <footer className="mt-8 border-t border-ink-700 pt-4 text-[11px] leading-relaxed text-slate-600">
      QuantLab is an experimental quantitative research and market-analysis platform. Its outputs are
      based on historical and current market data and are not guarantees of future performance.
      QuantLab does not execute trades or provide personalized financial advice.
    </footer>
  );
}

export function PipelineDiagram() {
  const steps = [
    "Public OKX Data",
    "Technical Indicators",
    "Strategy / Signal Engine",
    "Prediction",
    "Historical Evaluation",
    "Statistical Analysis",
  ];
  return (
    <ol className="flex flex-col gap-2 lg:flex-row lg:flex-wrap lg:items-center lg:gap-x-2 lg:gap-y-3">
      {steps.map((step, i) => (
        <li key={step} className="flex items-center gap-2">
          <span className="rounded border border-ink-600 bg-ink-900/70 px-3 py-2 font-mono text-[11px] text-slate-300">
            {step}
          </span>
          {i < steps.length - 1 && (
            <span className="hidden text-slate-600 lg:inline" aria-hidden>
              →
            </span>
          )}
          {i < steps.length - 1 && (
            <span className="text-slate-600 lg:hidden" aria-hidden>
              ↓
            </span>
          )}
        </li>
      ))}
    </ol>
  );
}
