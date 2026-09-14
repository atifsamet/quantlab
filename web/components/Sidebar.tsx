"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Activity,
  BarChart3,
  Beaker,
  BookOpen,
  BrainCircuit,
  LayoutDashboard,
  LineChart,
  Menu,
  ShieldOff,
  Target,
  X,
} from "lucide-react";
import { useState } from "react";

const NAV = [
  { href: "/", label: "Overview", icon: LayoutDashboard },
  { href: "/predictions", label: "Predictions", icon: Activity },
  { href: "/prediction-performance", label: "Pred. Performance", icon: Target },
  { href: "/backtests", label: "Backtests", icon: LineChart },
  { href: "/strategies", label: "Strategies", icon: BarChart3 },
  { href: "/analytics", label: "Analytics", icon: Beaker },
  { href: "/monte-carlo", label: "Monte Carlo", icon: BrainCircuit },
  { href: "/research", label: "Research", icon: Beaker },
  { href: "/methodology", label: "Methodology", icon: BookOpen },
];

export function Sidebar() {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);

  const nav = (
    <div className="flex h-full flex-col">
      <div className="border-b border-ink-600 px-5 py-5">
        <Link href="/" className="block focus:outline-none focus-visible:ring-2 focus-visible:ring-accent" onClick={() => setOpen(false)}>
          <div className="font-mono text-xs uppercase tracking-[0.2em] text-accent">QuantLab</div>
          <div className="mt-1 text-sm leading-snug text-slate-400">Quantitative Crypto Market Intelligence</div>
        </Link>
      </div>
      <nav className="flex-1 space-y-1 overflow-y-auto px-3 py-4" aria-label="Primary">
        {NAV.map(({ href, label, icon: Icon }) => {
          const active = pathname === href;
          return (
            <Link
              key={href}
              href={href}
              onClick={() => setOpen(false)}
              aria-current={active ? "page" : undefined}
              className={`flex items-center gap-3 rounded-md px-3 py-2 text-sm transition focus:outline-none focus-visible:ring-2 focus-visible:ring-accent ${
                active
                  ? "bg-ink-600 text-white"
                  : "text-slate-400 hover:bg-ink-700 hover:text-slate-200"
              }`}
            >
              <Icon size={16} aria-hidden />
              {label}
            </Link>
          );
        })}
      </nav>
      <div className="m-3 rounded-md border border-amber-700/50 bg-amber-950/40 p-3">
        <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-amber-400">
          <ShieldOff size={14} aria-hidden />
          Research Mode
        </div>
        <div className="mt-1 text-[11px] leading-snug text-amber-200/80">LIVE TRADING DISABLED</div>
      </div>
    </div>
  );

  return (
    <>
      <button
        type="button"
        className="fixed left-3 top-3 z-40 rounded-md border border-ink-600 bg-ink-800 p-2 text-slate-300 focus:outline-none focus-visible:ring-2 focus-visible:ring-accent lg:hidden"
        onClick={() => setOpen((v) => !v)}
        aria-label={open ? "Close navigation" : "Open navigation"}
        aria-expanded={open}
      >
        {open ? <X size={18} /> : <Menu size={18} />}
      </button>
      <aside className="hidden w-64 shrink-0 border-r border-ink-600 bg-ink-900 lg:block" aria-label="Sidebar">
        {nav}
      </aside>
      {open && (
        <div className="fixed inset-0 z-30 lg:hidden">
          <div className="absolute inset-0 bg-black/60" onClick={() => setOpen(false)} aria-hidden />
          <aside className="absolute left-0 top-0 h-full w-72 bg-ink-900 shadow-xl" aria-label="Mobile navigation">
            {nav}
          </aside>
        </div>
      )}
    </>
  );
}
