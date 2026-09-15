const DEFAULT_LOCAL_API = "http://127.0.0.1:8000";
const DEFAULT_PRODUCTION_API = "https://quantlab-api-0mtu.onrender.com";

function resolveApiBase(): string {
  const fromEnv = (process.env.NEXT_PUBLIC_API_BASE || "").trim().replace(/\/$/, "");
  if (fromEnv) return fromEnv;

  // Never point a deployed browser build at loopback.
  if (process.env.NODE_ENV === "production") {
    return DEFAULT_PRODUCTION_API;
  }
  return DEFAULT_LOCAL_API;
}

const API_BASE = resolveApiBase();

function safeErrorDetail(body: unknown): string {
  if (body && typeof body === "object" && "detail" in body) {
    const detail = (body as { detail: unknown }).detail;
    if (typeof detail === "string") return detail;
    try {
      return JSON.stringify(detail);
    } catch {
      return "Request failed";
    }
  }
  try {
    return JSON.stringify(body);
  } catch {
    return "Request failed";
  }
}

async function getJson<T>(path: string, init?: RequestInit): Promise<T> {
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  const url = `${API_BASE}${normalizedPath}`;

  let res: Response;
  try {
    res = await fetch(url, {
      ...init,
      cache: "no-store",
      headers: { Accept: "application/json", ...(init?.headers || {}) },
    });
  } catch (err) {
    const reason = err instanceof Error ? err.message : "Network error";
    throw new Error(`Network error for ${url}: ${reason}`);
  }

  if (!res.ok) {
    let detail = res.statusText || "Request failed";
    try {
      detail = safeErrorDetail(await res.json());
    } catch {
      /* ignore non-JSON error bodies */
    }
    throw new Error(`HTTP ${res.status} for ${url}: ${detail}`);
  }

  return res.json() as Promise<T>;
}

export type Prediction = {
  symbol: string;
  timeframe: string;
  timestamp: string;
  current_price: number;
  prediction: "LONG" | "SHORT" | "NEUTRAL";
  signal_strength: number;
  signal_strength_label: string;
  signal_strength_tooltip?: string;
  ml_probability: number | null;
  ml_probability_note: string | null;
  trend: string;
  strategy_signal: string;
  levels: {
    entry_low: number | null;
    entry_high: number | null;
    stop_loss: number | null;
    take_profit: number | null;
    risk_reward: number | null;
    label: string;
  };
  indicators: Record<string, number | null>;
  reasons: { positive: string[]; negative: string[] };
  warnings: string[];
  research_conclusion: string;
  disclaimer: string;
  engine_version?: string | null;
  live?: {
    market_status: "LIVE" | "STALE" | "ERROR";
    prediction_status: "CURRENT" | "STALE";
    data_timestamp?: string | null;
    last_market_update?: string | null;
    prediction_candle_open?: string | null;
    prediction_candle_close?: string | null;
    candle_source?: string;
    dropped_incomplete_candle?: boolean;
  } | null;
};

export type HistoryRecord = {
  id: string;
  symbol: string;
  timeframe: string;
  prediction_timestamp: string;
  candle_timestamp: string;
  current_price: number;
  prediction: string;
  signal_strength: number;
  entry_price: number;
  stop_loss: number | null;
  take_profit: number | null;
  evaluated: boolean;
  outcome: string | null;
  return_pct: number | null;
  mfe_pct: number | null;
  mae_pct: number | null;
  trend?: string;
  indicators?: Record<string, number | null>;
  reasons?: { positive: string[]; negative: string[] };
};

export const api = {
  health: () => getJson<{ status: string; mode: string; live_trading_enabled: boolean }>("/api/health"),
  overview: (tf = "4h") => getJson<any>(`/api/overview?timeframe=${tf}`),
  prediction: (symbol: string, tf: string, refresh = false) =>
    getJson<Prediction>(`/api/prediction/${symbol}/${tf}${refresh ? "?refresh=true" : ""}`),
  refreshPrediction: (symbol: string, tf: string) =>
    getJson<any>(`/api/predictions/refresh?symbol=${encodeURIComponent(symbol)}&timeframe=${encodeURIComponent(tf)}`, {
      method: "POST",
    }),
  market: (symbol: string, tf: string) => getJson<any>(`/api/market/${symbol}/${tf}`),
  strategies: () => getJson<any>("/api/strategies"),
  analytics: () => getJson<any>("/api/analytics"),
  monteCarlo: () => getJson<any>("/api/monte-carlo"),
  research: () => getJson<any>("/api/research"),
  equity: (symbol: string, tf: string) => getJson<any>(`/api/equity/${symbol}/${tf}`),
  trades: (symbol: string, tf: string) => getJson<any>(`/api/trades/${symbol}/${tf}`),
  meta: () => getJson<{ symbols: string[]; timeframes: string[] }>("/api/meta"),
  predictionHistory: (params: Record<string, string | number | undefined> = {}) => {
    const q = new URLSearchParams();
    Object.entries(params).forEach(([k, v]) => {
      if (v != null && v !== "") q.set(k, String(v));
    });
    const qs = q.toString();
    return getJson<{ count: number; records: HistoryRecord[] }>(`/api/predictions/history${qs ? `?${qs}` : ""}`);
  },
  predictionStats: (symbol?: string, timeframe?: string) => {
    const q = new URLSearchParams();
    if (symbol) q.set("symbol", symbol);
    if (timeframe) q.set("timeframe", timeframe);
    const qs = q.toString();
    return getJson<any>(`/api/predictions/stats${qs ? `?${qs}` : ""}`);
  },
  predictionDetail: (id: string) => getJson<any>(`/api/predictions/${id}`),
};

export { API_BASE };
