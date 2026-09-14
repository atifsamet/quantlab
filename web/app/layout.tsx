import type { Metadata } from "next";
import "./globals.css";
import { Sidebar } from "@/components/Sidebar";
import { DisclaimerFooter, StatusBanner } from "@/components/ui";

export const metadata: Metadata = {
  title: {
    default: "QuantLab — Quantitative Crypto Market Intelligence",
    template: "%s · QuantLab",
  },
  description:
    "Real-time crypto market analysis, quantitative research, and historical prediction evaluation powered by public market data.",
  applicationName: "QuantLab",
  keywords: [
    "quantitative finance",
    "crypto research",
    "market intelligence",
    "backtesting",
    "OKX public data",
  ],
  openGraph: {
    title: "QuantLab — Quantitative Crypto Market Intelligence",
    description:
      "Real-time crypto market analysis, quantitative research, and historical prediction evaluation powered by public market data.",
    type: "website",
    siteName: "QuantLab",
  },
  twitter: {
    card: "summary",
    title: "QuantLab — Quantitative Crypto Market Intelligence",
    description:
      "Read-only quantitative crypto research and prediction evaluation. No trade execution.",
  },
  robots: {
    index: true,
    follow: true,
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <div className="flex min-h-screen">
          <Sidebar />
          <div className="flex min-w-0 flex-1 flex-col">
            <StatusBanner />
            <main className="flex-1 px-4 py-6 pt-14 lg:px-8 lg:pt-6">{children}</main>
            <div className="px-4 pb-6 lg:px-8">
              <DisclaimerFooter />
            </div>
          </div>
        </div>
      </body>
    </html>
  );
}
