import type { Metadata } from "next";
import "./globals.css";
import { Navbar } from "@/components/Navbar";

export const metadata: Metadata = {
  title: "SpamGuard XAI | Email Spam Detection & Explainability",
  description:
    "Audited, explainable machine learning system for email spam detection on the SpamAssassin corpus with exact token attributions and deletion faithfulness evaluation.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="min-h-screen flex flex-col bg-background text-foreground antialiased selection:bg-primary/20">
        <Navbar />
        <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {children}
        </main>
        <footer className="border-t border-border py-6 bg-card/50 text-xs text-muted-foreground text-center">
          <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-2">
            <div>
              SpamAssassin Corpus Evaluation &bull; Exact Linear &amp; Naive Bayes Attributions &bull; Faithfulness Deletion Benchmarks
            </div>
            <div className="font-mono text-[11px]">
              Grouped Stratified Split (Seed 42) &bull; Verified No Data Leakage
            </div>
          </div>
        </footer>
      </body>
    </html>
  );
}
