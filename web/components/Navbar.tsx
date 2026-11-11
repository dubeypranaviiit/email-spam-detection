"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { ShieldCheck, Moon, Sun, Activity, Terminal, Sparkles, BookOpen } from "lucide-react";
import { api } from "@/lib/api";

export function Navbar() {
  const pathname = usePathname();
  const [apiOnline, setApiOnline] = useState<boolean | null>(null);
  const [isDark, setIsDark] = useState<boolean>(false);

  useEffect(() => {
    // Check dark mode preference
    if (typeof window !== "undefined") {
      const darkModeSaved = localStorage.getItem("theme") === "dark";
      setIsDark(darkModeSaved);
      if (darkModeSaved) {
        document.documentElement.classList.add("dark");
      }
    }

    // Ping API health
    api
      .getHealth()
      .then(() => setApiOnline(true))
      .catch(() => setApiOnline(false));
  }, []);

  const toggleTheme = () => {
    const nextDark = !isDark;
    setIsDark(nextDark);
    if (nextDark) {
      document.documentElement.classList.add("dark");
      localStorage.setItem("theme", "dark");
    } else {
      document.documentElement.classList.remove("dark");
      localStorage.setItem("theme", "light");
    }
  };

  const navLinks = [
    { href: "/", label: "Overview" },
    { href: "/detector", label: "Live Detector" },
    { href: "/explainability", label: "Explainability" },
  ];

  return (
    <header className="sticky top-0 z-40 w-full border-b border-border bg-card/80 backdrop-blur-md">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        <div className="flex items-center space-x-8">
          <Link href="/" className="flex items-center space-x-2 text-foreground font-semibold text-lg">
            <ShieldCheck className="w-5 h-5 text-primary" />
            <span>SpamGuard <span className="text-xs font-mono uppercase bg-accent text-accent-foreground px-1.5 py-0.5 rounded">XAI</span></span>
          </Link>
          <nav className="hidden md:flex space-x-1" aria-label="Main Navigation">
            {navLinks.map((link) => {
              const active = pathname === link.href;
              return (
                <Link
                  key={link.href}
                  href={link.href}
                  className={`px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
                    active
                      ? "bg-secondary text-foreground font-semibold shadow-sm"
                      : "text-muted-foreground hover:text-foreground hover:bg-secondary/50"
                  }`}
                >
                  {link.label}
                </Link>
              );
            })}
          </nav>
        </div>

        <div className="flex items-center space-x-3">
          {/* API Status badge */}
          <div
            className="flex items-center space-x-1.5 text-xs px-2.5 py-1 rounded-full border border-border bg-secondary/50 text-muted-foreground"
            title={apiOnline ? "FastAPI backend reachable" : "FastAPI backend offline"}
          >
            <span
              className={`w-2 h-2 rounded-full ${
                apiOnline === true
                  ? "bg-emerald-500"
                  : apiOnline === false
                  ? "bg-rose-500 animate-pulse"
                  : "bg-amber-400"
              }`}
            />
            <span className="hidden sm:inline">
              {apiOnline === true ? "API Online" : apiOnline === false ? "API Disconnected" : "Checking..."}
            </span>
          </div>

          {/* Theme toggle */}
          <button
            onClick={toggleTheme}
            aria-label="Toggle theme"
            className="p-2 rounded-md text-muted-foreground hover:text-foreground hover:bg-secondary transition-colors"
          >
            {isDark ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
          </button>
        </div>
      </div>
    </header>
  );
}
