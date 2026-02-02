"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { BarChart3, Settings, TrendingUp, Globe, List, Zap, Target } from "lucide-react";

const navItems = [
  { href: "/", label: "Dashboard", icon: BarChart3 },
  { href: "/signals", label: "Signals", icon: Zap },
  { href: "/edge", label: "Edge", icon: Target },
  { href: "/watchlist", label: "Watchlist", icon: List },
  { href: "/positions", label: "Positions", icon: TrendingUp },
  { href: "/markets", label: "Markets", icon: Globe },
  { href: "/settings", label: "Settings", icon: Settings },
];

export function Navigation() {
  const pathname = usePathname();

  return (
    <header className="bg-zinc-900 border-b border-zinc-800 sticky top-0 z-50">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center gap-8">
            <Link href="/" className="flex items-center gap-2">
              <span className="text-2xl">📈</span>
              <span className="font-bold text-xl">Market Trader</span>
            </Link>
            <nav className="hidden md:flex items-center gap-1">
              {navItems.map((item) => {
                const isActive = pathname === item.href;
                const Icon = item.icon;
                return (
                  <Link
                    key={item.href}
                    href={item.href}
                    className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-colors ${
                      isActive
                        ? "bg-emerald-600 text-white"
                        : "text-zinc-400 hover:text-white hover:bg-zinc-800"
                    }`}
                  >
                    <Icon className="w-4 h-4" />
                    <span>{item.label}</span>
                  </Link>
                );
              })}
            </nav>
          </div>
          <div className="flex items-center gap-4">
            <div className="text-sm text-zinc-400">
              <span id="clock"></span>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}
