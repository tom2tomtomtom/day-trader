"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState, useEffect } from "react";
import {
  BarChart3, Settings, Settings2, TrendingUp, Globe, List,
  Zap, Target, Brain, Landmark, Activity, Cpu, PieChart,
  Menu, X
} from "lucide-react";

const navItems = [
  { href: "/", label: "Dashboard", icon: BarChart3 },
  { href: "/intel", label: "Intel", icon: Brain },
  { href: "/council", label: "Council", icon: Brain },
  { href: "/congress", label: "Congress", icon: Landmark },
  { href: "/performance", label: "Performance", icon: Activity },
  { href: "/analytics", label: "Analytics", icon: PieChart },
  { href: "/ml", label: "ML", icon: Cpu },
  { href: "/signals", label: "Signals", icon: Zap },
  { href: "/edge", label: "Edge", icon: Target },
  { href: "/positions", label: "Positions", icon: TrendingUp },
  { href: "/markets", label: "Markets", icon: Globe },
  { href: "/strategies", label: "Strategies", icon: Settings2 },
  { href: "/settings", label: "Settings", icon: Settings },
];

export function Navigation() {
  const pathname = usePathname();
  const [mobileOpen, setMobileOpen] = useState(false);

  useEffect(() => {
    setMobileOpen(false);
  }, [pathname]);

  useEffect(() => {
    document.body.style.overflow = mobileOpen ? "hidden" : "";
    return () => { document.body.style.overflow = ""; };
  }, [mobileOpen]);

  return (
    <>
      <header className="bg-black-deep border-b-2 border-red-hot sticky top-0 z-50">
        <div className="container mx-auto px-4">
          <div className="flex items-center justify-between h-14">
            <div className="flex items-center gap-8">
              <Link href="/" className="flex items-center gap-2">
                <span className="text-xl font-bold text-red-hot uppercase tracking-tight">
                  APEX // TRADER
                </span>
              </Link>
              <nav className="hidden lg:flex items-center gap-1">
                {navItems.map((item) => {
                  const isActive = pathname === item.href;
                  const Icon = item.icon;
                  return (
                    <Link
                      key={item.href}
                      href={item.href}
                      className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-bold uppercase tracking-wide transition-all ${
                        isActive
                          ? "bg-red-hot text-white border border-red-hot"
                          : "text-white-muted hover:text-orange-accent hover:bg-black-card border border-transparent"
                      }`}
                    >
                      <Icon className="w-3.5 h-3.5" />
                      <span>{item.label}</span>
                    </Link>
                  );
                })}
              </nav>
            </div>
            <button
              onClick={() => setMobileOpen(!mobileOpen)}
              className="lg:hidden p-2 text-white-muted hover:text-red-hot transition-colors"
              aria-label={mobileOpen ? "Close menu" : "Open menu"}
            >
              {mobileOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </button>
          </div>
        </div>
      </header>

      {mobileOpen && (
        <div
          className="fixed inset-0 z-40 bg-black/60 backdrop-blur-sm lg:hidden"
          onClick={() => setMobileOpen(false)}
        />
      )}

      <nav
        className={`fixed top-14 right-0 bottom-0 z-50 w-64 bg-black-deep border-l-2 border-red-hot overflow-y-auto transition-transform duration-200 ease-out lg:hidden ${
          mobileOpen ? "translate-x-0" : "translate-x-full"
        }`}
      >
        <div className="p-3 space-y-1">
          {navItems.map((item) => {
            const isActive = pathname === item.href;
            const Icon = item.icon;
            return (
              <Link
                key={item.href}
                href={item.href}
                className={`flex items-center gap-3 px-4 py-3 text-xs font-bold uppercase tracking-wide transition-all ${
                  isActive
                    ? "bg-red-hot text-white"
                    : "text-white-muted hover:text-orange-accent hover:bg-black-card"
                }`}
              >
                <Icon className="w-4 h-4" />
                <span>{item.label}</span>
              </Link>
            );
          })}
        </div>
      </nav>
    </>
  );
}
