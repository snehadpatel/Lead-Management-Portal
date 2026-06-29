"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Cpu, Activity, Info, Play } from "lucide-react";

export default function Header() {
  const pathname = usePathname();

  const navItems = [
    { href: "/demo", label: "Live Demo", icon: Play },
    { href: "/insights", label: "Model Insights", icon: Activity },
    { href: "/about", label: "About & Tech", icon: Info },
  ];

  return (
    <header className="sticky top-0 z-50 w-full border-b border-[var(--border-subtle)] bg-[var(--bg-primary)]/80 backdrop-blur-md">
      <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-4 sm:px-6 lg:px-8">
        <div className="flex items-center gap-3">
          <Link href="/" className="flex items-center gap-2 group">
            <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-[var(--accent-mint-dim)] border border-[var(--accent-mint)]/20 text-[var(--accent-mint)] group-hover:scale-105 transition-transform">
              <Cpu className="h-5 w-5" />
            </div>
            <span className="text-xl font-bold tracking-tight text-slate-100 group-hover:text-[var(--accent-mint)] transition-colors">
              Lume <span className="gradient-text">AI</span>
            </span>
          </Link>
        </div>

        <nav className="flex items-center gap-1 sm:gap-4">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = pathname === item.href;
            return (
              <Link
                key={item.href}
                href={item.href}
                className={`flex items-center gap-1.5 px-3 py-2 rounded-lg text-sm font-medium transition-all ${
                  isActive
                    ? "bg-[var(--accent-mint-dim)] text-[var(--accent-mint)] border border-[var(--accent-mint)]/20"
                    : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/40"
                }`}
              >
                <Icon className="h-4 w-4" />
                <span className="hidden sm:inline">{item.label}</span>
              </Link>
            );
          })}
          <a
            href="https://github.com/snehadpatel"
            target="_blank"
            rel="noopener noreferrer"
            className="ml-2 flex items-center gap-1.5 px-3 py-2 rounded-lg text-sm font-medium text-slate-400 hover:text-slate-200 hover:bg-slate-800/40 transition-all border border-transparent hover:border-[var(--border-subtle)]"
          >
            GitHub
          </a>
        </nav>
      </div>
    </header>
  );
}
