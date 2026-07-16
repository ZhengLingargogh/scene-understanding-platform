import { useEffect, useRef, useState, type ReactNode } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { hubModesForPath } from "../../config/inferenceHubModes";

interface InferenceShellProps {
  title: string;
  subtitle: string;
  badge?: string;
  children: ReactNode;
}

export default function InferenceShell({ title, subtitle, badge, children }: InferenceShellProps) {
  const location = useLocation();
  const navigate = useNavigate();
  const hubModes = hubModesForPath(location.pathname);
  const [menuOpen, setMenuOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setMenuOpen(false);
      }
    }
    if (menuOpen) {
      document.addEventListener("mousedown", handleClickOutside);
      return () => document.removeEventListener("mousedown", handleClickOutside);
    }
    return undefined;
  }, [menuOpen]);

  function handleSelectMode(path: string) {
    setMenuOpen(false);
    if (path !== location.pathname) {
      navigate(path);
    }
  }

  return (
    <div className="-mx-8 -mt-6 flex h-[calc(100vh-0px)] min-h-[640px] flex-col bg-slate-950">
      <header className="flex shrink-0 items-center justify-between border-b border-cyan-900/30 bg-slate-900/80 px-6 py-3 backdrop-blur">
        <div className="flex items-center gap-4">
          {hubModes && hubModes.length > 0 ? (
            <div className="relative" ref={menuRef}>
              <button
                type="button"
                onClick={() => setMenuOpen((open) => !open)}
                className="flex items-center gap-1 rounded-md border border-slate-600/60 px-2.5 py-1 text-xs text-slate-300 transition hover:border-cyan-600/60 hover:text-cyan-200"
                aria-expanded={menuOpen}
                aria-haspopup="listbox"
              >
                功能选择
                <span className="text-[10px] text-slate-500">{menuOpen ? "▴" : "▾"}</span>
              </button>
              {menuOpen && (
                <div
                  role="listbox"
                  className="absolute left-0 top-full z-50 mt-1 min-w-[168px] overflow-hidden rounded-md border border-slate-600/60 bg-slate-900 py-1 shadow-lg"
                >
                  {hubModes.map((mode) => {
                    const isActive = location.pathname === mode.path;
                    return (
                      <button
                        key={mode.path}
                        type="button"
                        role="option"
                        aria-selected={isActive}
                        onClick={() => handleSelectMode(mode.path)}
                        className={`block w-full px-3 py-2 text-left text-xs transition ${
                          isActive
                            ? "bg-cyan-950/60 text-cyan-200"
                            : "text-slate-300 hover:bg-slate-800 hover:text-cyan-100"
                        }`}
                      >
                        {mode.label}
                      </button>
                    );
                  })}
                </div>
              )}
            </div>
          ) : null}
          <div>
            <h1 className="text-lg font-semibold tracking-tight text-cyan-100">{title}</h1>
            <p className="text-xs text-slate-400">{subtitle}</p>
          </div>
        </div>
        {badge && (
          <div className="rounded-full border border-cyan-700/40 bg-cyan-950/50 px-3 py-1 text-[10px] uppercase tracking-widest text-cyan-300/80">
            {badge}
          </div>
        )}
      </header>
      <div className="flex min-h-0 flex-1">{children}</div>
    </div>
  );
}
