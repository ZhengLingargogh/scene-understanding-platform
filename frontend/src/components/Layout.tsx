import { NavLink, Outlet, useLocation } from "react-router-dom";
import { useEffect, useState } from "react";
import { checkHealth } from "../api/client";
import "./Layout.css";

const navItems = [
  { to: "/scenes", label: "场景管理" },
  { to: "/models", label: "模型管理" },
  { to: "/inference", label: "单图推理" },
  { to: "/scene-inference", label: "场景推理" },
];

export default function Layout() {
  const [health, setHealth] = useState<string>("checking...");
  const location = useLocation();

  useEffect(() => {
    checkHealth()
      .then((data) => setHealth(`${data.status} (v${data.version})`))
      .catch(() => setHealth("offline"));
  }, []);

  return (
    <div className="layout">
      <aside className="sidebar">
        <h1 className="brand">Scene Understanding</h1>
        <nav>
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) =>
                isActive || location.pathname.startsWith(`${item.to}/`)
                  ? "nav-link active"
                  : "nav-link"
              }
            >
              {item.label}
            </NavLink>
          ))}
        </nav>
        <p className="health">API: {health}</p>
      </aside>
      <main className="content">
        <Outlet />
      </main>
    </div>
  );
}
