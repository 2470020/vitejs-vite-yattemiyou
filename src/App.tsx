import { useEffect, useRef } from "react";

const STLITE = "https://cdn.jsdelivr.net/npm/@stlite/browser@0.83.0/build";

// stlite の mount() が返すオブジェクト（unmount だけ使う）
type StliteController = { unmount?: () => void };

export default function App() {
  const rootRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let cancelled = false;
    let controller: StliteController | undefined;

    (async () => {
      // stlite 用の CSS を一度だけ読み込む
      if (!document.getElementById("stlite-css")) {
        const link = document.createElement("link");
        link.id = "stlite-css";
        link.rel = "stylesheet";
        link.href = `${STLITE}/stlite.css`;
        document.head.appendChild(link);
      }

      // npm に入れず、CDN から直接読み込む（StackBlitz で軽く動かすため）
      const mod = await import(/* @vite-ignore */ `${STLITE}/stlite.js`);
      if (cancelled || !rootRef.current) return;

      const abs = (name: string) => new URL(name, window.location.href).href;

      controller = mod.mount(
        {
          entrypoint: "app_analysis.py",
          requirements: ["plotly", "openpyxl"],
          files: {
            "app_analysis.py": { url: abs("app_analysis.py") },
            "sa_core.py": { url: abs("sa_core.py") },
          },
        },
        rootRef.current,
      ) as StliteController;
    })();

    return () => {
      cancelled = true;
      controller?.unmount?.();
    };
  }, []);

  return <div ref={rootRef} style={{ minHeight: "100vh" }} />;
}
