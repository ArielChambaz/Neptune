import { useEffect, useState } from "react";
import Alert from "./components/Alert";
import VideoPlayer from "./components/VideoPlayer";
import Logo from "./components/Logo";

export default function App() {
  const [alerts, setAlerts] = useState<
    { id: number; type: "red" | "yellow" | "green"; text: string }[]
  >([]);

  useEffect(() => {
    const es = new EventSource("http://localhost:5000/api/alerts");

    es.onmessage = (ev) => {
      const incoming = JSON.parse(ev.data);
      setAlerts((prev) => [...incoming, ...prev]);
    };

    es.onerror = (e) => {
      console.error("SSE error :", e);
      es.close();
    };
    return () => es.close();
  }, []);

  const handleDelete = (id: number) =>
    setAlerts((prev) => prev.filter((a) => a.id !== id));

  return (
    <main className="flex flex-col pt-4 pb-4 gap-8 px-8">
      <header className="flex flex-col items-center gap-9">
        <Logo />
      </header>

      <div className="flex flex-row w-full gap-8 items-stretch">
        {/* Video */}
        <section className="flex-grow">
          <VideoPlayer />
        </section>

        {/* Alerts Panel */}
        <aside className="w-[400px] relative">
          <div className="absolute inset-0 flex flex-col gap-4 bg-white/5 p-4 rounded-lg overflow-y-auto">
            <button
              onClick={() => setAlerts([])}
              className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
            >
              Clear Alerts
            </button>
            {alerts.map((a) => (
              <Alert key={a.id} {...a} onDelete={handleDelete} />
            ))}
          </div>
        </aside>
      </div>
    </main>
  );
}
