import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../lib/api";

interface Playlist {
  id: string;
  name: string;
  image: string | null;
  total_tracks: number;
  owner: string;
}

interface PlaylistsResult {
  playlists: Playlist[];
}

interface RadarSettings {
  auto_refresh: boolean;
  source_playlist_ids: string[];
  include_on_repeat: boolean;
  last_spotify_playlist_id: string | null;
}

interface RadarResult {
  playlist_url: string;
  playlist_id: string;
  playlist_name: string;
  total_tracks: number;
  inspiration_count: number;
  new_discoveries_count: number;
  auto_refresh: boolean;
}

type Step = "select" | "generating" | "done";

export default function OffTheRadarPage({ onLogout: _onLogout }: { onLogout: () => void }) {
  const navigate = useNavigate();
  const token = localStorage.getItem("token") || "";
  const [step, setStep] = useState<Step>("select");
  const [playlists, setPlaylists] = useState<Playlist[]>([]);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [includeOnRepeat, setIncludeOnRepeat] = useState(true);
  const [autoRefresh, setAutoRefresh] = useState(false);
  const [loading, setLoading] = useState(true);
  const [savingRefresh, setSavingRefresh] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<RadarResult | null>(null);
  const [generatingStep, setGeneratingStep] = useState(0);

  const generatingSteps = [
    { emoji: "🎵", text: "Loading your taste sources..." },
    { emoji: "🧠", text: "Sol is mapping your taste profile..." },
    { emoji: "🛰️", text: "Luna is finding overlooked tracks..." },
    { emoji: "🔍", text: "Verifying fresh tracks on Spotify..." },
    { emoji: "🕵️", text: "Updating your Off the Radar playlist..." },
  ];

  useEffect(() => {
    if (step !== "generating") return;
    const interval = window.setInterval(() => {
      setGeneratingStep((current) => Math.min(current + 1, generatingSteps.length - 1));
    }, 4000);
    return () => window.clearInterval(interval);
  }, [step, generatingSteps.length]);

  useEffect(() => {
    if (!token) return;
    setLoading(true);
    Promise.all([
      api<PlaylistsResult>("/my-playlists", { token }),
      api<RadarSettings>("/off-the-radar/settings", { token }).catch(() => null),
    ])
      .then(([playlistData, settings]) => {
        setPlaylists(playlistData.playlists);
        if (settings) {
          setAutoRefresh(settings.auto_refresh);
          setIncludeOnRepeat(settings.include_on_repeat);
          setSelectedIds(new Set(settings.source_playlist_ids));
        }
      })
      .catch((requestError: unknown) => {
        setError(requestError instanceof Error ? requestError.message : "Could not load playlists");
      })
      .finally(() => setLoading(false));
  }, [token]);

  const canGenerate = includeOnRepeat || selectedIds.size > 0;

  const togglePlaylist = (id: string) => {
    setSelectedIds((current) => {
      const next = new Set(current);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  };

  const saveAutoRefresh = async (enabled: boolean) => {
    if (!canGenerate) {
      setError("Enable On Repeat or select at least one playlist first.");
      return;
    }
    setSavingRefresh(true);
    try {
      await api("/off-the-radar/auto-refresh", {
        method: "PUT",
        token,
        body: {
          auto_refresh: enabled,
          source_playlist_ids: Array.from(selectedIds),
          include_on_repeat: includeOnRepeat,
        },
      });
      setAutoRefresh(enabled);
    } catch (requestError: unknown) {
      setError(requestError instanceof Error ? requestError.message : "Could not save auto-refresh");
    } finally {
      setSavingRefresh(false);
    }
  };

  const generate = async () => {
    if (!canGenerate) return;
    setError("");
    setGeneratingStep(0);
    setStep("generating");
    try {
      const data = await api<RadarResult>("/off-the-radar/generate", {
        method: "POST",
        token,
        body: {
          source_playlist_ids: Array.from(selectedIds),
          include_on_repeat: includeOnRepeat,
        },
      });
      setResult(data);
      setAutoRefresh(data.auto_refresh);
      setStep("done");
    } catch (requestError: unknown) {
      setError(requestError instanceof Error ? requestError.message : "Could not create Off the Radar");
      setStep("select");
    }
  };

  return (
    <div className="min-h-screen px-4 py-8">
      <div className="mx-auto max-w-2xl">
        <header className="mb-8 flex items-center gap-4">
          <button onClick={() => navigate("/")} className="rounded-lg p-2 text-gray-400 ring-1 ring-white/10 transition hover:text-white hover:ring-white/20" aria-label="Back to home">
            <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M15 19l-7-7 7-7" /></svg>
          </button>
          <div className="flex-1">
            <h1 className="text-2xl font-bold tracking-tight"><span className="text-indigo-400">Off the</span> <span className="text-gray-100">Radar</span></h1>
            <p className="text-sm text-gray-400">30 fresh paths beyond your current rotation</p>
          </div>
          <span className="text-3xl">🕵️</span>
        </header>

        {error && <div className="mb-6 rounded-lg bg-red-500/10 px-4 py-3 text-sm text-red-400 ring-1 ring-red-500/20">{error}</div>}

        {step === "select" && <>
          <section className="mb-6 rounded-2xl bg-gradient-to-br from-indigo-500/15 to-fuchsia-500/5 p-5 ring-1 ring-indigo-500/25">
            <h2 className="mb-2 text-sm font-semibold text-indigo-300">How it works</h2>
            <ul className="space-y-1.5 text-xs text-gray-400">
              <li>🧠 GPT-6.1 Sol distills your music taste into a compact profile.</li>
              <li>🛰️ GPT-6 Luna searches for 30 less-obvious matching discoveries.</li>
              <li>🛡️ The last 30 days of accepted tracks are excluded again by Spotify URI.</li>
              <li>⏰ Optional auto-refresh updates the same private playlist every day at 05:00.</li>
            </ul>
          </section>

          <section className="mb-6">
            <h2 className="mb-3 text-sm font-semibold text-gray-300">Choose your taste sources</h2>
            <label className={`mb-3 flex cursor-pointer items-center gap-3 rounded-xl p-3 transition ${includeOnRepeat ? "bg-indigo-500/15 ring-1 ring-indigo-500/35" : "bg-white/5 ring-1 ring-white/10 hover:bg-white/10"}`}>
              <input type="checkbox" checked={includeOnRepeat} onChange={(event) => setIncludeOnRepeat(event.target.checked)} className="h-5 w-5 accent-indigo-500" />
              <div className="min-w-0 flex-1"><p className="text-sm font-medium text-gray-200">🔁 Use On Repeat as a taste signal</p><p className="mt-0.5 text-xs text-gray-500">Those songs are analysed but can never be added to Off the Radar.</p></div>
            </label>

            {loading ? <div className="space-y-2">{[0, 1, 2, 3].map((item) => <div key={item} className="h-[72px] animate-pulse rounded-xl bg-white/5" />)}</div> :
              <div className="max-h-[400px] space-y-2 overflow-y-auto rounded-xl pr-1">
                {playlists.map((playlist) => {
                  const selected = selectedIds.has(playlist.id);
                  return <button key={playlist.id} onClick={() => togglePlaylist(playlist.id)} className={`flex w-full items-center gap-3 rounded-xl p-3 text-left transition ${selected ? "bg-indigo-500/15 ring-1 ring-indigo-500/35" : "bg-white/5 hover:bg-white/10"}`}>
                    <span className={`flex h-5 w-5 flex-shrink-0 items-center justify-center rounded ${selected ? "bg-indigo-500 text-white" : "bg-white/10 ring-1 ring-white/20"}`}>{selected && "✓"}</span>
                    <div className="h-12 w-12 flex-shrink-0 overflow-hidden rounded-lg bg-gray-800">{playlist.image ? <img src={playlist.image} alt="" className="h-full w-full object-cover" /> : <span className="flex h-full items-center justify-center">🎵</span>}</div>
                    <div className="min-w-0 flex-1"><p className="truncate text-sm font-medium text-gray-200">{playlist.name}</p><p className="truncate text-xs text-gray-500">{playlist.owner}{playlist.total_tracks > 0 && ` · ${playlist.total_tracks} songs`}</p></div>
                  </button>;
                })}
              </div>}
            {selectedIds.size > 0 && <p className="mt-2 text-xs text-indigo-300/80">{selectedIds.size} playlist{selectedIds.size > 1 ? "s" : ""} selected</p>}
          </section>

          <button type="button" onClick={() => saveAutoRefresh(!autoRefresh)} disabled={savingRefresh} className={`mb-6 flex w-full items-center gap-4 rounded-2xl p-4 text-left transition ${autoRefresh ? "bg-indigo-500/10 ring-1 ring-indigo-500/35" : "bg-white/5 ring-1 ring-white/10 hover:bg-white/10"}`}>
            <span className={`relative h-6 w-11 flex-shrink-0 rounded-full ${autoRefresh ? "bg-indigo-500" : "bg-gray-300"}`}><span className={`absolute top-0.5 h-5 w-5 rounded-full bg-white shadow transition-transform ${autoRefresh ? "translate-x-[22px]" : "translate-x-0.5"}`} /></span>
            <span><span className="block text-sm font-medium text-gray-200">⏰ Daily auto-refresh</span><span className="text-xs text-gray-500">Refresh the same playlist every morning at 05:00.</span></span>
          </button>

          <button onClick={generate} disabled={!canGenerate} className={`flex w-full items-center justify-center gap-3 rounded-2xl px-6 py-4 text-base font-bold shadow-lg transition ${canGenerate ? "bg-gradient-to-r from-indigo-500 to-fuchsia-500 text-white shadow-indigo-500/25 hover:brightness-110" : "cursor-not-allowed bg-gray-700 text-gray-500"}`}>
            <span className="text-xl">🕵️</span> Discover 30 tracks
          </button>
        </>}

        {step === "generating" && <div className="flex flex-col items-center py-16"><div className="mb-8 animate-bounce text-6xl">🛰️</div><div className="w-full max-w-sm space-y-4">{generatingSteps.map((item, index) => { const active = index === generatingStep; const done = index < generatingStep; return <div key={item.text} className={`flex items-center gap-3 rounded-xl px-4 py-3 ${active ? "bg-indigo-500/15 ring-1 ring-indigo-500/30" : done ? "bg-green-500/10 ring-1 ring-green-500/20" : "bg-white/5 opacity-40"}`}><span>{done ? "✅" : item.emoji}</span><span className={`text-sm ${active ? "text-indigo-200" : done ? "text-green-400" : "text-gray-500"}`}>{item.text}</span>{active && <span className="ml-auto h-4 w-4 animate-spin rounded-full border-2 border-indigo-400 border-t-transparent" />}</div>; })}</div><p className="mt-8 text-xs text-gray-500">This can take up to two minutes while candidates are verified.</p></div>}

        {step === "done" && result && <div className="flex flex-col items-center py-8"><div className="mb-6 text-6xl">🎉</div><h2 className="mb-2 text-xl font-bold text-gray-100">Your discoveries are ready!</h2><p className="mb-8 text-center text-sm text-gray-400">{result.playlist_name}</p><div className="mb-8 grid w-full max-w-sm grid-cols-3 gap-3"><div className="rounded-xl bg-indigo-500/10 p-4 text-center ring-1 ring-indigo-500/20"><p className="text-2xl font-bold text-indigo-300">{result.total_tracks}</p><p className="mt-1 text-[10px] text-gray-400">Fresh tracks</p></div><div className="rounded-xl bg-fuchsia-500/10 p-4 text-center ring-1 ring-fuchsia-500/20"><p className="text-2xl font-bold text-fuchsia-300">30</p><p className="mt-1 text-[10px] text-gray-400">Days protected</p></div><div className="rounded-xl bg-sky-500/10 p-4 text-center ring-1 ring-sky-500/20"><p className="text-2xl font-bold text-sky-300">{result.inspiration_count}</p><p className="mt-1 text-[10px] text-gray-400">Taste signals</p></div></div><div className="mb-8 w-full overflow-hidden rounded-2xl"><iframe src={`https://open.spotify.com/embed/playlist/${result.playlist_id}?utm_source=generator&theme=0`} width="100%" height="352" frameBorder="0" allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture" loading="lazy" style={{ border: "none", borderRadius: "16px" }} /></div>{autoRefresh && <div className="mb-6 w-full rounded-xl bg-indigo-500/10 p-4 text-center text-sm text-indigo-200 ring-1 ring-indigo-500/20">⏰ Auto-refresh is active for 05:00 every morning.</div>}<div className="flex w-full max-w-sm flex-col gap-3"><a href={result.playlist_url} target="_blank" rel="noreferrer" className="rounded-2xl bg-green-500 px-6 py-3.5 text-center text-sm font-bold text-gray-950 hover:bg-green-400">Open in Spotify</a><button onClick={() => { setStep("select"); setResult(null); }} className="rounded-2xl px-6 py-3.5 text-sm text-gray-400 ring-1 ring-white/10 hover:text-white">🔄 Discover another mix</button></div></div>}
      </div>
    </div>
  );
}
