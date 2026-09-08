import { useState, useEffect } from "react";
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

interface GymPlaylistResult {
    playlist_url: string;
    playlist_id: string;
    playlist_name: string;
    total_tracks: number;
    inspiration_count: number;
    new_discoveries_count: number;
    auto_refresh: boolean;
}

interface GymSettings {
    auto_refresh: boolean;
    source_playlist_ids: string[];
    include_on_repeat: boolean;
    last_spotify_playlist_id: string | null;
}

type Step = "select" | "generating" | "done";

export default function GymPlaylistPage({ onLogout: _onLogout }: { onLogout: () => void }) {
    const navigate = useNavigate();
    const token = localStorage.getItem("token") || "";

    const [step, setStep] = useState<Step>("select");
    const [playlists, setPlaylists] = useState<Playlist[]>([]);
    const [loadingPlaylists, setLoadingPlaylists] = useState(true);
    const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
    const [includeOnRepeat, setIncludeOnRepeat] = useState(false);
    const [autoRefresh, setAutoRefresh] = useState(false);
    const [error, setError] = useState("");
    const [result, setResult] = useState<GymPlaylistResult | null>(null);
    const [generatingStep, setGeneratingStep] = useState(0);

    const generatingSteps = [
        { emoji: "🎵", text: "Loading songs from your playlists..." },
        { emoji: "🤖", text: "AI is creating your gym mix..." },
        { emoji: "🔍", text: "Searching songs on Spotify..." },
        { emoji: "🗑️", text: "Deleting old gym playlist..." },
        { emoji: "🏋️", text: "Creating your new gym playlist..." },
    ];

    // Auto-cycle through generating steps
    useEffect(() => {
        if (step !== "generating") return;
        const interval = setInterval(() => {
            setGeneratingStep((prev) =>
                prev < generatingSteps.length - 1 ? prev + 1 : prev
            );
        }, 4000);
        return () => clearInterval(interval);
    }, [step, generatingSteps.length]);

    // Load playlists + existing settings on mount
    useEffect(() => {
        if (!token) return;
        setLoadingPlaylists(true);

        Promise.all([
            api<PlaylistsResult>("/my-playlists", { method: "GET", token }),
            api<GymSettings>("/gym-playlist/settings", { method: "GET", token }).catch(() => null),
        ])
            .then(([playlistData, settingsData]) => {
                setPlaylists(playlistData.playlists);
                if (settingsData) {
                    setAutoRefresh(settingsData.auto_refresh);
                    setIncludeOnRepeat(settingsData.include_on_repeat);
                    if (settingsData.source_playlist_ids.length > 0) {
                        setSelectedIds(new Set(settingsData.source_playlist_ids));
                    }
                }
            })
            .catch((err) =>
                setError(err instanceof Error ? err.message : "Could not load playlists")
            )
            .finally(() => setLoadingPlaylists(false));
    }, [token]);

    const togglePlaylist = (id: string) => {
        setSelectedIds((prev) => {
            const next = new Set(prev);
            if (next.has(id)) next.delete(id);
            else next.add(id);
            return next;
        });
    };

    const handleToggleAutoRefresh = async () => {
        const newValue = !autoRefresh;
        setAutoRefresh(newValue);
        try {
            await api("/gym-playlist/auto-refresh", {
                method: "PUT",
                body: { auto_refresh: newValue },
                token,
            });
        } catch {
            setAutoRefresh(!newValue); // revert on error
        }
    };

    const handleGenerate = async () => {
        setError("");
        setStep("generating");
        setGeneratingStep(0);

        try {
            const data = await api<GymPlaylistResult>("/gym-playlist/generate", {
                method: "POST",
                body: {
                    source_playlist_ids: Array.from(selectedIds),
                    include_on_repeat: includeOnRepeat,
                },
                token,
            });
            setResult(data);
            setAutoRefresh(data.auto_refresh);
            setStep("done");
        } catch (err: unknown) {
            setError(err instanceof Error ? err.message : "Something went wrong");
            setStep("select");
        }
    };

    const handleReset = () => {
        setStep("select");
        setResult(null);
        setGeneratingStep(0);
    };

    return (
        <div className="min-h-screen px-4 py-8">
            <div className="mx-auto max-w-2xl">
                {/* Header */}
                <div className="mb-8 flex items-center gap-4">
                    <button
                        onClick={() => navigate("/")}
                        className="rounded-lg p-2 text-gray-400 ring-1 ring-white/10 transition hover:text-white hover:ring-white/20"
                    >
                        <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                            <path strokeLinecap="round" strokeLinejoin="round" d="M15 19l-7-7 7-7" />
                        </svg>
                    </button>
                    <div className="flex-1">
                        <h1 className="text-2xl font-bold tracking-tight">
                            <span className="text-red-400">Gym</span>{" "}
                            <span className="text-gray-100">Playlist</span>
                        </h1>
                        <p className="text-sm text-gray-400">
                            Your personal workout mix with AI power
                        </p>
                    </div>
                    <span className="text-3xl">🏋️</span>
                </div>

                {/* Error */}
                {error && (
                    <div className="mb-6 rounded-lg bg-red-500/10 px-4 py-3 text-sm text-red-400 ring-1 ring-red-500/20">
                        {error}
                    </div>
                )}

                {/* ── STEP 1: Select playlists ── */}
                {step === "select" && (
                    <>
                        {/* Info card */}
                        <div className="mb-6 rounded-2xl bg-gradient-to-br from-red-500/10 to-orange-500/5 p-5 ring-1 ring-red-500/20">
                            <h2 className="mb-2 text-sm font-semibold text-red-400">
                                How it works
                            </h2>
                            <ul className="space-y-1.5 text-xs text-gray-400">
                                <li className="flex items-start gap-2">
                                    <span className="mt-0.5 text-red-400">🎵</span>
                                    Choose playlists as inspiration for your music taste
                                </li>
                                <li className="flex items-start gap-2">
                                    <span className="mt-0.5 text-red-400">🤖</span>
                                    AI creates 40 personal gym tracks, staying close to your current taste
                                </li>
                                <li className="flex items-start gap-2">
                                    <span className="mt-0.5 text-red-400">🔄</span>
                                    Old gym playlist is automatically deleted & replaced
                                </li>
                                <li className="flex items-start gap-2">
                                    <span className="mt-0.5 text-red-400">⏰</span>
                                    Optional: Auto-regenerate daily at 3 AM
                                </li>
                            </ul>
                        </div>

                        {/* Playlist selection */}
                        <div className="mb-6">
                            <h3 className="mb-3 text-sm font-semibold text-gray-300">
                                🎵 Choose playlists as inspiration
                            </h3>

                            <label className={`mb-3 flex cursor-pointer items-center gap-3 rounded-xl p-3 transition-all ${includeOnRepeat
                                ? "bg-red-500/15 ring-1 ring-red-500/30"
                                : "bg-white/5 ring-1 ring-white/10 hover:bg-white/10"
                                }`}>
                                <input
                                    type="checkbox"
                                    checked={includeOnRepeat}
                                    onChange={(event) => setIncludeOnRepeat(event.target.checked)}
                                    className="h-5 w-5 cursor-pointer accent-red-500"
                                />
                                <div className="min-w-0 flex-1">
                                    <p className="text-sm font-medium text-gray-200">
                                        🔁 Include your On Repeat songs
                                    </p>
                                    <p className="mt-0.5 text-xs text-gray-500">
                                        Use your current top tracks alongside the selected playlists.
                                    </p>
                                </div>
                            </label>

                            {loadingPlaylists ? (
                                <div className="space-y-2">
                                    {[...Array(5)].map((_, i) => (
                                        <div key={i} className="flex animate-pulse items-center gap-3 rounded-xl bg-white/5 p-3">
                                            <div className="h-12 w-12 rounded-lg bg-gray-700" />
                                            <div className="flex-1 space-y-2">
                                                <div className="h-4 w-3/4 rounded bg-gray-700" />
                                                <div className="h-3 w-1/2 rounded bg-gray-700" />
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            ) : playlists.length === 0 ? (
                                <div className="rounded-xl bg-white/5 p-6 text-center">
                                    <span className="mb-2 block text-3xl">🎵</span>
                                    <p className="text-sm text-gray-400">
                                        No playlists found.
                                    </p>
                                    <p className="mt-1 text-xs text-gray-500">
                                        Create playlists on Spotify so they show up here.
                                    </p>
                                </div>
                            ) : (
                                <div className="max-h-[400px] space-y-2 overflow-y-auto rounded-xl pr-1">
                                    {playlists.map((pl) => {
                                        const isSelected = selectedIds.has(pl.id);
                                        return (
                                            <button
                                                key={pl.id}
                                                onClick={() => togglePlaylist(pl.id)}
                                                className={`flex w-full items-center gap-3 rounded-xl p-3 text-left transition-all ${isSelected
                                                    ? "bg-red-500/15 ring-1 ring-red-500/30"
                                                    : "bg-white/5 hover:bg-white/10"
                                                    }`}
                                            >
                                                {/* Checkbox */}
                                                <div
                                                    className={`flex h-5 w-5 flex-shrink-0 items-center justify-center rounded transition-all ${isSelected
                                                        ? "bg-red-500 text-white"
                                                        : "bg-white/10 ring-1 ring-white/20"
                                                        }`}
                                                >
                                                    {isSelected && (
                                                        <svg className="h-3 w-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={3}>
                                                            <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                                                        </svg>
                                                    )}
                                                </div>

                                                {/* Cover */}
                                                <div className="h-12 w-12 flex-shrink-0 overflow-hidden rounded-lg bg-gray-800">
                                                    {pl.image ? (
                                                        <img
                                                            src={pl.image}
                                                            alt={pl.name}
                                                            className="h-full w-full object-cover"
                                                        />
                                                    ) : (
                                                        <div className="flex h-full w-full items-center justify-center text-lg text-gray-600">
                                                            🎵
                                                        </div>
                                                    )}
                                                </div>

                                                {/* Info */}
                                                <div className="min-w-0 flex-1">
                                                    <p className="truncate text-sm font-medium text-gray-200">
                                                        {pl.name}
                                                    </p>
                                                    <p className="truncate text-xs text-gray-500">
                                                        {pl.owner}
                                                        {pl.total_tracks > 0 && ` · ${pl.total_tracks} Songs`}
                                                    </p>
                                                </div>
                                            </button>
                                        );
                                    })}
                                </div>
                            )}

                            {selectedIds.size > 0 && (
                                <p className="mt-2 text-xs text-red-400/70">
                                    {selectedIds.size} playlist{selectedIds.size > 1 ? "s" : ""} selected
                                </p>
                            )}
                        </div>

                        {/* Auto-refresh toggle */}
                        <div className="mb-6">
                            <button
                                onClick={handleToggleAutoRefresh}
                                className={`flex w-full items-center gap-4 rounded-2xl p-4 text-left transition-all ${autoRefresh
                                    ? "bg-red-500/10 ring-1 ring-red-500/30"
                                    : "bg-white/5 ring-1 ring-white/10 hover:bg-white/10"
                                    }`}
                            >
                                {/* Toggle switch */}
                                <div
                                    className={`relative h-6 w-11 flex-shrink-0 rounded-full transition-colors ${autoRefresh ? "bg-red-500" : "bg-gray-600"
                                        }`}
                                >
                                    <div
                                        className={`absolute top-0.5 h-5 w-5 rounded-full bg-white shadow transition-transform ${autoRefresh ? "translate-x-[22px]" : "translate-x-0.5"
                                            }`}
                                    />
                                </div>

                                <div className="flex-1">
                                    <p className="text-sm font-medium text-gray-200">
                                        ⏰ Daily Auto-Refresh
                                    </p>
                                    <p className="text-xs text-gray-500">
                                        Every day at 3:00 AM your gym playlist
                                        is automatically recreated with fresh songs
                                    </p>
                                </div>
                            </button>
                        </div>

                        {/* Generate button */}
                        <button
                            onClick={handleGenerate}
                            disabled={selectedIds.size === 0}
                            className={`flex w-full items-center justify-center gap-3 rounded-2xl px-6 py-4 text-base font-bold shadow-lg transition ${selectedIds.size === 0
                                ? "cursor-not-allowed bg-gray-700 text-gray-500"
                                : "bg-gradient-to-r from-red-500 to-orange-500 text-white shadow-red-500/25 hover:shadow-red-500/40 hover:brightness-110"
                                }`}
                        >
                            <span className="text-xl">🏋️</span>
                            Generate Gym Playlist
                        </button>
                    </>
                )}

                {/* ── STEP 2: Generating ── */}
                {step === "generating" && (
                    <div className="flex flex-col items-center py-16">
                        <div className="mb-8 text-6xl animate-bounce">💪</div>

                        <div className="w-full max-w-sm space-y-4">
                            {generatingSteps.map((s, i) => {
                                const isActive = i === generatingStep;
                                const isDone = i < generatingStep;
                                return (
                                    <div
                                        key={i}
                                        className={`flex items-center gap-3 rounded-xl px-4 py-3 transition-all duration-500 ${isActive
                                            ? "bg-red-500/15 ring-1 ring-red-500/30"
                                            : isDone
                                                ? "bg-green-500/10 ring-1 ring-green-500/20"
                                                : "bg-white/5 opacity-40"
                                            }`}
                                    >
                                        <span className="text-lg">
                                            {isDone ? "✅" : s.emoji}
                                        </span>
                                        <span
                                            className={`text-sm ${isActive
                                                ? "font-medium text-red-300"
                                                : isDone
                                                    ? "text-green-400"
                                                    : "text-gray-500"
                                                }`}
                                        >
                                            {s.text}
                                        </span>
                                        {isActive && (
                                            <div className="ml-auto h-4 w-4 animate-spin rounded-full border-2 border-red-400 border-t-transparent" />
                                        )}
                                    </div>
                                );
                            })}
                        </div>

                        <p className="mt-8 text-xs text-gray-500">
                            This may take up to 60 seconds…
                        </p>
                    </div>
                )}

                {/* ── STEP 3: Done ── */}
                {step === "done" && result && (
                    <div className="flex flex-col items-center py-8">
                        <div className="mb-6 text-6xl">🎉</div>

                        <h2 className="mb-2 text-xl font-bold text-gray-100">
                            Your Gym Playlist is ready!
                        </h2>
                        <p className="mb-8 text-center text-sm text-gray-400">
                            {result.playlist_name}
                        </p>

                        {/* Stats */}
                        <div className="mb-8 grid w-full max-w-sm grid-cols-3 gap-3">
                            <div className="rounded-xl bg-red-500/10 p-4 text-center ring-1 ring-red-500/20">
                                <p className="text-2xl font-bold text-red-400">
                                    {result.total_tracks}
                                </p>
                                <p className="mt-1 text-[10px] text-gray-400">Tracks</p>
                            </div>
                            <div className="rounded-xl bg-purple-500/10 p-4 text-center ring-1 ring-purple-500/20">
                                <p className="text-2xl font-bold text-purple-400">
                                    {result.new_discoveries_count}
                                </p>
                                <p className="mt-1 text-[10px] text-gray-400">New gym tracks</p>
                            </div>
                            <div className="rounded-xl bg-orange-500/10 p-4 text-center ring-1 ring-orange-500/20">
                                <p className="text-2xl font-bold text-orange-400">
                                    {result.inspiration_count}
                                </p>
                                <p className="mt-1 text-[10px] text-gray-400">Inspirations</p>
                            </div>
                        </div>

                        {/* Spotify embed */}
                        <div className="mb-8 w-full overflow-hidden rounded-2xl">
                            <iframe
                                src={`https://open.spotify.com/embed/playlist/${result.playlist_id}?utm_source=generator&theme=0`}
                                width="100%"
                                height="352"
                                frameBorder="0"
                                allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture"
                                loading="lazy"
                                style={{ border: "none", borderRadius: "16px" }}
                            />
                        </div>

                        {/* Auto-refresh info */}
                        {autoRefresh && (
                            <div className="mb-6 w-full max-w-sm rounded-xl bg-red-500/10 p-4 text-center ring-1 ring-red-500/20">
                                <p className="text-sm text-red-300">
                                    ⏰ Auto-refresh is active
                                </p>
                                <p className="mt-1 text-xs text-gray-500">
                                    Tomorrow at 3:00 AM this playlist will be deleted and a
                                    fresh gym playlist will be created.
                                </p>
                            </div>
                        )}

                        {/* Action buttons */}
                        <div className="flex w-full max-w-sm flex-col gap-3">
                            <a
                                href={result.playlist_url}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="flex items-center justify-center gap-2 rounded-2xl bg-green-500 px-6 py-3.5 text-sm font-bold text-gray-950 transition hover:bg-green-400"
                            >
                                <svg className="h-5 w-5" viewBox="0 0 24 24" fill="currentColor">
                                    <path d="M12 0C5.4 0 0 5.4 0 12s5.4 12 12 12 12-5.4 12-12S18.66 0 12 0zm5.521 17.34c-.24.359-.66.48-1.021.24-2.82-1.74-6.36-2.101-10.561-1.141-.418.122-.779-.179-.899-.539-.12-.421.18-.78.54-.9 4.56-1.021 8.52-.6 11.64 1.32.42.18.479.659.301 1.02zm1.44-3.3c-.301.42-.841.6-1.262.3-3.239-1.98-8.159-2.58-11.939-1.38-.479.12-1.02-.12-1.14-.6-.12-.48.12-1.021.6-1.141C9.6 9.9 15 10.561 18.72 12.84c.361.181.54.78.241 1.2zm.12-3.36C15.24 8.4 8.82 8.16 5.16 9.301c-.6.179-1.2-.181-1.38-.721-.18-.601.18-1.2.72-1.381 4.26-1.26 11.28-1.02 15.721 1.621.539.3.719 1.02.419 1.56-.299.421-1.02.599-1.559.3z" />
                                </svg>
                                Open in Spotify
                            </a>
                            <button
                                onClick={handleReset}
                                className="flex items-center justify-center gap-2 rounded-2xl px-6 py-3.5 text-sm font-medium text-gray-400 ring-1 ring-white/10 transition hover:text-white hover:ring-white/20"
                            >
                                🔄 Create new Gym Playlist
                            </button>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
