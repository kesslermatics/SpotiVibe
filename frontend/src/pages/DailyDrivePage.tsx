import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../lib/api";

interface PodcastShow {
    id: string;
    name: string;
    publisher: string;
    image: string | null;
    total_episodes: number;
}

interface ShowsResult {
    shows: PodcastShow[];
}

interface DailyDriveResult {
    playlist_url: string;
    playlist_id: string;
    playlist_name: string;
    total_tracks: number;
    on_repeat_count: number;
    new_discoveries_count: number;
    episodes_count: number;
}

type Step = "select" | "generating" | "done";
type DailyDriveMode = "morning" | "normal" | "night_drive";

const formatDuration = (minutes: number) =>
    minutes < 60 ? `${minutes} min` : `${Math.floor(minutes / 60)} h${minutes % 60 ? ` ${minutes % 60} min` : ""}`;

export default function DailyDrivePage({ onLogout: _onLogout }: { onLogout: () => void }) {
    const navigate = useNavigate();
    const token = localStorage.getItem("token");

    const [step, setStep] = useState<Step>("select");
    const [shows, setShows] = useState<PodcastShow[]>([]);
    const [loadingShows, setLoadingShows] = useState(true);
    const [selectedShowIds, setSelectedShowIds] = useState<Set<string>>(new Set());
    const [error, setError] = useState("");
    const [result, setResult] = useState<DailyDriveResult | null>(null);
    const [generatingStep, setGeneratingStep] = useState(0);
    const [durationMinutes, setDurationMinutes] = useState(60);
    const [dayMode, setDayMode] = useState<DailyDriveMode>("normal");
    const [familiarity, setFamiliarity] = useState(50);

    const generatingSteps = [
        { emoji: "🎧", text: "Loading your on-repeat songs..." },
        { emoji: "🤖", text: "AI is analyzing your music taste..." },
        { emoji: "🔍", text: "Searching for matching new songs on Spotify..." },
        { emoji: "🎙️", text: "Selecting podcast episodes..." },
        { emoji: "🎶", text: "Creating your Daily Drive playlist..." },
    ];

    // Auto-cycle through generating steps for visual feedback
    useEffect(() => {
        if (step !== "generating") return;
        const interval = setInterval(() => {
            setGeneratingStep((prev) =>
                prev < generatingSteps.length - 1 ? prev + 1 : prev
            );
        }, 3000);
        return () => clearInterval(interval);
    }, [step, generatingSteps.length]);

    // Load saved shows on mount
    useEffect(() => {
        if (!token) return;
        setLoadingShows(true);
        api<ShowsResult>("/daily-drive/shows", { method: "GET", token })
            .then((data) => setShows(data.shows))
            .catch((err) =>
                setError(err instanceof Error ? err.message : "Could not load podcasts")
            )
            .finally(() => setLoadingShows(false));
    }, [token]);

    const toggleShow = (id: string) => {
        setSelectedShowIds((prev) => {
            const next = new Set(prev);
            if (next.has(id)) {
                next.delete(id);
            } else {
                next.add(id);
            }
            return next;
        });
    };

    const handleGenerate = async () => {
        setError("");
        setStep("generating");
        setGeneratingStep(0);

        try {
            const data = await api<DailyDriveResult>("/daily-drive/generate", {
                method: "POST",
                body: {
                    selected_show_ids: Array.from(selectedShowIds),
                    duration_minutes: durationMinutes,
                    day_mode: dayMode,
                    familiarity,
                },
                token: token || "",
            });
            setResult(data);
            setStep("done");
        } catch (err: unknown) {
            setError(err instanceof Error ? err.message : "Something went wrong");
            setStep("select");
        }
    };

    const handleReset = () => {
        setStep("select");
        setResult(null);
        setSelectedShowIds(new Set());
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
                            <span className="text-orange-400">Daily</span>{" "}
                            <span className="text-gray-100">Drive</span>
                        </h1>
                        <p className="text-sm text-gray-400">
                            Your personal mix of music & podcasts
                        </p>
                    </div>
                    <span className="text-3xl">🚗</span>
                </div>

                {/* Error */}
                {error && (
                    <div className="mb-6 rounded-lg bg-red-500/10 px-4 py-3 text-sm text-red-400 ring-1 ring-red-500/20">
                        {error}
                    </div>
                )}

                {/* ─── STEP 1: Select podcasts ─── */}
                {step === "select" && (
                    <>
                        {/* Info card */}
                        <div className="mb-6 rounded-2xl bg-gradient-to-br from-orange-500/10 to-amber-500/5 p-5 ring-1 ring-orange-500/20">
                            <h2 className="mb-2 text-sm font-semibold text-orange-400">
                                How it works
                            </h2>
                            <ul className="space-y-1.5 text-xs text-gray-400">
                                <li className="flex items-start gap-2">
                                    <span className="mt-0.5 text-orange-400">🎵</span>
                                    Your on-repeat songs are analyzed
                                </li>
                                <li className="flex items-start gap-2">
                                    <span className="mt-0.5 text-orange-400">🤖</span>
                                    AI picks 20 of your favorites + 20 new discoveries
                                </li>
                                <li className="flex items-start gap-2">
                                    <span className="mt-0.5 text-orange-400">🎙️</span>
                                    Podcasts are mixed in between (4 songs → 1 episode)
                                </li>
                                <li className="flex items-start gap-2">
                                    <span className="mt-0.5 text-orange-400">💾</span>
                                    Finished playlist is saved to your Spotify
                                </li>
                            </ul>
                        </div>

                        {/* Daily Drive settings */}
                        <div className="mb-6 space-y-5 rounded-2xl bg-white/5 p-5 ring-1 ring-white/10">
                            <div>
                                <div className="mb-2 flex items-center justify-between">
                                    <h3 className="text-sm font-semibold text-gray-300">⏱️ Drive length</h3>
                                    <span className="rounded-full bg-orange-500/15 px-2.5 py-1 text-xs font-semibold text-orange-300">
                                        {formatDuration(durationMinutes)}
                                    </span>
                                </div>
                                <input
                                    type="range"
                                    min="20"
                                    max="240"
                                    step="10"
                                    value={durationMinutes}
                                    onChange={(event) => setDurationMinutes(Number(event.target.value))}
                                    className="h-2 w-full cursor-pointer accent-orange-500"
                                    aria-label="Drive length"
                                />
                                <div className="mt-1 flex justify-between text-[11px] text-gray-500">
                                    <span>20 min</span>
                                    <span>4 h</span>
                                </div>
                            </div>

                            <div>
                                <h3 className="mb-2 text-sm font-semibold text-gray-300">🌤️ Day mode</h3>
                                <div className="grid grid-cols-3 gap-2">
                                    {([
                                        ["morning", "☀️", "Morning"],
                                        ["normal", "🚗", "Normal"],
                                        ["night_drive", "🌙", "Night drive"],
                                    ] as const).map(([mode, emoji, label]) => (
                                        <button
                                            key={mode}
                                            type="button"
                                            onClick={() => setDayMode(mode)}
                                            className={`rounded-xl px-2 py-3 text-xs font-medium transition ${dayMode === mode
                                                ? "bg-orange-500/20 text-orange-200 ring-1 ring-orange-500/40"
                                                : "bg-white/5 text-gray-400 hover:bg-white/10 hover:text-gray-200"
                                                }`}
                                        >
                                            <span className="mb-1 block text-base">{emoji}</span>
                                            {label}
                                        </button>
                                    ))}
                                </div>
                            </div>

                            <div>
                                <div className="mb-2 flex items-center justify-between">
                                    <h3 className="text-sm font-semibold text-gray-300">✨ Familiar or new?</h3>
                                    <span className="text-xs font-medium text-orange-300">{familiarity}% new</span>
                                </div>
                                <input
                                    type="range"
                                    min="0"
                                    max="100"
                                    step="10"
                                    value={familiarity}
                                    onChange={(event) => setFamiliarity(Number(event.target.value))}
                                    className="h-2 w-full cursor-pointer accent-orange-500"
                                    aria-label="Balance between familiar songs and discoveries"
                                />
                                <div className="mt-1 flex justify-between text-[11px] text-gray-500">
                                    <span>More familiar</span>
                                    <span>More discoveries</span>
                                </div>
                            </div>
                        </div>

                        {/* Podcast selection */}
                        <div className="mb-6">
                            <h3 className="mb-3 text-sm font-semibold text-gray-300">
                                🎙️ Select Podcasts{" "}
                                <span className="font-normal text-gray-500">(optional)</span>
                            </h3>

                            {loadingShows ? (
                                <div className="space-y-2">
                                    {[...Array(4)].map((_, i) => (
                                        <div key={i} className="flex animate-pulse items-center gap-3 rounded-xl bg-white/5 p-3">
                                            <div className="h-12 w-12 rounded-lg bg-gray-700" />
                                            <div className="flex-1 space-y-2">
                                                <div className="h-4 w-3/4 rounded bg-gray-700" />
                                                <div className="h-3 w-1/2 rounded bg-gray-700" />
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            ) : shows.length === 0 ? (
                                <div className="rounded-xl bg-white/5 p-6 text-center">
                                    <span className="mb-2 block text-3xl">🎙️</span>
                                    <p className="text-sm text-gray-400">
                                        You don't have any saved podcasts yet.
                                    </p>
                                    <p className="mt-1 text-xs text-gray-500">
                                        Follow podcasts on Spotify so they show up here.
                                        You can also create a Daily Drive without podcasts!
                                    </p>
                                </div>
                            ) : (
                                <div className="max-h-[400px] space-y-2 overflow-y-auto rounded-xl pr-1">
                                    {shows.map((show) => {
                                        const isSelected = selectedShowIds.has(show.id);
                                        return (
                                            <button
                                                key={show.id}
                                                onClick={() => toggleShow(show.id)}
                                                className={`flex w-full items-center gap-3 rounded-xl p-3 text-left transition-all ${isSelected
                                                    ? "bg-orange-500/15 ring-1 ring-orange-500/30"
                                                    : "bg-white/5 hover:bg-white/10"
                                                    }`}
                                            >
                                                {/* Checkbox */}
                                                <div
                                                    className={`flex h-5 w-5 flex-shrink-0 items-center justify-center rounded transition-all ${isSelected
                                                        ? "bg-orange-500 text-white"
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
                                                    {show.image ? (
                                                        <img
                                                            src={show.image}
                                                            alt={show.name}
                                                            className="h-full w-full object-cover"
                                                        />
                                                    ) : (
                                                        <div className="flex h-full w-full items-center justify-center text-lg text-gray-600">
                                                            🎙️
                                                        </div>
                                                    )}
                                                </div>

                                                {/* Info */}
                                                <div className="min-w-0 flex-1">
                                                    <p className="truncate text-sm font-medium text-gray-200">
                                                        {show.name}
                                                    </p>
                                                    <p className="truncate text-xs text-gray-500">
                                                        {show.publisher}
                                                        {show.total_episodes > 0 && ` · ${show.total_episodes} episodes`}
                                                    </p>
                                                </div>
                                            </button>
                                        );
                                    })}
                                </div>
                            )}

                            {selectedShowIds.size > 0 && (
                                <p className="mt-2 text-xs text-orange-400/70">
                                    {selectedShowIds.size} podcast{selectedShowIds.size > 1 ? "s" : ""} selected
                                </p>
                            )}
                        </div>

                        {/* Generate button */}
                        <button
                            onClick={handleGenerate}
                            className="flex w-full items-center justify-center gap-3 rounded-2xl bg-gradient-to-r from-orange-500 to-amber-500 px-6 py-4 text-base font-bold text-white shadow-lg shadow-orange-500/25 transition hover:shadow-orange-500/40 hover:brightness-110"
                        >
                            <span className="text-xl">🚗</span>
                            Generate Daily Drive
                        </button>
                    </>
                )}

                {/* ─── STEP 2: Generating ─── */}
                {step === "generating" && (
                    <div className="flex flex-col items-center py-16">
                        {/* Animated car */}
                        <div className="mb-8 text-6xl animate-bounce">🚗</div>

                        {/* Progress steps */}
                        <div className="w-full max-w-sm space-y-4">
                            {generatingSteps.map((s, i) => {
                                const isActive = i === generatingStep;
                                const isDone = i < generatingStep;
                                return (
                                    <div
                                        key={i}
                                        className={`flex items-center gap-3 rounded-xl px-4 py-3 transition-all duration-500 ${isActive
                                            ? "bg-orange-500/15 ring-1 ring-orange-500/30"
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
                                                ? "font-medium text-orange-300"
                                                : isDone
                                                    ? "text-green-400"
                                                    : "text-gray-500"
                                                }`}
                                        >
                                            {s.text}
                                        </span>
                                        {isActive && (
                                            <div className="ml-auto h-4 w-4 animate-spin rounded-full border-2 border-orange-400 border-t-transparent" />
                                        )}
                                    </div>
                                );
                            })}
                        </div>

                        <p className="mt-8 text-xs text-gray-500">
                            This may take up to 30 seconds…
                        </p>
                    </div>
                )}

                {/* ─── STEP 3: Done ─── */}
                {step === "done" && result && (
                    <div className="flex flex-col items-center py-8">
                        {/* Success animation */}
                        <div className="mb-6 text-6xl">🎉</div>

                        <h2 className="mb-2 text-xl font-bold text-gray-100">
                            Your Daily Drive is ready!
                        </h2>
                        <p className="mb-8 text-center text-sm text-gray-400">
                            {result.playlist_name}
                        </p>

                        {/* Stats */}
                        <div className="mb-8 grid w-full max-w-sm grid-cols-3 gap-3">
                            <div className="rounded-xl bg-green-500/10 p-4 text-center ring-1 ring-green-500/20">
                                <p className="text-2xl font-bold text-green-400">
                                    {result.on_repeat_count}
                                </p>
                                <p className="mt-1 text-[10px] text-gray-400">On Repeat</p>
                            </div>
                            <div className="rounded-xl bg-purple-500/10 p-4 text-center ring-1 ring-purple-500/20">
                                <p className="text-2xl font-bold text-purple-400">
                                    {result.new_discoveries_count}
                                </p>
                                <p className="mt-1 text-[10px] text-gray-400">New Songs</p>
                            </div>
                            <div className="rounded-xl bg-orange-500/10 p-4 text-center ring-1 ring-orange-500/20">
                                <p className="text-2xl font-bold text-orange-400">
                                    {result.episodes_count}
                                </p>
                                <p className="mt-1 text-[10px] text-gray-400">Podcasts</p>
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
                                🔄 Create new Daily Drive
                            </button>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
