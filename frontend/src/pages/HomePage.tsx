import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../lib/api";

interface UserProfile {
    spotify_id: string;
    email: string | null;
    display_name: string | null;
}

interface Tile {
    id: string;
    emoji: string;
    title: string;
    description: string;
    color: string;
    route: string;
}

const tiles: Tile[] = [
    {
        id: "discover",
        emoji: "🔮",
        title: "Discover",
        description: "Describe your mood and discover songs with AI",
        color: "from-green-500/20 to-emerald-500/10 hover:ring-green-500/40",
        route: "/discover",
    },
    {
        id: "daily-drive",
        emoji: "🚗",
        title: "Daily Drive",
        description: "Your personal mix of on-repeat songs, new discoveries & podcasts",
        color: "from-orange-500/20 to-amber-500/10 hover:ring-orange-500/40",
        route: "/daily-drive",
    },
    {
        id: "daily-walk",
        emoji: "🚶",
        title: "Daily Walk",
        description: "Chill walk mix – more podcasts, less music (2 songs → 1 episode)",
        color: "from-teal-500/20 to-cyan-500/10 hover:ring-teal-500/40",
        route: "/daily-walk",
    },
    {
        id: "swipe",
        emoji: "💿",
        title: "Swipe Deck",
        description: "Swipe through song suggestions — right to save, left to skip",
        color: "from-purple-500/20 to-violet-500/10 hover:ring-purple-500/40",
        route: "/swipe-deck",
    },
    {
        id: "vibe-roast",
        emoji: "🔥",
        title: "Vibe Roast",
        description: "AI analyzes your music taste and roasts you mercilessly",
        color: "from-orange-500/20 to-red-500/10 hover:ring-orange-500/40",
        route: "/vibe-roast",
    },
    {
        id: "gym-playlist",
        emoji: "🏋️‍♂️",
        title: "Gym Playlist",
        description: "Create a motivating gym playlist from your playlists",
        color: "from-pink-500/20 to-red-500/10 hover:ring-pink-500/40",
        route: "/gym-playlist",
    },
];

export default function HomePage({ onLogout }: { onLogout: () => void }) {
    const navigate = useNavigate();
    const [user, setUser] = useState<UserProfile | null>(null);
    const token = localStorage.getItem("token");

    useEffect(() => {
        if (!token) {
            onLogout();
            return;
        }

        api<UserProfile>("/me", { token })
            .then(setUser)
            .catch(() => {
                onLogout();
            });
    }, [token, onLogout]);

    const handleLogout = () => {
        onLogout();
    };

    return (
        <div className="min-h-screen px-4 py-8">
            <div className="mx-auto max-w-2xl">
                {/* Header */}
                <div className="mb-10 flex items-center justify-between">
                    <div>
                        <h1 className="text-3xl font-bold tracking-tight">
                            <span className="text-green-400">Spoti</span>Vibe
                        </h1>
                        {user && (
                            <p className="mt-1 text-sm text-gray-400">
                                Hey, {user.display_name || user.spotify_id} 👋
                            </p>
                        )}
                    </div>
                    <button
                        onClick={handleLogout}
                        className="rounded-lg px-4 py-2 text-sm font-medium text-gray-400 ring-1 ring-white/10 transition hover:text-white hover:ring-white/20"
                    >
                        Logout
                    </button>
                </div>

                {/* Feature Tiles */}
                <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
                    {tiles.map((tile) => {
                        const isAvailable = tile.id === "discover" || tile.id === "daily-drive" || tile.id === "daily-walk" || tile.id === "gym-playlist" || tile.id === "swipe" || tile.id === "vibe-roast";
                        return (
                            <button
                                key={tile.id}
                                onClick={() => isAvailable && navigate(tile.route)}
                                className={`glass group relative cursor-pointer rounded-2xl bg-gradient-to-br p-6 text-left ring-1 ring-white/10 transition-all ${tile.color} ${!isAvailable ? "opacity-50 cursor-not-allowed" : ""
                                    }`}
                                disabled={!isAvailable}
                            >
                                <div className="mb-4 text-4xl">{tile.emoji}</div>
                                <h2 className="text-lg font-semibold text-gray-100">
                                    {tile.title}
                                </h2>
                                <p className="mt-1 text-sm text-gray-400">
                                    {tile.description}
                                </p>
                                {!isAvailable && (
                                    <span className="absolute right-4 top-4 rounded-full bg-gray-800 px-2 py-0.5 text-[10px] font-medium uppercase tracking-wider text-gray-500">
                                        Soon
                                    </span>
                                )}
                                {isAvailable && (
                                    <div className="absolute right-4 top-4 rounded-full p-1.5 text-gray-500 transition group-hover:text-green-400">
                                        <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                                            <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
                                        </svg>
                                    </div>
                                )}
                            </button>
                        );
                    })}
                </div>
            </div>
        </div>
    );
}
