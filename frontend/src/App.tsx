import { useState, useCallback } from "react";
import { Routes, Route, Navigate, useNavigate } from "react-router-dom";
import LoginPage from "./pages/LoginPage";
import CallbackPage from "./pages/CallbackPage";
import HomePage from "./pages/HomePage";
import DiscoverPage from "./pages/DiscoverPage";
import DailyDrivePage from "./pages/DailyDrivePage";
import DailyWalkPage from "./pages/DailyWalkPage";
import GymPlaylistPage from "./pages/GymPlaylistPage";
import SwipeDeckPage from "./pages/SwipeDeckPage";
import RoastPage from "./pages/RoastPage";

function App() {
  const [isLoggedIn, setIsLoggedIn] = useState(!!localStorage.getItem("token"));
  const navigate = useNavigate();

  const logout = useCallback(() => {
    localStorage.removeItem("token");
    sessionStorage.removeItem("spotify_redirect_uri");
    setIsLoggedIn(false);
    navigate("/login", { replace: true });
  }, [navigate]);

  return (
    <div className="relative min-h-screen overflow-hidden bg-gray-950">
      {/* Ambient background blobs */}
      <div className="pointer-events-none fixed inset-0 -z-10">
        <div className="absolute -left-40 -top-40 h-[500px] w-[500px] rounded-full bg-green-500/10 blur-3xl" />
        <div className="absolute -bottom-40 -right-40 h-[500px] w-[500px] rounded-full bg-green-600/10 blur-3xl" />
      </div>

      <Routes>
        <Route path="/login" element={isLoggedIn ? <Navigate to="/" replace /> : <LoginPage />} />
        <Route path="/callback" element={<CallbackPage />} />
        <Route path="/" element={isLoggedIn ? <HomePage onLogout={logout} /> : <Navigate to="/login" replace />} />
        <Route path="/discover" element={isLoggedIn ? <DiscoverPage onLogout={logout} /> : <Navigate to="/login" replace />} />
        <Route path="/daily-drive" element={isLoggedIn ? <DailyDrivePage onLogout={logout} /> : <Navigate to="/login" replace />} />
        <Route path="/daily-walk" element={isLoggedIn ? <DailyWalkPage onLogout={logout} /> : <Navigate to="/login" replace />} />
        <Route path="/gym-playlist" element={isLoggedIn ? <GymPlaylistPage onLogout={logout} /> : <Navigate to="/login" replace />} />
        <Route path="/swipe-deck" element={isLoggedIn ? <SwipeDeckPage onLogout={logout} /> : <Navigate to="/login" replace />} />
        <Route path="/vibe-roast" element={isLoggedIn ? <RoastPage onLogout={logout} /> : <Navigate to="/login" replace />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </div>
  );
}

export default App;
