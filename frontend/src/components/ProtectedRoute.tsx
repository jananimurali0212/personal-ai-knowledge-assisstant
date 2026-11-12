import React from 'react';
import { Navigate, useLocation } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { Sparkles, Loader2 } from 'lucide-react';

interface ProtectedRouteProps {
  children?: React.ReactNode;
}

export const ProtectedRoute: React.FC<ProtectedRouteProps> = ({ children }) => {
  const { user, loading } = useAuth();
  const location = useLocation();

  if (loading) {
    return (
      <div className="flex h-screen w-screen items-center justify-center bg-gradient-to-br from-slate-50 via-indigo-50/30 to-purple-50/40">
        <div className="flex flex-col items-center space-y-4 rounded-3xl border border-purple-100 bg-white/90 p-8 shadow-xl shadow-purple-500/5 backdrop-blur-md">
          <div className="relative flex h-14 w-14 items-center justify-center rounded-2xl bg-gradient-to-tr from-purple-600 to-indigo-500 shadow-md shadow-purple-600/25">
            <Sparkles className="h-7 w-7 text-white animate-pulse" />
            <div className="absolute -inset-1 rounded-2xl border border-purple-300 animate-ping opacity-25" />
          </div>
          <div className="text-center">
            <h3 className="text-sm font-semibold text-slate-800">Verifying Session</h3>
            <p className="text-xs text-slate-400 mt-0.5">Connecting to Supabase Auth...</p>
          </div>
          <Loader2 className="h-5 w-5 animate-spin text-purple-600" />
        </div>
      </div>
    );
  }

  if (!user) {
    return <Navigate to="/login" replace state={{ from: location }} />;
  }

  return children ? <>{children}</> : null;
};

