import React, { useState } from 'react';
import { Link, useNavigate, useLocation } from 'react-router-dom';
import { 
  Sparkles, 
  Mail, 
  Lock, 
  Eye, 
  EyeOff, 
  AlertCircle, 
  ArrowRight, 
  Loader2, 
  ShieldCheck,
  CheckCircle2,
  Info
} from 'lucide-react';
import { useAuth } from '../context/AuthContext';

export const LoginPage: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { signIn, isConfigured } = useAuth();

  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);

  // Validation & Error states
  const [fieldErrors, setFieldErrors] = useState<{ email?: string; password?: string }>({});
  const [generalError, setGeneralError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const fromLocation = (location.state as any)?.from?.pathname || '/dashboard';

  // Email format validator
  const isValidEmail = (val: string) => {
    return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(val.trim());
  };

  const validate = () => {
    const errors: { email?: string; password?: string } = {};

    if (!email.trim()) {
      errors.email = 'Email address is required';
    } else if (!isValidEmail(email)) {
      errors.email = 'Please provide a valid email address';
    }

    if (!password) {
      errors.password = 'Password is required';
    } else if (password.length < 6) {
      errors.password = 'Password must be at least 6 characters';
    }

    setFieldErrors(errors);
    return Object.keys(errors).length === 0;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setGeneralError(null);

    if (!validate()) {
      return;
    }

    setIsLoading(true);

    try {
      const { data, error } = await signIn(email, password);

      if (error) {
        if (error.message.includes('Invalid login credentials')) {
          setGeneralError('Invalid email or password. Please verify your credentials.');
        } else if (error.message.includes('Email not confirmed')) {
          setGeneralError('Your email address has not been confirmed yet. Please check your inbox.');
        } else {
          setGeneralError(error.message);
        }
      } else if (data?.user || data?.session) {
        navigate(fromLocation, { replace: true });
      }
    } catch (err: any) {
      setGeneralError(err.message || 'An unexpected error occurred during sign in.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="relative flex min-h-screen w-screen items-center justify-center overflow-x-hidden bg-gradient-to-br from-[#f8faff] via-[#eef2ff] to-[#fdf4ff] px-4 py-12 sm:px-6 lg:px-8">
      {/* Decorative Soft Background Glows */}
      <div className="pointer-events-none absolute -left-20 top-1/4 h-96 w-96 rounded-full bg-purple-200/40 blur-3xl" />
      <div className="pointer-events-none absolute -right-20 top-1/3 h-96 w-96 rounded-full bg-blue-200/40 blur-3xl" />
      <div className="pointer-events-none absolute bottom-10 left-1/3 h-72 w-72 rounded-full bg-indigo-100/50 blur-3xl" />

      <div className="relative z-10 w-full max-w-md">
        {/* Supabase Configuration Alert (if anon key not yet supplied) */}
        {!isConfigured && (
          <div className="mb-6 rounded-2xl border border-amber-200/80 bg-amber-50/90 p-4 text-xs text-amber-800 shadow-sm backdrop-blur-sm">
            <div className="flex items-start gap-2.5">
              <Info className="h-4 w-4 text-amber-600 mt-0.5 shrink-0" />
              <div>
                <p className="font-semibold text-amber-900">Supabase API Key Needed</p>
                <p className="mt-1 text-amber-700 leading-relaxed">
                  The client is connected to <span className="font-mono font-medium">https://oeczcuvxbqdbepgzzohk.supabase.co</span>.
                  To enable live authentication, set your <span className="font-mono font-medium">VITE_SUPABASE_ANON_KEY</span> in <span className="font-mono font-medium">frontend/.env</span>.
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Brand Card */}
        <div className="overflow-hidden rounded-3xl border border-slate-200/80 bg-white/95 p-8 shadow-2xl shadow-indigo-500/10 backdrop-blur-md">
          {/* Header */}
          <div className="text-center">
            <div className="inline-flex h-12 w-12 items-center justify-center rounded-2xl bg-gradient-to-tr from-purple-600 to-indigo-600 shadow-lg shadow-purple-600/30">
              <Sparkles className="h-6 w-6 text-white" />
            </div>
            <div className="mt-4 flex items-center justify-center gap-2">
              <h1 className="text-2xl font-bold tracking-tight text-slate-900">ReviewAI</h1>
              <span className="rounded-full bg-purple-100 px-2.5 py-0.5 text-[11px] font-semibold text-purple-700">
                AI Assistant
              </span>
            </div>
            <h2 className="mt-2 text-base font-semibold text-slate-700">Welcome Back</h2>
            <p className="mt-1 text-xs text-slate-500">
              Sign in to access your AI-powered code intelligence workspace
            </p>
          </div>

          {/* General Error Message */}
          {generalError && (
            <div className="mt-6 flex items-start gap-2.5 rounded-2xl border border-rose-200 bg-rose-50/90 p-3.5 text-xs text-rose-800 animate-shake">
              <AlertCircle className="h-4 w-4 shrink-0 text-rose-600 mt-0.5" />
              <div className="flex-1 font-medium leading-relaxed">{generalError}</div>
            </div>
          )}

          {/* Form */}
          <form onSubmit={handleSubmit} className="mt-6 space-y-4" noValidate>
            {/* Email Field */}
            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1.5">
                Email Address
              </label>
              <div className="relative">
                <div className="pointer-events-none absolute inset-y-0 left-0 flex items-center pl-3.5">
                  <Mail className="h-4 w-4 text-slate-400" />
                </div>
                <input
                  type="email"
                  value={email}
                  onChange={(e) => {
                    setEmail(e.target.value);
                    if (fieldErrors.email) setFieldErrors({ ...fieldErrors, email: undefined });
                  }}
                  placeholder="developer@example.com"
                  className={`w-full rounded-xl border bg-slate-50/50 py-2.5 pl-10 pr-3.5 text-sm text-slate-900 placeholder-slate-400 outline-none transition-all ${
                    fieldErrors.email
                      ? 'border-rose-300 focus:border-rose-500 focus:ring-4 focus:ring-rose-500/10'
                      : 'border-slate-200 focus:border-purple-500 focus:bg-white focus:ring-4 focus:ring-purple-500/10'
                  }`}
                  disabled={isLoading}
                />
              </div>
              {fieldErrors.email && (
                <p className="mt-1.5 text-xs font-medium text-rose-600 flex items-center gap-1">
                  <AlertCircle className="h-3 w-3" />
                  {fieldErrors.email}
                </p>
              )}
            </div>

            {/* Password Field */}
            <div>
              <div className="flex items-center justify-between mb-1.5">
                <label className="block text-xs font-semibold text-slate-700">
                  Password
                </label>
                <span className="text-[11px] text-purple-600 hover:text-purple-700 cursor-pointer">
                  Forgot password?
                </span>
              </div>
              <div className="relative">
                <div className="pointer-events-none absolute inset-y-0 left-0 flex items-center pl-3.5">
                  <Lock className="h-4 w-4 text-slate-400" />
                </div>
                <input
                  type={showPassword ? 'text' : 'password'}
                  value={password}
                  onChange={(e) => {
                    setPassword(e.target.value);
                    if (fieldErrors.password) setFieldErrors({ ...fieldErrors, password: undefined });
                  }}
                  placeholder="Enter your password"
                  className={`w-full rounded-xl border bg-slate-50/50 py-2.5 pl-10 pr-10 text-sm text-slate-900 placeholder-slate-400 outline-none transition-all ${
                    fieldErrors.password
                      ? 'border-rose-300 focus:border-rose-500 focus:ring-4 focus:ring-rose-500/10'
                      : 'border-slate-200 focus:border-purple-500 focus:bg-white focus:ring-4 focus:ring-purple-500/10'
                  }`}
                  disabled={isLoading}
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute inset-y-0 right-0 flex items-center pr-3.5 text-slate-400 hover:text-slate-600 focus:outline-none"
                  aria-label={showPassword ? 'Hide password' : 'Show password'}
                >
                  {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                </button>
              </div>
              {fieldErrors.password && (
                <p className="mt-1.5 text-xs font-medium text-rose-600 flex items-center gap-1">
                  <AlertCircle className="h-3 w-3" />
                  {fieldErrors.password}
                </p>
              )}
            </div>

            {/* Submit Button */}
            <button
              type="submit"
              disabled={isLoading}
              className="mt-2 flex w-full items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-purple-600 via-indigo-600 to-blue-600 py-2.5 px-4 text-sm font-semibold text-white shadow-md shadow-purple-600/20 transition-all hover:opacity-95 hover:shadow-lg hover:shadow-purple-600/30 active:scale-[0.99] disabled:cursor-not-allowed disabled:opacity-60"
            >
              {isLoading ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  <span>Signing in...</span>
                </>
              ) : (
                <>
                  <span>Sign In</span>
                  <ArrowRight className="h-4 w-4" />
                </>
              )}
            </button>
          </form>

          {/* Footer Navigation */}
          <div className="mt-6 border-t border-slate-100 pt-5 text-center">
            <p className="text-xs text-slate-500">
              Don't have an account?{' '}
              <Link
                to="/register"
                className="font-semibold text-purple-600 transition-colors hover:text-purple-700 hover:underline"
              >
                Create an account
              </Link>
            </p>
          </div>

          {/* Trust Highlights */}
          <div className="mt-6 rounded-2xl bg-slate-50/80 p-3.5 border border-slate-100 flex items-center justify-around text-[11px] text-slate-600">
            <span className="flex items-center gap-1">
              <ShieldCheck className="h-3.5 w-3.5 text-purple-600" />
              Supabase Auth
            </span>
            <span className="h-3 w-px bg-slate-200" />
            <span className="flex items-center gap-1">
              <CheckCircle2 className="h-3.5 w-3.5 text-indigo-600" />
              End-to-End Secure
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};

