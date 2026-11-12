import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { 
  Sparkles, 
  Mail, 
  Lock, 
  User, 
  Eye, 
  EyeOff, 
  AlertCircle, 
  ArrowRight, 
  Loader2, 
  ShieldCheck,
  CheckCircle2,
  Check,
  Info
} from 'lucide-react';
import { useAuth } from '../context/AuthContext';

export const RegisterPage: React.FC = () => {
  const navigate = useNavigate();
  const { signUp, isConfigured } = useAuth();

  const [fullName, setFullName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);

  // Validation & Error states
  const [fieldErrors, setFieldErrors] = useState<{
    fullName?: string;
    email?: string;
    password?: string;
    confirmPassword?: string;
  }>({});
  const [generalError, setGeneralError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  // Email format validator
  const isValidEmail = (val: string) => {
    return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(val.trim());
  };

  const validate = () => {
    const errors: {
      fullName?: string;
      email?: string;
      password?: string;
      confirmPassword?: string;
    } = {};

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

    if (!confirmPassword) {
      errors.confirmPassword = 'Please confirm your password';
    } else if (password !== confirmPassword) {
      errors.confirmPassword = 'Passwords do not match';
    }

    setFieldErrors(errors);
    return Object.keys(errors).length === 0;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setGeneralError(null);
    setSuccessMessage(null);

    if (!validate()) {
      return;
    }

    setIsLoading(true);

    try {
      const { data, error } = await signUp(email, password, { fullName });

      if (error) {
        if (error.message.includes('User already registered')) {
          setGeneralError('An account with this email address already exists. Please log in.');
        } else if (error.message.includes('Password should be at least')) {
          setGeneralError('Password should be at least 6 characters long.');
        } else {
          setGeneralError(error.message);
        }
      } else if (data?.session) {
        // Automatically signed in
        navigate('/dashboard', { replace: true });
      } else if (data?.user) {
        // Email confirmation is required by Supabase project settings
        setSuccessMessage(
          'Account created successfully! If email verification is enabled on your Supabase project, please check your inbox to confirm your address before logging in.'
        );
      }
    } catch (err: any) {
      setGeneralError(err.message || 'An unexpected error occurred during registration.');
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
            <h2 className="mt-2 text-base font-semibold text-slate-700">Create an Account</h2>
            <p className="mt-1 text-xs text-slate-500">
              Join ReviewAI to start reviewing code with state-of-the-art AI
            </p>
          </div>

          {/* General Error Message */}
          {generalError && (
            <div className="mt-6 flex items-start gap-2.5 rounded-2xl border border-rose-200 bg-rose-50/90 p-3.5 text-xs text-rose-800 animate-shake">
              <AlertCircle className="h-4 w-4 shrink-0 text-rose-600 mt-0.5" />
              <div className="flex-1 font-medium leading-relaxed">{generalError}</div>
            </div>
          )}

          {/* Success Message Banner */}
          {successMessage && (
            <div className="mt-6 rounded-2xl border border-emerald-200 bg-emerald-50/95 p-4 text-xs text-emerald-800">
              <div className="flex items-start gap-2.5">
                <CheckCircle2 className="h-4 w-4 shrink-0 text-emerald-600 mt-0.5" />
                <div className="flex-1 leading-relaxed">
                  <p className="font-semibold text-emerald-900">Registration Complete</p>
                  <p className="mt-1 text-emerald-700">{successMessage}</p>
                  <div className="mt-3">
                    <Link
                      to="/login"
                      className="inline-flex items-center gap-1.5 rounded-lg bg-emerald-600 px-3 py-1.5 text-xs font-semibold text-white shadow-sm hover:bg-emerald-700"
                    >
                      <span>Proceed to Sign In</span>
                      <ArrowRight className="h-3 w-3" />
                    </Link>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Form */}
          {!successMessage && (
            <form onSubmit={handleSubmit} className="mt-6 space-y-4" noValidate>
              {/* Full Name */}
              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1.5">
                  Full Name (Optional)
                </label>
                <div className="relative">
                  <div className="pointer-events-none absolute inset-y-0 left-0 flex items-center pl-3.5">
                    <User className="h-4 w-4 text-slate-400" />
                  </div>
                  <input
                    type="text"
                    value={fullName}
                    onChange={(e) => setFullName(e.target.value)}
                    placeholder="Alex Morgan"
                    className="w-full rounded-xl border border-slate-200 bg-slate-50/50 py-2.5 pl-10 pr-3.5 text-sm text-slate-900 placeholder-slate-400 outline-none transition-all focus:border-purple-500 focus:bg-white focus:ring-4 focus:ring-purple-500/10"
                    disabled={isLoading}
                  />
                </div>
              </div>

              {/* Email Address */}
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

              {/* Password */}
              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1.5">
                  Password
                </label>
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
                    placeholder="Minimum 6 characters"
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
                {fieldErrors.password ? (
                  <p className="mt-1.5 text-xs font-medium text-rose-600 flex items-center gap-1">
                    <AlertCircle className="h-3 w-3" />
                    {fieldErrors.password}
                  </p>
                ) : (
                  <p className="mt-1 text-[11px] text-slate-400 flex items-center gap-1">
                    <Check className={`h-3 w-3 ${password.length >= 6 ? 'text-emerald-500' : 'text-slate-300'}`} />
                    At least 6 characters
                  </p>
                )}
              </div>

              {/* Confirm Password */}
              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1.5">
                  Confirm Password
                </label>
                <div className="relative">
                  <div className="pointer-events-none absolute inset-y-0 left-0 flex items-center pl-3.5">
                    <Lock className="h-4 w-4 text-slate-400" />
                  </div>
                  <input
                    type={showConfirmPassword ? 'text' : 'password'}
                    value={confirmPassword}
                    onChange={(e) => {
                      setConfirmPassword(e.target.value);
                      if (fieldErrors.confirmPassword) setFieldErrors({ ...fieldErrors, confirmPassword: undefined });
                    }}
                    placeholder="Re-enter your password"
                    className={`w-full rounded-xl border bg-slate-50/50 py-2.5 pl-10 pr-10 text-sm text-slate-900 placeholder-slate-400 outline-none transition-all ${
                      fieldErrors.confirmPassword
                        ? 'border-rose-300 focus:border-rose-500 focus:ring-4 focus:ring-rose-500/10'
                        : 'border-slate-200 focus:border-purple-500 focus:bg-white focus:ring-4 focus:ring-purple-500/10'
                    }`}
                    disabled={isLoading}
                  />
                  <button
                    type="button"
                    onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                    className="absolute inset-y-0 right-0 flex items-center pr-3.5 text-slate-400 hover:text-slate-600 focus:outline-none"
                    aria-label={showConfirmPassword ? 'Hide confirm password' : 'Show confirm password'}
                  >
                    {showConfirmPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                  </button>
                </div>
                {fieldErrors.confirmPassword && (
                  <p className="mt-1.5 text-xs font-medium text-rose-600 flex items-center gap-1">
                    <AlertCircle className="h-3 w-3" />
                    {fieldErrors.confirmPassword}
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
                    <span>Creating account...</span>
                  </>
                ) : (
                  <>
                    <span>Create Account</span>
                    <ArrowRight className="h-4 w-4" />
                  </>
                )}
              </button>
            </form>
          )}

          {/* Footer Navigation */}
          <div className="mt-6 border-t border-slate-100 pt-5 text-center">
            <p className="text-xs text-slate-500">
              Already have an account?{' '}
              <Link
                to="/login"
                className="font-semibold text-purple-600 transition-colors hover:text-purple-700 hover:underline"
              >
                Sign in
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
              SSL Encrypted
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};

