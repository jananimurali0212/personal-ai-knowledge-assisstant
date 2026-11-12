import React, { createContext, useContext, useEffect, useState } from 'react';
import { User, Session, AuthError } from '@supabase/supabase-js';
import { supabase, isSupabaseConfigured } from '../services/supabase';

interface AuthContextType {
  user: User | null;
  session: Session | null;
  loading: boolean;
  isConfigured: boolean;
  signIn: (email: string, password: string) => Promise<{ data: any; error: AuthError | Error | null }>;
  signUp: (
    email: string,
    password: string,
    metadata?: { fullName?: string }
  ) => Promise<{ data: any; error: AuthError | Error | null }>;
  signOut: () => Promise<{ error: AuthError | Error | null }>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [session, setSession] = useState<Session | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let mounted = true;

    if (!isSupabaseConfigured) {
      setLoading(false);
      return;
    }

    // 1. Get initial session
    supabase.auth.getSession().then(({ data: { session }, error }) => {
      if (!mounted) return;
      if (error) {
        console.warn('Supabase getSession error:', error.message);
      }
      setSession(session);
      setUser(session?.user ?? null);
      setLoading(false);
    }).catch((err) => {
      if (mounted) {
        console.warn('Failed to retrieve session:', err);
        setLoading(false);
      }
    });

    // 2. Listen to auth state transitions
    const { data: { subscription } } = supabase.auth.onAuthStateChange((_event, newSession) => {
      if (!mounted) return;
      setSession(newSession);
      setUser(newSession?.user ?? null);
      setLoading(false);
    });

    return () => {
      mounted = false;
      subscription.unsubscribe();
    };
  }, []);

  const signIn = async (email: string, password: string) => {
    if (!isSupabaseConfigured) {
      return {
        data: null,
        error: new Error(
          'Supabase anon key is not configured. Please set VITE_SUPABASE_ANON_KEY in frontend/.env.'
        ),
      };
    }

    try {
      const response = await supabase.auth.signInWithPassword({
        email: email.trim(),
        password,
      });
      if (response.data.session) {
        setSession(response.data.session);
        setUser(response.data.user);
      }
      return response;
    } catch (err: any) {
      return { data: null, error: err };
    }
  };

  const signUp = async (
    email: string,
    password: string,
    metadata?: { fullName?: string }
  ) => {
    if (!isSupabaseConfigured) {
      return {
        data: null,
        error: new Error(
          'Supabase anon key is not configured. Please set VITE_SUPABASE_ANON_KEY in frontend/.env.'
        ),
      };
    }

    try {
      const response = await supabase.auth.signUp({
        email: email.trim(),
        password,
        options: {
          data: {
            full_name: metadata?.fullName?.trim() || '',
          },
        },
      });
      return response;
    } catch (err: any) {
      return { data: null, error: err };
    }
  };

  const signOut = async () => {
    if (!isSupabaseConfigured) {
      setUser(null);
      setSession(null);
      return { error: null };
    }

    try {
      const { error } = await supabase.auth.signOut();
      setUser(null);
      setSession(null);
      return { error };
    } catch (err: any) {
      setUser(null);
      setSession(null);
      return { error: err };
    }
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        session,
        loading,
        isConfigured: isSupabaseConfigured,
        signIn,
        signUp,
        signOut,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

