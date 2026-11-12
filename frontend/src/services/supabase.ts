import { createClient } from '@supabase/supabase-js';

const rawUrl = import.meta.env.VITE_SUPABASE_URL;
const rawAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY;

// Normalize values and check for configuration
const defaultUrl = 'https://oeczcuvxbqdbepgzzohk.supabase.co';
export const supabaseUrl = (rawUrl && rawUrl.trim() !== '') ? rawUrl.trim() : defaultUrl;
export const supabaseAnonKey = (rawAnonKey && rawAnonKey.trim() !== '' && rawAnonKey !== 'your_supabase_anon_key_here') 
  ? rawAnonKey.trim() 
  : '';

export const isSupabaseConfigured = Boolean(
  supabaseUrl && 
  supabaseAnonKey && 
  supabaseAnonKey.length > 20
);

/**
 * Supabase client instance.
 * Safe fallback credentials are used if the environment variable is not yet populated,
 * preventing module evaluation crashes while alerting the user in the UI.
 */
export const supabase = createClient(
  supabaseUrl,
  supabaseAnonKey || 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.dummy_anon_key',
  {
    auth: {
      persistSession: true,
      autoRefreshToken: true,
      detectSessionInUrl: true,
      storage: window.localStorage,
    },
  }
);

