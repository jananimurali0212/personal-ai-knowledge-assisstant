import { describe, it, expect } from 'vitest';
import { supabaseUrl, supabase, isSupabaseConfigured } from '../services/supabase';

describe('Supabase Client Configuration', () => {
  it('correctly sets the target project URL to user provided endpoint', () => {
    expect(supabaseUrl).toBe('https://oeczcuvxbqdbepgzzohk.supabase.co');
  });

  it('instantiates the supabase auth client without crash', () => {
    expect(supabase).toBeDefined();
    expect(supabase.auth).toBeDefined();
    expect(typeof supabase.auth.signInWithPassword).toBe('function');
    expect(typeof supabase.auth.signUp).toBe('function');
    expect(typeof supabase.auth.signOut).toBe('function');
    expect(typeof supabase.auth.getSession).toBe('function');
    expect(typeof supabase.auth.onAuthStateChange).toBe('function');
  });

  it('tracks configuration status accurately', () => {
    expect(typeof isSupabaseConfigured).toBe('boolean');
  });
});

