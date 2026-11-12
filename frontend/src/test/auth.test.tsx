import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { LoginPage } from '../pages/LoginPage';
import { RegisterPage } from '../pages/RegisterPage';
import { ProtectedRoute } from '../components/ProtectedRoute';
import * as AuthContextModule from '../context/AuthContext';

describe('Supabase Authentication Suite', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Login Page', () => {
    it('renders all form elements, brand header, and register link', () => {
      vi.spyOn(AuthContextModule, 'useAuth').mockReturnValue({
        user: null,
        session: null,
        loading: false,
        isConfigured: true,
        signIn: vi.fn(),
        signUp: vi.fn(),
        signOut: vi.fn(),
      });

      render(
        <MemoryRouter>
          <LoginPage />
        </MemoryRouter>
      );

      expect(screen.getByRole('heading', { name: /Welcome Back/i })).toBeInTheDocument();
      expect(screen.getByPlaceholderText(/developer@example.com/i)).toBeInTheDocument();
      expect(screen.getByPlaceholderText(/Enter your password/i)).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /Sign In/i })).toBeInTheDocument();
      expect(screen.getByText(/Create an account/i)).toBeInTheDocument();
    });

    it('toggles password visibility when the eye button is clicked', () => {
      vi.spyOn(AuthContextModule, 'useAuth').mockReturnValue({
        user: null,
        session: null,
        loading: false,
        isConfigured: true,
        signIn: vi.fn(),
        signUp: vi.fn(),
        signOut: vi.fn(),
      });

      render(
        <MemoryRouter>
          <LoginPage />
        </MemoryRouter>
      );

      const passwordInput = screen.getByPlaceholderText(/Enter your password/i) as HTMLInputElement;
      const toggleButton = screen.getByLabelText(/Show password/i);

      expect(passwordInput.type).toBe('password');
      fireEvent.click(toggleButton);
      expect(passwordInput.type).toBe('text');
      fireEvent.click(screen.getByLabelText(/Hide password/i));
      expect(passwordInput.type).toBe('password');
    });

    it('displays form validation error when email is empty or invalid', async () => {
      vi.spyOn(AuthContextModule, 'useAuth').mockReturnValue({
        user: null,
        session: null,
        loading: false,
        isConfigured: true,
        signIn: vi.fn(),
        signUp: vi.fn(),
        signOut: vi.fn(),
      });

      render(
        <MemoryRouter>
          <LoginPage />
        </MemoryRouter>
      );

      const submitButton = screen.getByRole('button', { name: /Sign In/i });
      fireEvent.click(submitButton);

      expect(await screen.findByText(/Email address is required/i)).toBeInTheDocument();
      expect(await screen.findByText(/Password is required/i)).toBeInTheDocument();

      const emailInput = screen.getByPlaceholderText(/developer@example.com/i);
      fireEvent.change(emailInput, { target: { value: 'not-an-email' } });
      fireEvent.click(submitButton);

      expect(await screen.findByText(/Please provide a valid email address/i)).toBeInTheDocument();
    });

    it('displays error if password is less than 6 characters', async () => {
      vi.spyOn(AuthContextModule, 'useAuth').mockReturnValue({
        user: null,
        session: null,
        loading: false,
        isConfigured: true,
        signIn: vi.fn(),
        signUp: vi.fn(),
        signOut: vi.fn(),
      });

      render(
        <MemoryRouter>
          <LoginPage />
        </MemoryRouter>
      );

      const emailInput = screen.getByPlaceholderText(/developer@example.com/i);
      const passwordInput = screen.getByPlaceholderText(/Enter your password/i);
      const submitButton = screen.getByRole('button', { name: /Sign In/i });

      fireEvent.change(emailInput, { target: { value: 'valid@example.com' } });
      fireEvent.change(passwordInput, { target: { value: '123' } });
      fireEvent.click(submitButton);

      expect(await screen.findByText(/Password must be at least 6 characters/i)).toBeInTheDocument();
    });

    it('calls signIn on valid submit and surfaces Supabase errors', async () => {
      const mockSignIn = vi.fn().mockResolvedValue({
        data: null,
        error: new Error('Invalid login credentials'),
      });

      vi.spyOn(AuthContextModule, 'useAuth').mockReturnValue({
        user: null,
        session: null,
        loading: false,
        isConfigured: true,
        signIn: mockSignIn,
        signUp: vi.fn(),
        signOut: vi.fn(),
      });

      render(
        <MemoryRouter>
          <LoginPage />
        </MemoryRouter>
      );

      fireEvent.change(screen.getByPlaceholderText(/developer@example.com/i), {
        target: { value: 'user@reviewai.dev' },
      });
      fireEvent.change(screen.getByPlaceholderText(/Enter your password/i), {
        target: { value: 'password123' },
      });

      fireEvent.click(screen.getByRole('button', { name: /Sign In/i }));

      await waitFor(() => {
        expect(mockSignIn).toHaveBeenCalledWith('user@reviewai.dev', 'password123');
      });

      expect(
        await screen.findByText(/Invalid email or password\. Please verify your credentials\./i)
      ).toBeInTheDocument();
    });
  });

  describe('Register Page', () => {
    it('validates password matching on registration', async () => {
      vi.spyOn(AuthContextModule, 'useAuth').mockReturnValue({
        user: null,
        session: null,
        loading: false,
        isConfigured: true,
        signIn: vi.fn(),
        signUp: vi.fn(),
        signOut: vi.fn(),
      });

      render(
        <MemoryRouter>
          <RegisterPage />
        </MemoryRouter>
      );

      fireEvent.change(screen.getByPlaceholderText(/developer@example.com/i), {
        target: { value: 'newuser@reviewai.dev' },
      });
      fireEvent.change(screen.getByPlaceholderText(/Minimum 6 characters/i), {
        target: { value: 'securePass1' },
      });
      fireEvent.change(screen.getByPlaceholderText(/Re-enter your password/i), {
        target: { value: 'differentPass' },
      });

      fireEvent.click(screen.getByRole('button', { name: /Create Account/i }));

      expect(await screen.findByText(/Passwords do not match/i)).toBeInTheDocument();
    });

    it('toggles password and confirm password visibility', () => {
      vi.spyOn(AuthContextModule, 'useAuth').mockReturnValue({
        user: null,
        session: null,
        loading: false,
        isConfigured: true,
        signIn: vi.fn(),
        signUp: vi.fn(),
        signOut: vi.fn(),
      });

      render(
        <MemoryRouter>
          <RegisterPage />
        </MemoryRouter>
      );

      const passInput = screen.getByPlaceholderText(/Minimum 6 characters/i) as HTMLInputElement;
      const confirmInput = screen.getByPlaceholderText(/Re-enter your password/i) as HTMLInputElement;

      expect(passInput.type).toBe('password');
      expect(confirmInput.type).toBe('password');

      fireEvent.click(screen.getByLabelText('Show password'));
      expect(passInput.type).toBe('text');

      fireEvent.click(screen.getByLabelText('Show confirm password'));
      expect(confirmInput.type).toBe('text');
    });

    it('calls signUp with metadata on valid submission', async () => {
      const mockSignUp = vi.fn().mockResolvedValue({
        data: {
          user: { id: 'user-123', email: 'alex@example.com' },
          session: null, // email confirmation required
        },
        error: null,
      });

      vi.spyOn(AuthContextModule, 'useAuth').mockReturnValue({
        user: null,
        session: null,
        loading: false,
        isConfigured: true,
        signIn: vi.fn(),
        signUp: mockSignUp,
        signOut: vi.fn(),
      });

      render(
        <MemoryRouter>
          <RegisterPage />
        </MemoryRouter>
      );

      fireEvent.change(screen.getByPlaceholderText(/Alex Morgan/i), {
        target: { value: 'Alex Morgan' },
      });
      fireEvent.change(screen.getByPlaceholderText(/developer@example.com/i), {
        target: { value: 'alex@example.com' },
      });
      fireEvent.change(screen.getByPlaceholderText(/Minimum 6 characters/i), {
        target: { value: 'securePass123' },
      });
      fireEvent.change(screen.getByPlaceholderText(/Re-enter your password/i), {
        target: { value: 'securePass123' },
      });

      fireEvent.click(screen.getByRole('button', { name: /Create Account/i }));

      await waitFor(() => {
        expect(mockSignUp).toHaveBeenCalledWith('alex@example.com', 'securePass123', {
          fullName: 'Alex Morgan',
        });
      });

      expect(await screen.findByText(/Registration Complete/i)).toBeInTheDocument();
      expect(screen.getByText(/Proceed to Sign In/i)).toBeInTheDocument();
    });
  });

  describe('Protected Route', () => {
    it('shows loading spinner while verifying session', () => {
      vi.spyOn(AuthContextModule, 'useAuth').mockReturnValue({
        user: null,
        session: null,
        loading: true,
        isConfigured: true,
        signIn: vi.fn(),
        signUp: vi.fn(),
        signOut: vi.fn(),
      });

      render(
        <MemoryRouter>
          <ProtectedRoute>
            <div>Protected Content</div>
          </ProtectedRoute>
        </MemoryRouter>
      );

      expect(screen.getByText(/Verifying Session/i)).toBeInTheDocument();
      expect(screen.queryByText(/Protected Content/i)).not.toBeInTheDocument();
    });

    it('renders protected child component when authenticated', () => {
      vi.spyOn(AuthContextModule, 'useAuth').mockReturnValue({
        user: { id: 'u-1', email: 'verified@user.com' } as any,
        session: { access_token: 'abc' } as any,
        loading: false,
        isConfigured: true,
        signIn: vi.fn(),
        signUp: vi.fn(),
        signOut: vi.fn(),
      });

      render(
        <MemoryRouter>
          <ProtectedRoute>
            <div>Protected Content Loaded</div>
          </ProtectedRoute>
        </MemoryRouter>
      );

      expect(screen.getByText(/Protected Content Loaded/i)).toBeInTheDocument();
    });
  });
});
