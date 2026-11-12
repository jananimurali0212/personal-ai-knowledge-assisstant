import React from 'react';
import { BrowserRouter, Routes, Route, Navigate, useNavigate } from 'react-router-dom';
import { AuthProvider, useAuth } from './context/AuthContext';
import { ProtectedRoute } from './components/ProtectedRoute';
import { DashboardLayout } from './layouts/DashboardLayout';
import { DashboardPage } from './pages/DashboardPage';
import { MyDocumentsPage } from './pages/MyDocumentsPage';
import { AIAssistantPage } from './pages/AIAssistantPage';
import { SearchPage } from './pages/SearchPage';
import { SettingsPage } from './pages/SettingsPage';
import { LoginPage } from './pages/LoginPage';
import { RegisterPage } from './pages/RegisterPage';
import { NavPage } from './types';

// Redirect authenticated users away from public auth pages (login/register) to dashboard
const PublicAuthRoute: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { user, loading } = useAuth();

  if (loading) {
    return null;
  }

  if (user) {
    return <Navigate to="/dashboard" replace />;
  }

  return <>{children}</>;
};

// Wrapper for dashboard pages ensuring consistent layout and navigation
const DashboardViewWrapper: React.FC<{ page: NavPage }> = ({ page }) => {
  const navigate = useNavigate();

  const handleNavigate = (targetPage: NavPage) => {
    navigate(targetPage === 'dashboard' ? '/dashboard' : `/${targetPage}`);
  };

  const renderCurrentPage = () => {
    switch (page) {
      case 'dashboard':
        return <DashboardPage onNavigate={handleNavigate} />;
      case 'documents':
        return <MyDocumentsPage />;
      case 'assistant':
        return <AIAssistantPage />;
      case 'search':
        return <SearchPage />;
      case 'settings':
        return <SettingsPage />;
      default:
        return <DashboardPage onNavigate={handleNavigate} />;
    }
  };

  return (
    <DashboardLayout currentPage={page} onNavigate={handleNavigate}>
      {renderCurrentPage()}
    </DashboardLayout>
  );
};

export const App: React.FC = () => {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Routes>
          {/* Public Auth Routes */}
          <Route
            path="/login"
            element={
              <PublicAuthRoute>
                <LoginPage />
              </PublicAuthRoute>
            }
          />
          <Route
            path="/register"
            element={
              <PublicAuthRoute>
                <RegisterPage />
              </PublicAuthRoute>
            }
          />

          {/* Protected Application Routes */}
          <Route
            path="/dashboard"
            element={
              <ProtectedRoute>
                <DashboardViewWrapper page="dashboard" />
              </ProtectedRoute>
            }
          />
          <Route
            path="/documents"
            element={
              <ProtectedRoute>
                <DashboardViewWrapper page="documents" />
              </ProtectedRoute>
            }
          />
          <Route
            path="/assistant"
            element={
              <ProtectedRoute>
                <DashboardViewWrapper page="assistant" />
              </ProtectedRoute>
            }
          />
          <Route
            path="/search"
            element={
              <ProtectedRoute>
                <DashboardViewWrapper page="search" />
              </ProtectedRoute>
            }
          />
          <Route
            path="/settings"
            element={
              <ProtectedRoute>
                <DashboardViewWrapper page="settings" />
              </ProtectedRoute>
            }
          />

          {/* Backwards compatibility aliases */}
          <Route path="/new-review" element={<Navigate to="/documents" replace />} />
          <Route path="/history" element={<Navigate to="/documents" replace />} />

          {/* Root and Fallback redirection */}
          <Route path="/" element={<Navigate to="/dashboard" replace />} />
          <Route path="*" element={<Navigate to="/dashboard" replace />} />
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  );
};

export default App;
