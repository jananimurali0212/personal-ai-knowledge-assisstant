import React, { useState, useEffect } from 'react';
import {
  HelpCircle,
  X,
  Sparkles,
  Loader2,
  CheckCircle2,
  XCircle,
  Award,
  BookOpen,
  RotateCcw,
  Clock,
  History,
  ChevronRight,
  Quote,
  Layers,
} from 'lucide-react';
import {
  DocumentRecord,
  QuizDifficulty,
  QuestionType,
  QuizResponse,
  QuizSummary,
  QuizSubmitResponse,
  QuizAttemptHistoryItem,
} from '../types';
import {
  generateQuiz,
  fetchDocumentQuizzes,
  submitQuiz,
  fetchQuizAttempts,
} from '../services/quizzes';

interface QuizModalProps {
  isOpen: boolean;
  onClose: () => void;
  document: DocumentRecord | null;
  userId: string;
  token?: string;
}

type ModalView = 'setup' | 'taking' | 'results' | 'history';

export const QuizModal: React.FC<QuizModalProps> = ({
  isOpen,
  onClose,
  document,
  userId,
  token,
}) => {
  // Navigation & View state
  const [activeView, setActiveView] = useState<ModalView>('setup');

  // Configuration state
  const [difficulty, setDifficulty] = useState<QuizDifficulty>('medium');
  const [numQuestions, setNumQuestions] = useState<number>(5);
  const [selectedTypes, setSelectedTypes] = useState<QuestionType[]>([
    'multiple_choice',
    'true_false',
    'short_answer',
  ]);

  // Active Quiz state
  const [activeQuiz, setActiveQuiz] = useState<QuizResponse | null>(null);
  const [userAnswers, setUserAnswers] = useState<Record<string, string>>({});
  const [quizResult, setQuizResult] = useState<QuizSubmitResponse | null>(null);

  // History & List state
  const [pastQuizzes, setPastQuizzes] = useState<QuizSummary[]>([]);
  const [pastAttempts, setPastAttempts] = useState<QuizAttemptHistoryItem[]>([]);

  // Loading & Error states
  const [isGenerating, setIsGenerating] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Load existing quizzes when modal opens
  useEffect(() => {
    if (isOpen && document) {
      loadPastQuizzes();
      setActiveView('setup');
      setQuizResult(null);
      setUserAnswers({});
      setErrorMessage(null);
    }
  }, [isOpen, document]);

  const loadPastQuizzes = async () => {
    if (!document) return;
    const { data, error } = await fetchDocumentQuizzes(document.id, userId, token);
    if (!error && data) {
      setPastQuizzes(data);
    }
  };

  if (!isOpen || !document) return null;

  const toggleQuestionType = (type: QuestionType) => {
    if (selectedTypes.includes(type)) {
      if (selectedTypes.length === 1) return; // Keep at least one
      setSelectedTypes(selectedTypes.filter((t) => t !== type));
    } else {
      setSelectedTypes([...selectedTypes, type]);
    }
  };

  const handleGenerate = async () => {
    setIsGenerating(true);
    setErrorMessage(null);
    const { data, error } = await generateQuiz(
      document.id,
      {
        difficulty,
        num_questions: numQuestions,
        question_types: selectedTypes,
      },
      userId,
      token
    );

    setIsGenerating(false);
    if (error || !data) {
      setErrorMessage(error || 'Failed to generate quiz.');
      return;
    }

    setActiveQuiz(data);
    setUserAnswers({});
    setQuizResult(null);
    setActiveView('taking');
    loadPastQuizzes();
  };

  const handleAnswerChange = (questionId: string, answer: string) => {
    setUserAnswers((prev) => ({
      ...prev,
      [questionId]: answer,
    }));
  };

  const handleSubmit = async () => {
    if (!activeQuiz) return;
    setIsSubmitting(true);
    setErrorMessage(null);

    const { data, error } = await submitQuiz(
      activeQuiz.quiz.id,
      { answers: userAnswers },
      userId,
      token
    );

    setIsSubmitting(false);
    if (error || !data) {
      setErrorMessage(error || 'Failed to submit quiz.');
      return;
    }

    setQuizResult(data);
    setActiveView('results');
  };

  const handleViewAttempts = async (quizId: string) => {
    setIsLoadingHistory(true);
    const { data } = await fetchQuizAttempts(quizId, userId, token);
    setIsLoadingHistory(false);
    if (data) {
      setPastAttempts(data);
      setActiveView('history');
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/75 backdrop-blur-sm animate-fadeIn">
      <div
        className="relative w-full max-w-4xl max-h-[90vh] flex flex-col rounded-2xl border border-slate-800 bg-[#0F172A] text-slate-100 shadow-2xl shadow-indigo-500/10 overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between border-b border-slate-800/80 px-6 py-4 bg-slate-900/60">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 shadow-inner">
              <HelpCircle className="h-5 w-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h3 className="text-base font-bold text-white tracking-tight truncate max-w-md">
                  {document.file_name}
                </h3>
                <span className="uppercase rounded-md border border-indigo-500/30 bg-indigo-500/10 px-2 py-0.5 text-[10px] font-bold text-indigo-300">
                  Quiz Assessment
                </span>
              </div>
              <p className="text-xs text-slate-400 mt-0.5">
                AI-Grounded Knowledge Test & Fact Verification
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            {activeView !== 'setup' && (
              <button
                onClick={() => setActiveView('setup')}
                className="px-3 py-1.5 rounded-lg border border-slate-700 bg-slate-800/60 hover:bg-slate-700/80 text-xs font-semibold text-slate-300 transition-colors flex items-center gap-1.5"
              >
                <RotateCcw className="h-3.5 w-3.5" />
                Setup
              </button>
            )}
            <button
              onClick={onClose}
              className="rounded-lg p-2 text-slate-400 hover:bg-slate-800 hover:text-white transition-colors"
            >
              <X className="h-5 w-5" />
            </button>
          </div>
        </div>

        {/* Content Body */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {errorMessage && (
            <div className="rounded-xl border border-rose-500/30 bg-rose-500/10 p-4 text-xs text-rose-300 flex items-center gap-2">
              <XCircle className="h-4 w-4 shrink-0 text-rose-400" />
              <span>{errorMessage}</span>
            </div>
          )}

          {/* VIEW 1: SETUP */}
          {activeView === 'setup' && (
            <div className="space-y-6">
              {/* Difficulty Selector */}
              <div>
                <label className="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">
                  Select Difficulty Level
                </label>
                <div className="grid grid-cols-3 gap-3">
                  {(
                    [
                      {
                        key: 'easy',
                        title: 'Easy',
                        desc: 'Factual recall, definitions & terminology',
                        badge: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30',
                      },
                      {
                        key: 'medium',
                        title: 'Medium',
                        desc: 'Conceptual understanding & applications',
                        badge: 'bg-amber-500/10 text-amber-400 border-amber-500/30',
                      },
                      {
                        key: 'hard',
                        title: 'Hard',
                        desc: 'Deep analysis, edge cases & synthesis',
                        badge: 'bg-rose-500/10 text-rose-400 border-rose-500/30',
                      },
                    ] as const
                  ).map((item) => (
                    <button
                      key={item.key}
                      onClick={() => setDifficulty(item.key)}
                      className={`flex flex-col text-left p-4 rounded-xl border transition-all ${
                        difficulty === item.key
                          ? `border-indigo-500 bg-indigo-500/10 shadow-lg shadow-indigo-500/10 ring-1 ring-indigo-500`
                          : 'border-slate-800 bg-slate-900/40 hover:border-slate-700 hover:bg-slate-800/40'
                      }`}
                    >
                      <span className={`text-xs font-bold px-2 py-0.5 rounded border inline-block w-fit ${item.badge}`}>
                        {item.title}
                      </span>
                      <span className="text-xs text-slate-400 mt-2">{item.desc}</span>
                    </button>
                  ))}
                </div>
              </div>

              {/* Number of Questions */}
              <div>
                <label className="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">
                  Number of Questions
                </label>
                <div className="flex gap-2">
                  {[3, 5, 8, 10].map((num) => (
                    <button
                      key={num}
                      onClick={() => setNumQuestions(num)}
                      className={`px-4 py-2 rounded-xl text-xs font-bold border transition-all ${
                        numQuestions === num
                          ? 'border-indigo-500 bg-indigo-500/20 text-indigo-300'
                          : 'border-slate-800 bg-slate-900/40 text-slate-400 hover:border-slate-700'
                      }`}
                    >
                      {num} Questions
                    </button>
                  ))}
                </div>
              </div>

              {/* Question Types */}
              <div>
                <label className="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">
                  Question Formats
                </label>
                <div className="flex flex-wrap gap-2">
                  {[
                    { key: 'multiple_choice' as QuestionType, label: 'Multiple Choice (4 Options)' },
                    { key: 'true_false' as QuestionType, label: 'True / False' },
                    { key: 'short_answer' as QuestionType, label: 'Short Answer' },
                  ].map((type) => {
                    const isSelected = selectedTypes.includes(type.key);
                    return (
                      <button
                        key={type.key}
                        onClick={() => toggleQuestionType(type.key)}
                        className={`px-3 py-1.5 rounded-lg text-xs font-semibold border transition-all ${
                          isSelected
                            ? 'border-indigo-500/60 bg-indigo-500/15 text-indigo-200'
                            : 'border-slate-800 bg-slate-900/40 text-slate-500 hover:border-slate-700'
                        }`}
                      >
                        {isSelected ? '✓ ' : '+ '}
                        {type.label}
                      </button>
                    );
                  })}
                </div>
              </div>

              {/* Generate Button */}
              <button
                onClick={handleGenerate}
                disabled={isGenerating}
                className="w-full flex items-center justify-center gap-2 py-3.5 px-4 rounded-xl bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 text-white font-bold text-sm shadow-lg shadow-indigo-500/20 transition-all disabled:opacity-50"
              >
                {isGenerating ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Generating Grounded Quiz with Gemini...
                  </>
                ) : (
                  <>
                    <Sparkles className="h-4 w-4" />
                    Generate Quiz ({numQuestions} Questions)
                  </>
                )}
              </button>

              {/* Previous Quizzes Section */}
              {pastQuizzes.length > 0 && (
                <div className="pt-4 border-t border-slate-800/80">
                  <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-3 flex items-center gap-1.5">
                    <History className="h-3.5 w-3.5 text-indigo-400" />
                    Previously Generated Quizzes ({pastQuizzes.length})
                  </h4>
                  <div className="space-y-2">
                    {pastQuizzes.map((pq) => (
                      <div
                        key={pq.id}
                        className="flex items-center justify-between p-3 rounded-xl border border-slate-800/80 bg-slate-900/30 hover:bg-slate-800/40 transition-colors"
                      >
                        <div>
                          <div className="flex items-center gap-2">
                            <span className="text-xs font-bold text-slate-200">{pq.title}</span>
                            <span className="text-[10px] font-semibold px-1.5 py-0.5 rounded border border-indigo-500/30 bg-indigo-500/10 text-indigo-300 uppercase">
                              {pq.difficulty}
                            </span>
                          </div>
                          <p className="text-[11px] text-slate-400 mt-0.5">
                            {pq.total_questions} Questions • Created {new Date(pq.created_at).toLocaleDateString()}
                          </p>
                        </div>
                        <button
                          onClick={() => handleViewAttempts(pq.id)}
                          className="px-3 py-1 rounded-lg border border-slate-700 bg-slate-800 hover:bg-slate-700 text-xs font-semibold text-slate-300 flex items-center gap-1 transition-colors"
                        >
                          Attempts
                          <ChevronRight className="h-3 w-3" />
                        </button>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* VIEW 2: TAKING THE QUIZ */}
          {activeView === 'taking' && activeQuiz && (
            <div className="space-y-6">
              {/* Progress & Info */}
              <div className="flex items-center justify-between p-4 rounded-xl border border-indigo-500/20 bg-indigo-500/5">
                <div>
                  <h4 className="text-sm font-bold text-white">{activeQuiz.quiz.title}</h4>
                  <p className="text-xs text-slate-400 mt-0.5">
                    Difficulty: <span className="text-indigo-300 font-semibold uppercase">{activeQuiz.quiz.difficulty}</span> •{' '}
                    {activeQuiz.questions.length} Questions
                  </p>
                </div>
                <div className="text-xs text-slate-400">
                  Answered:{' '}
                  <span className="font-bold text-indigo-400">
                    {Object.keys(userAnswers).length}/{activeQuiz.questions.length}
                  </span>
                </div>
              </div>

              {/* Questions List */}
              <div className="space-y-6">
                {activeQuiz.questions.map((q, idx) => (
                  <div
                    key={q.id}
                    className="p-5 rounded-xl border border-slate-800 bg-slate-900/40 space-y-3"
                  >
                    <div className="flex items-start justify-between gap-2">
                      <div className="flex items-center gap-2">
                        <span className="flex h-6 w-6 items-center justify-center rounded-full bg-indigo-500/20 text-indigo-400 text-xs font-bold border border-indigo-500/30">
                          {idx + 1}
                        </span>
                        <span className="text-xs font-bold uppercase tracking-wider text-slate-400">
                          {q.question_type.replace('_', ' ')}
                        </span>
                      </div>
                      {q.page_number && (
                        <span className="text-[11px] text-slate-400 flex items-center gap-1 bg-slate-800/80 px-2 py-0.5 rounded border border-slate-700">
                          <BookOpen className="h-3 w-3 text-slate-400" />
                          Page {q.page_number}
                        </span>
                      )}
                    </div>

                    <p className="text-sm font-medium text-slate-200">{q.question_text}</p>

                    {/* Multiple Choice Options */}
                    {q.question_type === 'multiple_choice' && (
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-2 pt-1">
                        {q.options.map((opt, optIdx) => {
                          const isSelected = userAnswers[q.id] === opt;
                          return (
                            <button
                              key={optIdx}
                              onClick={() => handleAnswerChange(q.id, opt)}
                              className={`flex items-center gap-2.5 p-3 rounded-lg border text-left text-xs transition-all ${
                                isSelected
                                  ? 'border-indigo-500 bg-indigo-500/15 text-indigo-200 font-semibold shadow-inner'
                                  : 'border-slate-800 bg-slate-900/60 text-slate-300 hover:border-slate-700 hover:bg-slate-800/40'
                              }`}
                            >
                              <span
                                className={`flex h-5 w-5 shrink-0 items-center justify-center rounded-full text-[10px] font-bold border ${
                                  isSelected
                                    ? 'border-indigo-400 bg-indigo-500 text-white'
                                    : 'border-slate-700 bg-slate-800 text-slate-400'
                                }`}
                              >
                                {String.fromCharCode(65 + optIdx)}
                              </span>
                              <span className="truncate">{opt}</span>
                            </button>
                          );
                        })}
                      </div>
                    )}

                    {/* True / False Options */}
                    {q.question_type === 'true_false' && (
                      <div className="flex gap-3 pt-1">
                        {['True', 'False'].map((tf) => {
                          const isSelected = userAnswers[q.id] === tf;
                          return (
                            <button
                              key={tf}
                              onClick={() => handleAnswerChange(q.id, tf)}
                              className={`flex-1 py-2.5 px-4 rounded-lg border text-center text-xs font-bold transition-all ${
                                isSelected
                                  ? 'border-indigo-500 bg-indigo-500/20 text-indigo-300 ring-1 ring-indigo-500'
                                  : 'border-slate-800 bg-slate-900/60 text-slate-400 hover:border-slate-700'
                              }`}
                            >
                              {tf}
                            </button>
                          );
                        })}
                      </div>
                    )}

                    {/* Short Answer Input */}
                    {q.question_type === 'short_answer' && (
                      <div className="pt-1">
                        <textarea
                          rows={2}
                          value={userAnswers[q.id] || ''}
                          onChange={(e) => handleAnswerChange(q.id, e.target.value)}
                          placeholder="Type your concise answer based on the document..."
                          className="w-full rounded-lg border border-slate-800 bg-slate-950/60 p-3 text-xs text-slate-200 placeholder-slate-500 focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
                        />
                      </div>
                    )}
                  </div>
                ))}
              </div>

              {/* Submit Action */}
              <div className="flex items-center justify-between pt-4 border-t border-slate-800/80">
                <p className="text-xs text-slate-400">
                  {Object.keys(userAnswers).length === activeQuiz.questions.length
                    ? 'All questions answered. Ready to submit!'
                    : `Please answer remaining ${activeQuiz.questions.length - Object.keys(userAnswers).length} question(s).`}
                </p>
                <button
                  onClick={handleSubmit}
                  disabled={isSubmitting || Object.keys(userAnswers).length === 0}
                  className="flex items-center gap-2 py-2.5 px-5 rounded-xl bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white font-bold text-xs shadow-lg shadow-emerald-500/20 transition-all disabled:opacity-50"
                >
                  {isSubmitting ? (
                    <>
                      <Loader2 className="h-4 w-4 animate-spin" />
                      Grading Answers...
                    </>
                  ) : (
                    <>
                      <CheckCircle2 className="h-4 w-4" />
                      Submit Quiz
                    </>
                  )}
                </button>
              </div>
            </div>
          )}

          {/* VIEW 3: RESULTS & DETAILED REVIEW */}
          {activeView === 'results' && quizResult && (
            <div className="space-y-6">
              {/* Score Card Banner */}
              <div
                className={`p-6 rounded-2xl border flex flex-col md:flex-row items-center justify-between gap-4 ${
                  quizResult.passed
                    ? 'border-emerald-500/30 bg-emerald-500/10'
                    : 'border-amber-500/30 bg-amber-500/10'
                }`}
              >
                <div className="flex items-center gap-4">
                  <div
                    className={`flex h-14 w-14 items-center justify-center rounded-2xl border shadow-inner ${
                      quizResult.passed
                        ? 'border-emerald-500/40 bg-emerald-500/20 text-emerald-400'
                        : 'border-amber-500/40 bg-amber-500/20 text-amber-400'
                    }`}
                  >
                    <Award className="h-8 w-8" />
                  </div>
                  <div>
                    <div className="flex items-center gap-2">
                      <h4 className="text-xl font-extrabold text-white">
                        {quizResult.percentage}% Score
                      </h4>
                      <span
                        className={`text-xs font-bold px-2 py-0.5 rounded border uppercase ${
                          quizResult.passed
                            ? 'border-emerald-500/40 bg-emerald-500/20 text-emerald-300'
                            : 'border-amber-500/40 bg-amber-500/20 text-amber-300'
                        }`}
                      >
                        {quizResult.passed ? 'Passed (≥70%)' : 'Needs Review (<70%)'}
                      </span>
                    </div>
                    <p className="text-xs text-slate-400 mt-1">
                      Scored {quizResult.score} out of {quizResult.total_points} points on{' '}
                      {new Date(quizResult.completed_at).toLocaleTimeString()}
                    </p>
                  </div>
                </div>

                <div className="flex gap-2">
                  <button
                    onClick={() => setActiveView('taking')}
                    className="px-3.5 py-2 rounded-xl border border-slate-700 bg-slate-800/80 hover:bg-slate-700 text-xs font-semibold text-slate-200 transition-colors flex items-center gap-1.5"
                  >
                    <RotateCcw className="h-3.5 w-3.5" />
                    Retake
                  </button>
                  <button
                    onClick={() => setActiveView('setup')}
                    className="px-3.5 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-xs font-bold text-white transition-colors"
                  >
                    New Quiz
                  </button>
                </div>
              </div>

              {/* Question-by-Question Review */}
              <div className="space-y-4">
                <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                  Question Breakdown & Grounded Explanations
                </h4>

                {quizResult.questions.map((item, idx) => (
                  <div
                    key={item.question_id}
                    className={`p-5 rounded-xl border transition-all ${
                      item.is_correct
                        ? 'border-emerald-500/20 bg-emerald-500/5'
                        : 'border-rose-500/30 bg-rose-500/5'
                    }`}
                  >
                    {/* Header */}
                    <div className="flex items-center justify-between gap-2 mb-2">
                      <div className="flex items-center gap-2">
                        <span
                          className={`flex h-6 w-6 items-center justify-center rounded-full text-xs font-bold border ${
                            item.is_correct
                              ? 'border-emerald-500/40 bg-emerald-500/20 text-emerald-400'
                              : 'border-rose-500/40 bg-rose-500/20 text-rose-400'
                          }`}
                        >
                          {idx + 1}
                        </span>
                        <span className="text-xs font-bold text-slate-300">
                          {item.question_text}
                        </span>
                      </div>
                      <span
                        className={`text-[11px] font-bold px-2 py-0.5 rounded border flex items-center gap-1 ${
                          item.is_correct
                            ? 'border-emerald-500/30 bg-emerald-500/10 text-emerald-300'
                            : 'border-rose-500/30 bg-rose-500/10 text-rose-300'
                        }`}
                      >
                        {item.is_correct ? (
                          <>
                            <CheckCircle2 className="h-3 w-3" /> Correct
                          </>
                        ) : (
                          <>
                            <XCircle className="h-3 w-3" /> Incorrect
                          </>
                        )}
                      </span>
                    </div>

                    {/* Answers Comparison */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-2 my-3 text-xs">
                      <div
                        className={`p-2.5 rounded-lg border ${
                          item.is_correct
                            ? 'border-emerald-500/20 bg-emerald-500/10 text-emerald-200'
                            : 'border-rose-500/20 bg-rose-500/10 text-rose-200'
                        }`}
                      >
                        <span className="block text-[10px] uppercase font-bold text-slate-400">
                          Your Answer
                        </span>
                        <span className="font-semibold">{item.user_answer || '(No answer provided)'}</span>
                      </div>

                      <div className="p-2.5 rounded-lg border border-indigo-500/20 bg-indigo-500/10 text-indigo-200">
                        <span className="block text-[10px] uppercase font-bold text-indigo-400">
                          Correct Answer
                        </span>
                        <span className="font-semibold">{item.correct_answer}</span>
                      </div>
                    </div>

                    {/* Explanation */}
                    <div className="p-3 rounded-lg bg-slate-900/60 border border-slate-800 text-xs text-slate-300 mt-2">
                      <span className="font-bold text-indigo-400 block mb-1">Explanation:</span>
                      {item.explanation}
                    </div>

                    {/* Grounded Document Citation */}
                    {item.source_excerpt && (
                      <div className="p-3 rounded-lg bg-slate-950/40 border border-slate-800/80 text-xs text-slate-400 mt-2 flex items-start gap-2">
                        <Quote className="h-4 w-4 text-slate-500 shrink-0 mt-0.5" />
                        <div>
                          <div className="flex items-center gap-2 mb-1">
                            <span className="text-[10px] font-bold uppercase text-slate-400">
                              Supporting Source Excerpt
                            </span>
                            {item.page_number && (
                              <span className="text-[10px] font-semibold text-slate-400 bg-slate-800 px-1.5 py-0.2 rounded border border-slate-700">
                                Page {item.page_number}
                              </span>
                            )}
                          </div>
                          <span className="italic text-slate-300">"{item.source_excerpt}"</span>
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* VIEW 4: PAST ATTEMPTS HISTORY */}
          {activeView === 'history' && (
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
                  <Clock className="h-3.5 w-3.5 text-indigo-400" />
                  Past Attempts History
                </h4>
                <button
                  onClick={() => setActiveView('setup')}
                  className="text-xs text-indigo-400 hover:text-indigo-300"
                >
                  ← Back to Setup
                </button>
              </div>

              {isLoadingHistory ? (
                <div className="flex justify-center p-8">
                  <Loader2 className="h-6 w-6 animate-spin text-indigo-400" />
                </div>
              ) : pastAttempts.length === 0 ? (
                <div className="p-8 text-center text-xs text-slate-500 border border-slate-800 rounded-xl">
                  No previous attempts recorded for this quiz yet.
                </div>
              ) : (
                <div className="space-y-3">
                  {pastAttempts.map((attempt) => (
                    <div
                      key={attempt.id}
                      className="p-4 rounded-xl border border-slate-800 bg-slate-900/40 flex items-center justify-between"
                    >
                      <div>
                        <div className="flex items-center gap-2">
                          <span className="text-sm font-bold text-white">
                            {attempt.percentage}%
                          </span>
                          <span
                            className={`text-[10px] font-bold px-1.5 py-0.5 rounded border uppercase ${
                              attempt.percentage >= 70
                                ? 'border-emerald-500/30 bg-emerald-500/10 text-emerald-400'
                                : 'border-amber-500/30 bg-amber-500/10 text-amber-400'
                            }`}
                          >
                            {attempt.percentage >= 70 ? 'Passed' : 'Needs Review'}
                          </span>
                        </div>
                        <p className="text-[11px] text-slate-400 mt-1">
                          Scored {attempt.score}/{attempt.total_points} points on{' '}
                          {new Date(attempt.completed_at).toLocaleString()}
                        </p>
                      </div>

                      <button
                        onClick={() => {
                          setQuizResult({
                            attempt_id: attempt.id,
                            quiz_id: attempt.quiz_id,
                            score: attempt.score,
                            total_points: attempt.total_points,
                            percentage: attempt.percentage,
                            passed: attempt.percentage >= 70,
                            completed_at: attempt.completed_at,
                            questions: attempt.review_details,
                          });
                          setActiveView('results');
                        }}
                        className="px-3 py-1.5 rounded-lg border border-slate-700 bg-slate-800 hover:bg-slate-700 text-xs font-semibold text-slate-200 transition-colors"
                      >
                        Review Answers
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
