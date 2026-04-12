'use client';

import { useState } from 'react';
import { ThumbsUp, ThumbsDown, Send, CheckCircle } from 'lucide-react';
import { submitFeedback } from '@/lib/api';
import { motion } from 'framer-motion';

interface SearchFeedbackProps {
    query: string;
    totalResults: number;
}

export default function SearchFeedback({ query, totalResults }: SearchFeedbackProps) {
    const [step, setStep] = useState<'initial' | 'details' | 'submitted'>('initial');
    const [isSatisfied, setIsSatisfied] = useState<boolean | null>(null);
    const [relevantCount, setRelevantCount] = useState<number>(0);
    const [loading, setLoading] = useState(false);

    const handleInitialFeedback = (satisfied: boolean) => {
        setIsSatisfied(satisfied);
        setStep('details');
        // Jika puas, asumsi awal mungkin semua relevan, user bisa ubah nanti
        if (satisfied) setRelevantCount(totalResults); 
        else setRelevantCount(0);
    };

    const handleSubmit = async () => {
        setLoading(true);
        try {
            await submitFeedback({
                query,
                satisfied: isSatisfied!,
                relevant_count: relevantCount,
                total_results: totalResults
            });
            setStep('submitted');
        } catch (error) {
            console.error(error);
            alert("Gagal mengirim feedback");
        } finally {
            setLoading(false);
        }
    };

    if (totalResults === 0) return null;

    return (
        <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="w-full max-w-2xl mx-auto mt-12 p-6 rounded-2xl bg-white/40 dark:bg-black/40 backdrop-blur-md border border-blue-100 dark:border-blue-900/30 shadow-lg"
        >
            <h3 className="text-lg font-semibold text-center text-gray-800 dark:text-gray-200 mb-4">
                Evaluasi Hasil Pencarian
            </h3>

            {step === 'initial' && (
                <div className="flex flex-col items-center gap-4">
                    <p className="text-gray-600 dark:text-gray-400">
                        Apakah Anda puas dengan hasil pencarian untuk "{query}"?
                    </p>
                    <div className="flex gap-4">
                        <button
                            onClick={() => handleInitialFeedback(true)}
                            className="flex items-center gap-2 px-6 py-3 rounded-xl bg-green-100 text-green-700 hover:bg-green-200 dark:bg-green-900/30 dark:text-green-300 transition-colors"
                        >
                            <ThumbsUp className="w-5 h-5" /> Puas
                        </button>
                        <button
                            onClick={() => handleInitialFeedback(false)}
                            className="flex items-center gap-2 px-6 py-3 rounded-xl bg-red-100 text-red-700 hover:bg-red-200 dark:bg-red-900/30 dark:text-red-300 transition-colors"
                        >
                            <ThumbsDown className="w-5 h-5" /> Tidak Puas
                        </button>
                    </div>
                </div>
            )}

            {step === 'details' && (
                <div className="flex flex-col items-center gap-6 animate-in fade-in slide-in-from-bottom-4">
                    <div className="w-full max-w-md">
                        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2 text-center">
                            {isSatisfied 
                                ? "Seberapa banyak dokumen yang AKURAT/RELEVAN?" 
                                : "Jika ada, berapa dokumen yang sebenarnya RELEVAN?"}
                        </label>
                        
                        <div className="flex items-center gap-4">
                            <span className="text-sm font-bold text-blue-600 dark:text-blue-400 w-8 text-center">{relevantCount}</span>
                            <input
                                type="range"
                                min="0"
                                max={totalResults}
                                value={relevantCount}
                                onChange={(e) => setRelevantCount(parseInt(e.target.value))}
                                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700 accent-blue-600"
                            />
                            <span className="text-sm text-gray-500 w-8 text-center">{totalResults}</span>
                        </div>
                        <p className="text-xs text-center text-gray-500 mt-2">
                            Geser slider untuk menentukan jumlah (0 - {totalResults})
                        </p>
                    </div>

                    <div className="flex gap-3">
                        <button
                            onClick={() => setStep('initial')}
                            className="px-4 py-2 text-sm text-gray-500 hover:text-gray-700 dark:text-gray-400 transition-colors"
                        >
                            Kembali
                        </button>
                        <button
                            onClick={handleSubmit}
                            disabled={loading}
                            className="flex items-center gap-2 px-6 py-2 rounded-xl bg-blue-600 text-white hover:bg-blue-700 transition-colors disabled:opacity-50"
                        >
                            {loading ? "Mengirim..." : (
                                <>
                                    <Send className="w-4 h-4" /> Kirim Evaluasi
                                </>
                            )}
                        </button>
                    </div>
                </div>
            )}

            {step === 'submitted' && (
                <div className="flex flex-col items-center gap-2 text-green-600 dark:text-green-400 animate-in zoom-in">
                    <CheckCircle className="w-12 h-12 mb-2" />
                    <p className="font-medium">Terima kasih atas feedback Anda!</p>
                    <p className="text-sm text-gray-500">Masukan Anda membantu kami belajar.</p>
                </div>
            )}
        </motion.div>
    );
}