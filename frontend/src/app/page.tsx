'use client';

import { useState } from 'react';
import { Search, Sparkles, Command } from 'lucide-react';
import { search, SearchResponse } from '@/lib/api';
import SearchResults from '@/components/SearchResults';
import { cn } from '@/lib/utils';
import { motion, AnimatePresence } from 'framer-motion';

export default function Home() {
    const [query, setQuery] = useState('');
    const [loading, setLoading] = useState(false);
    const [data, setData] = useState<SearchResponse | null>(null);
    const [hasSearched, setHasSearched] = useState(false);

    const handleSearch = async (e?: React.FormEvent) => {
        e?.preventDefault();
        if (!query.trim()) return;

        setLoading(true);
        setHasSearched(true);
        try {
            const res = await search(query);
            setData(res);
        } catch (err) {
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    const applyCorrection = (correction: string) => {
        setQuery(correction);
        setLoading(true);
        search(correction).then(res => {
            setData(res);
        }).finally(() => setLoading(false));
    };

    return (
        <main className="min-h-screen bg-gray-50 dark:bg-[#050505] relative overflow-hidden selection:bg-blue-200 dark:selection:bg-blue-900/50">

            {/* Animated Background Blobs */}
            <div className="fixed inset-0 overflow-hidden pointer-events-none">
                <div className="absolute top-0 left-1/4 w-96 h-96 bg-purple-300 dark:bg-purple-900/20 rounded-full mix-blend-multiply filter blur-3xl opacity-30 animate-blob" />
                <div className="absolute top-0 right-1/4 w-96 h-96 bg-blue-300 dark:bg-blue-900/20 rounded-full mix-blend-multiply filter blur-3xl opacity-30 animate-blob animation-delay-2000" />
                <div className="absolute -bottom-32 left-1/2 w-96 h-96 bg-indigo-300 dark:bg-indigo-900/20 rounded-full mix-blend-multiply filter blur-3xl opacity-30 animate-blob animation-delay-4000" />
            </div>

            <div className="relative z-10 flex flex-col items-center min-h-screen">

                {/* Hero Section */}
                <motion.div
                    layout
                    className={cn(
                        "flex flex-col items-center w-full max-w-3xl px-4 transition-all duration-700 ease-in-out",
                        hasSearched ? "pt-12 pb-8" : "justify-center flex-1"
                    )}
                >
                    <motion.div layout className="text-center mb-8">
                        <motion.h1
                            layout
                            className={cn(
                                "font-bold text-gray-900 dark:text-white tracking-tight leading-tight",
                                hasSearched ? "text-3xl" : "text-5xl md:text-7xl mb-6"
                            )}
                        >
                            {hasSearched ? (
                                "Search Results"
                            ) : (
                                <>
                                    <span className="text-gradient">Search Engine</span>
                                    <br />
                                    Information Retrieval
                                </>
                            )}
                        </motion.h1>

                        {!hasSearched && (
                            <motion.p
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 1 }}
                                transition={{ delay: 0.2 }}
                                className="text-lg text-gray-500 dark:text-gray-400 max-w-lg mx-auto"
                            >
                                Kelompok 1
                            </motion.p>
                        )}
                    </motion.div>

                    <motion.div
                        layout
                        className="w-full relative group"
                    >
                        <div className="absolute inset-0 bg-gradient-to-r from-blue-500 to-purple-500 rounded-2xl blur opacity-20 group-hover:opacity-30 transition-opacity duration-300" />
                        <form onSubmit={handleSearch} className="relative flex items-center">
                            <Search className={cn(
                                "absolute left-5 w-6 h-6 transition-colors z-10",
                                loading ? "text-blue-500 animate-pulse" : "text-gray-400"
                            )} />
                            <input
                                type="text"
                                value={query}
                                onChange={(e) => setQuery(e.target.value)}
                                placeholder="Find document titles, topics, or abstract keywords..."
                                className="w-full pl-14 pr-6 py-5 bg-white dark:bg-white/10 backdrop-blur-xl border border-gray-200 dark:border-white/10 rounded-2xl shadow-xl hover:shadow-2xl focus:shadow-2xl focus:border-blue-500/50 outline-none transition-all duration-300 text-lg placeholder:text-gray-400 dark:text-white"
                            />
                            <button
                                type="submit"
                                disabled={loading || !query.trim()}
                                className="absolute right-3 p-2.5 bg-gray-900 dark:bg-white text-white dark:text-gray-900 rounded-xl hover:scale-105 active:scale-95 transition-all disabled:opacity-50 disabled:hover:scale-100"
                            >
                                {loading ? (
                                    <div className="w-5 h-5 border-2 border-current border-t-transparent rounded-full animate-spin" />
                                ) : (
                                    <Sparkles className="w-5 h-5" />
                                )}
                            </button>
                        </form>
                    </motion.div>

                    {/* Suggestions / Correction */}
                    <AnimatePresence>
                        {data?.correction && (
                            <motion.div
                                initial={{ opacity: 0, height: 0 }}
                                animate={{ opacity: 1, height: 'auto' }}
                                exit={{ opacity: 0, height: 0 }}
                                className="mt-4 w-full"
                            >
                                <div className="p-4 bg-yellow-50/50 dark:bg-yellow-900/10 border border-yellow-200 dark:border-yellow-800/30 rounded-xl flex items-center justify-center gap-2 text-sm text-yellow-800 dark:text-yellow-200 backdrop-blur-sm">
                                    <span className="text-yellow-600 dark:text-yellow-400">Did you mean to search for:</span>
                                    <button
                                        onClick={() => applyCorrection(data.correction!)}
                                        className="font-bold underline decoration-yellow-400 decoration-2 underline-offset-2 hover:text-yellow-600 transition-colors"
                                    >
                                        {data.correction}
                                    </button>
                                </div>
                            </motion.div>
                        )}
                    </AnimatePresence>
                </motion.div>

                {/* Results Section */}
                <div className="w-full flex-1 px-4 bg-white/50 dark:bg-black/50 backdrop-blur-3xl border-t border-gray-100 dark:border-white/5 min-h-[50vh]">
                    <div className="max-w-7xl mx-auto pt-8">
                        {loading ? (
                            <div className="w-full max-w-5xl mx-auto space-y-6">
                                {[...Array(3)].map((_, i) => (
                                    <div key={i} className="h-40 bg-white dark:bg-white/5 rounded-2xl border border-gray-100 dark:border-white/5 animate-pulse" />
                                ))}
                            </div>
                        ) : (
                            data && <SearchResults results={data.data} query={query} />
                        )}

                        {hasSearched && !loading && (!data || data.data.length === 0) && (
                            <div className="text-center py-20">
                                <div className="w-20 h-20 bg-gray-100 dark:bg-gray-800 rounded-full flex items-center justify-center mx-auto mb-6">
                                    <Search className="w-10 h-10 text-gray-300 dark:text-gray-600" />
                                </div>
                                <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">No results found</h3>
                                <p className="text-gray-500">Try adjusting your keywords or search for broader topics.</p>
                            </div>
                        )}
                    </div>
                </div>

            </div>
        </main>
    );
}
