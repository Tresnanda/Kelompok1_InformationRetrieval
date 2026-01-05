'use client';

import { SearchResult } from '@/lib/api';
import { motion } from 'framer-motion';
import { FileText, Download, Eye } from 'lucide-react';
// Import komponen baru
import SearchFeedback from './SearchFeedback';
// Asumsi kamu menambahkan prop 'query' dari parent juga untuk dikirim ke feedback


interface SearchResultsProps {
    results: SearchResult[];
    query: string;
}

export default function SearchResults({ results, query }: SearchResultsProps) {

    if (results.length === 0) return null;

    return (
        <div className="w-full max-w-5xl mx-auto space-y-6 pb-20">
            {/* Bagian Header Hasil Search (KODE LAMA TETAP SAMA) */}
            <div className="flex items-center justify-between px-2">
                <h2 className="text-xl font-semibold text-gray-800 dark:text-gray-200">
                    Search Results <span className="text-gray-400 font-normal">({results.length})</span>
                </h2>
            </div>

            {/* Grid Hasil Search (KODE LAMA TETAP SAMA) */}
            <div className="grid gap-6">
                {results.map((result, index) => (
                    <motion.div
                        key={index}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.4, delay: index * 0.1, ease: "easeOut" }}
                        className="group relative glass-card rounded-2xl p-6 hover:-translate-y-1 hover:shadow-2xl overflow-hidden"
                    >
                        {/* ... ISI KARTU TETAP SAMA SEPERTI FILE KAMU ... */}

                        <div className="flex flex-col md:flex-row gap-6 relative z-10">
                            {/* ... Konten Icon ... */}
                            <div className="hidden md:flex flex-col items-center justify-center w-16 h-16 rounded-2xl bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 border border-blue-100 dark:border-blue-800 shrink-0">
                                <FileText className="w-8 h-8 text-blue-600 dark:text-blue-400" />
                            </div>

                            {/* Content */}
                            <div className="flex-1 min-w-0">
                                <h3 className="text-xl font-bold text-gray-900 dark:text-white leading-tight mb-3 group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors">
                                    {result.title}
                                </h3>
                                <div className="text-sm text-gray-500 mb-2">{result.filename}</div>

                                {/* Score pills, dll tetap sama */}
                            </div>

                            {/* Actions Buttons tetap sama */}
                            <div className="flex flex-row md:flex-col gap-2 shrink-0 pt-4 md:pt-0 justify-end md:justify-center">
                                <div className="flex flex-row md:flex-col gap-2 shrink-0 pt-4 md:pt-0 border-t md:border-t-0 md:border-l border-gray-100 dark:border-gray-800 md:pl-6 justify-end md:justify-center">
                                    <a
                                        href={`/files/${result.filename}`}
                                        target="_blank"
                                        rel="noopener noreferrer"
                                        className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-gray-900 dark:bg-white text-white dark:text-gray-900 text-sm font-medium hover:bg-blue-600 dark:hover:bg-blue-100 transition-colors shadow-lg shadow-gray-200 dark:shadow-none"
                                    >
                                        <Eye className="w-4 h-4" />
                                        View PDF
                                    </a>
                                    <a
                                        href={`/files/${result.filename}`}
                                        download
                                        className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 text-sm font-medium hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
                                    >
                                        <Download className="w-4 h-4" />
                                        Save
                                    </a>
                                </div>
                            </div>
                        </div>
                    </motion.div>
                ))}
            </div>

            {/* --- TAMBAHKAN INI DI BAGIAN BAWAH --- */}
            <div className="pt-10 border-t border-gray-200 dark:border-gray-800 mt-10">
                <SearchFeedback query={query} totalResults={results.length} />
            </div>
        </div>
    );
}