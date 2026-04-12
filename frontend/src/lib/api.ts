export interface SearchResult {
    title: string;
    filename: string;
    score: number;
    content_score: number;
    title_score: number;
}

export interface SearchResponse {
    status: string;
    correction: string | null;
    data: SearchResult[];
}

export interface FeedbackData {
    query: string;
    satisfied: boolean;
    relevant_count: number;
    total_results: number;
}

// Tambahkan fungsi ini di bawah function search
export async function submitFeedback(data: FeedbackData): Promise<void> {
    const res = await fetch('/api/feedback', { // Next.js rewrite akan mengarah ke backend flask
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
    });

    if (!res.ok) {
        throw new Error('Failed to submit feedback');
    }
}

export async function search(query: string): Promise<SearchResponse> {
    const res = await fetch(`/api/search?q=${encodeURIComponent(query)}`);
    if (!res.ok) {
        throw new Error('Search failed');
    }
    return res.json();
}
