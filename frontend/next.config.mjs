/** @type {import('next').NextConfig} */
const nextConfig = {
    rewrites: async () => {
        const isDev = process.env.NODE_ENV === 'development';
        const backendUrl = process.env.BACKEND_URL || (isDev ? 'http://localhost:5000' : 'http://backend:5000');
        console.log('Using Backend URL:', backendUrl);
        return [
            {
                source: '/api/:path*',
                destination: `${backendUrl}/:path*`, // Proxy to Backend
            },
            {
                source: '/files/:path*',
                destination: `${backendUrl}/files/:path*`, // Proxy to Backend
            },
        ]
    },
};

export default nextConfig;
