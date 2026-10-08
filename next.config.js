/** @type {import('next').NextConfig} */
const nextConfig = {
  // The worker code and model weights never belong in the Vercel bundle.
  outputFileTracingExcludes: { "*": ["./kaggle/**", "./models/**"] },
};
module.exports = nextConfig;
