import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "LTX Video Studio",
  description: "Personal AI video generator",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body style={{ margin: 0, background: "#fafafa" }}>{children}</body>
    </html>
  );
}
