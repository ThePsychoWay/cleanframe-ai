import { AuthGate } from "@/components/AuthGate";
import { AuthProvider } from "@/components/AuthProvider";

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  return (
    <AuthProvider>
      <AuthGate>{children}</AuthGate>
    </AuthProvider>
  );
}
