import { AuthProvider } from "@/components/AuthProvider";
import { Faq } from "@/components/Faq";
import { Features } from "@/components/Features";
import { FinalCta } from "@/components/FinalCta";
import { Footer } from "@/components/Footer";
import { Hero } from "@/components/Hero";
import { HowItWorks } from "@/components/HowItWorks";
import { LandingHeader } from "@/components/LandingHeader";
import { UseCases } from "@/components/UseCases";

export default function LandingPage() {
  return (
    <AuthProvider>
      <LandingHeader />
      <main id="main">
        <Hero />
        <HowItWorks />
        <UseCases />
        <Features />
        <Faq />
        <FinalCta />
      </main>
      <Footer />
    </AuthProvider>
  );
}
