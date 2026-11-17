import { Navbar } from "@/components/Navbar";
import { ProjectsGrid } from "@/components/ProjectsGrid";

export default function ProjectsPage() {
  return (
    <>
      <Navbar active="projects" />
      <main id="main" className="mx-auto max-w-6xl px-4 py-10 sm:px-5">
        <ProjectsGrid />
      </main>
    </>
  );
}
