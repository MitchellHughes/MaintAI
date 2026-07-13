import { cn } from "@/lib/utils";

export default function PageShell({ children, className }) {
  return (
    <div className={cn("flex w-full min-w-0 max-w-full flex-col gap-6 overflow-x-hidden lg:gap-8", className)}>
      {children}
    </div>
  );
}
