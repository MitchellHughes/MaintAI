import { Outlet } from "react-router-dom";

import AppSidebar from "./AppSidebar";
import { SidebarInset, SidebarProvider } from "@/components/ui/sidebar";

function Layout() {
  return (
    <SidebarProvider defaultOpen className="overflow-x-hidden">
      <AppSidebar />
      <SidebarInset className="min-w-0 flex-1 overflow-x-hidden">
        <div className="flex min-h-svh min-w-0 flex-1 flex-col overflow-x-hidden">
          <div className="flex w-full min-w-0 max-w-full flex-1 flex-col overflow-x-hidden p-4 md:p-6 lg:p-8">
            <Outlet />
          </div>
        </div>
      </SidebarInset>
    </SidebarProvider>
  );
}

export default Layout;
