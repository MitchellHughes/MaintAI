import { NavLink, useLocation } from "react-router-dom";
import {
  Activity,
  ClipboardList,
  History,
  LayoutDashboard,
  Sparkles,
  Wrench,
} from "lucide-react";

import { SidebarThemeToggle } from "./ThemeToggle";
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarRail,
  SidebarSeparator,
} from "@/components/ui/sidebar";

const links = [
  { to: "/", label: "Analysis workspace", icon: LayoutDashboard, end: true },
  { to: "/results", label: "Results & review", icon: ClipboardList },
  { to: "/history", label: "Run history", icon: History },
  { to: "/health", label: "System status", icon: Activity },
];

function AppSidebar() {
  const location = useLocation();

  return (
    <Sidebar variant="inset" collapsible="icon">
      <SidebarHeader>
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton size="lg" asChild>
              <NavLink to="/">
                <div className="flex aspect-square size-8 items-center justify-center rounded-lg bg-sidebar-primary text-sidebar-primary-foreground">
                  <Sparkles className="size-4" />
                </div>
                <div className="grid flex-1 text-left text-sm leading-tight">
                  <span className="truncate font-semibold">MaintAI</span>
                  <span className="truncate text-xs text-muted-foreground">
                    Failure mechanism NLP
                  </span>
                </div>
              </NavLink>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarHeader>

      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupLabel>Platform</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {links.map((link) => {
                const Icon = link.icon;
                const active = link.end
                  ? location.pathname === link.to
                  : location.pathname.startsWith(link.to);

                return (
                  <SidebarMenuItem key={link.to}>
                    <SidebarMenuButton asChild isActive={active} tooltip={link.label}>
                      <NavLink to={link.to} end={link.end}>
                        <Icon />
                        <span>{link.label}</span>
                      </NavLink>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                );
              })}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>

      <SidebarFooter>
        <SidebarThemeToggle />
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton className="text-muted-foreground" disabled>
              <Wrench />
              <span>Unsupervised ensemble</span>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
        <SidebarSeparator />
        <p className="px-2 text-xs text-muted-foreground group-data-[collapsible=icon]:hidden">
          v1.0 · RMIT Capstone
        </p>
      </SidebarFooter>
      <SidebarRail />
    </Sidebar>
  );
}

export default AppSidebar;
