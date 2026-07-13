import { useEffect, useState } from "react";

import { checkBackend } from "../services/api";
import PageHeader from "../components/PageHeader";
import PageShell from "../components/PageShell";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

function Health() {
  const [message, setMessage] = useState("Checking…");
  const [ok, setOk] = useState(null);

  useEffect(() => {
    checkBackend()
      .then((data) => {
        setMessage(data.status);
        setOk(true);
      })
      .catch(() => {
        setMessage("Backend connection failed");
        setOk(false);
      });
  }, []);

  return (
    <PageShell>
      <PageHeader
        title="System status"
        description="API connectivity and service health for deployments and demos."
      />
      <Card className="min-w-0 shadow-sm">
        <CardHeader>
          <CardTitle className="text-lg">API</CardTitle>
          <CardDescription>Backend health endpoint</CardDescription>
        </CardHeader>
        <CardContent className="flex items-center gap-3">
          <Badge variant={ok ? "success" : ok === false ? "destructive" : "secondary"}>
            {ok === null ? "…" : ok ? "Online" : "Offline"}
          </Badge>
          <span className="text-sm">{message}</span>
        </CardContent>
      </Card>
    </PageShell>
  );
}

export default Health;
