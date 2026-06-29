import { AlertCircle } from "lucide-react";

export default function Disclaimer() {
  return (
    <div className="disclaimer mt-3">
      <AlertCircle className="h-3 w-3 text-amber-500" />
      <span>Demo using synthetic/sample data. Not financial advice.</span>
    </div>
  );
}
