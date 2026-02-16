import { useEffect, useState } from "react";

export default function Dashboard() {
  const [wallets, setWallets] = useState([]);

  useEffect(() => {
    fetch("/api/finance/wallets/", { credentials: "include" })
      .then(r => r.json())
      .then(setWallets);
  }, []);

  return (
    <div>
      <h1>Dashboard</h1>
      <ul>
        {wallets.map(w => (
          <li key={w.id}>{w.kind}: {w.balance}</li>
        ))}
      </ul>
    </div>
  );
}
