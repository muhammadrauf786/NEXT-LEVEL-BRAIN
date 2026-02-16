export default function WalletsPanel({ wallets }) {
  return (
    <div>
      <h2>Wallets</h2>
      <ul>
        {wallets.map(w => (
          <li key={w.id}>{w.kind}: {w.balance}</li>
        ))}
      </ul>
    </div>
  );
}
