import { useState } from "react";
import { Link, Outlet, useSearchParams } from "react-router-dom";

export function BookLayout() {
  const [searchParams, setSearchParams] = useSearchParams({ n: 3 });
  // const [number, setNumber] = useState(3);
  const number = searchParams.get("n");
  return (
    <>
      <Link to="/books/1">Book 1</Link>
      <Link to="/books/2">Book 2</Link>
      <Outlet context={{ hello: "World" }} />
      <input
        type="number"
        value={number}
        onChange={(e) => setSearchParams({ n: e.target.value })}
      />
    </>
  );
}
