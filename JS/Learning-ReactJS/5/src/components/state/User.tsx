import React, { useState } from "react";
type AuthUser = {
  name: string;
  email: string;
};
export default function User() {
  // const [user, setUser] = useState<null | AuthUser>(null);
  const [user, setUser] = useState({} as AuthUser);
  const handleLogin = () => {
    setUser({
      name: "John",
      email: "email",
    });
  };
  const handleLogout = () => {
    setUser({} as AuthUser);
  };
  return (
    <div>
      <button onClick={handleLogin}>Login</button>
      <button onClick={handleLogout}>Logout</button>
      {/* <div>User name is {user?.name}</div>
      <div>User email is {user?.email}</div> */}
      <div>User name is {user.name}</div>
      <div>User email is {user.email}</div>
    </div>
  );
}
