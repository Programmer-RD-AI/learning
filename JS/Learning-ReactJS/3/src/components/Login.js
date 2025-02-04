import React from "react";
import { login, logout } from "../features/user";
import { useDispatch } from "react-redux";

export default function Login() {
  const dispatch = useDispatch();
  return (
    <div>
      <button
        onClick={() => {
          dispatch(
            login({ name: "Ranuga", age: 20, email: "go2ranuga@gmail.com" })
          );
        }}
      >
        Login
      </button>
      <button
        onClick={() => {
          dispatch(logout());
        }}
      >
        Logout
      </button>
    </div>
  );
}
