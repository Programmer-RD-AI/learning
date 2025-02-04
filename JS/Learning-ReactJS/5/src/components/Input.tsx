import React from "react";
import { InputProps } from "./Input.types";

export default function Input({ value, handleChange }: InputProps) {
  const handleInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    console.log(event);
  };
  return <input type="text" value={value} onChange={handleChange} />;
}
