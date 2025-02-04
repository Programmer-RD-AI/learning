import React from "react";

type HeadingProps = {
  children: string;
};
export default function Heading(props: HeadingProps) {
  return <h2>{props.children}</h2>;
}
