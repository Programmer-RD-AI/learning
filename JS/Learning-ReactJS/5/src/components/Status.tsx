import React from "react";

type StatusProp = {
  status: "loading" | "sucess" | "error";
};
export default function Status(props: StatusProp) {
  let message;
  if (props.status === "loading") {
    message = "Loading";
  } else if (props.status === "sucess") {
    message = "Data feteched successfully";
  } else if (props.status === "error") {
    message = "Error fetching data";
  }
  return (
    <div>
      <h2>Loading...</h2>
      <h2>Data fetched successfully!</h2>
      <h2>Error fetching data</h2>
    </div>
  );
}
