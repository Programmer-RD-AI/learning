import { useState } from "react";

function App() {
  const [count, setCount] = useState(() => {
    return 0;
  });
  return (
    <>
      <button
        onClick={() => {
          // setCount(count - 1);
          setCount((prevCount) => prevCount - 1);
        }}
      >
        -
      </button>
      <span>{{ count }}</span>
      <button
        onClick={() => {
          setCount((prevCount) => prevCount + 1);
        }}
      >
        +
      </button>
    </>
  );
}

export default App;
