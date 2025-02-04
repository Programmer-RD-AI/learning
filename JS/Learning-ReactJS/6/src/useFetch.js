import { useState, useEffect } from "react";
const useFetch = (url) => {
  const [data, setData] = useState(null);
  const [isPending, setIsPending] = useState(true);
  const [error, setError] = useState(null);
  useEffect(() => {
    const abortCont = new AbortController();
    setTimeout(
      fetch(url, { signal: abortCont.signal })
        .then((res) => {
          console.log(res);
          if (!res.ok) {
            throw Error("Couldnt Fetch data for that resource");
          }
          return res.json();
        })
        .then((data) => {
          setData(data);
          setIsPending(false);
          setError(null);
        })
        .catch((error) => {
          if (error.name === "AbortError") {
            console.log("Fetch Abort");
          } else {
            setError(error.message);
            setIsPending(false);
          }
        }),
      1000
    );
    return () => abortCont.abort();
  }, [url]);
  return { data, isPending, error };
};

export default useFetch;
