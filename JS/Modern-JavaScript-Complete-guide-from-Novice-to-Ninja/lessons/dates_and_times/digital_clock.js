const clock = document.querySelector(".clock");

const tick = () => {
  const now = new Date();
  const hrs = now.getHours();
  const mins = now.getMinutes();
  const secs = now.getSeconds();
  const html = `
        <span>${hrs}</span> : 
        <span>${mins}</span> : 
        <span>${secs}</span>
    `;
  clock.innerHTML = html;
};
setInterval(tick, 1);
