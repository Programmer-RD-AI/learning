const card = document.querySelector(".card");
const details = document.querySelector(".details");
const location_form = document.querySelector("form");
const time = document.querySelector("img.time");
const icon = document.querySelector(".icon img");
const updateUI = ({ cityDets, cityWeather }) => {
  details.innerHTML = `
  <h5 class="my-3">${cityDets["EnglishName"]}</h5>
          <div class="my-3">${cityWeather[0]["WeatherText"]}</div>
          <div class="display-4 my-4">
            <span>${cityWeather[0]["Temperature"]["Metric"]["Value"]}</span>
            <span>&deg;C</span>
          </div>`;
  const iconSrc = `/imgs/icons/${cityWeather[0]["WeatherIcon"]}.svg`;
  icon.setAttribute("src", iconSrc);
  let timeSrc = cityWeather[0]["IsDayTime"]
    ? "./imgs/day.svg"
    : "./imgs/night.svg";
  time.setAttribute("src", timeSrc);
  if (card.classList.contains("d-none")) {
    card.classList.remove("d-none");
  }
};

const updateCity = async (city) => {
  const cityDets = await getCity(city);
  const cityWeather = await getConditions(cityDets["Key"]);
  return {
    cityDets,
    cityWeather,
  };
};
location_form.addEventListener("submit", (e) => {
  e.preventDefault();
  const city = location_form.city.value.trim();
  location_form.reset();
  // update the ui with the new city
  updateCity(city)
    .then((data) => {
      updateUI(data);
    })
    .catch((error) => {
      alert(error);
    });
});
