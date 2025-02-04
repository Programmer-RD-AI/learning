const key = "pS8euTRb9TEA8LwTUoRDPVhUsqvENABz";
const getCity = async (cityName) => {
  const base = "http://dataservice.accuweather.com/locations/v1/cities/search";
  const query = `?apikey=${key}&q=${cityName}`;
  const response = await fetch(base + query);
  const data = await response.json();
  return data[0];
};
const getConditions = async (cityCode) => {
  const base = `http://dataservice.accuweather.com/currentconditions/v1/`;
  const query = `${cityCode}?apikey=${key}`;
  const response = await fetch(base + query);
  const data = await response.json();
  return data;
};
