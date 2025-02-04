import { initializeApp } from "firebase/app";
import { getAnalytics } from "firebase/analytics";

const firebaseConfig = {
  apiKey: "AIzaSyBzfR1KB3sx1Iu6r1xc7JJZahIOwd4MJHo",
  authDomain: "blog-stories-654fb.firebaseapp.com",
  databaseURL: "https://blog-stories-654fb-default-rtdb.firebaseio.com",
  projectId: "blog-stories-654fb",
  storageBucket: "blog-stories-654fb.appspot.com",
  messagingSenderId: "991959463528",
  appId: "1:991959463528:web:7514ae94a5024c0812eddd",
  measurementId: "G-2DJZ8HKL2L",
};

const app = initializeApp(firebaseConfig);
const analytics = getAnalytics(app);
export default { firebaseConfig, analytics, app };
