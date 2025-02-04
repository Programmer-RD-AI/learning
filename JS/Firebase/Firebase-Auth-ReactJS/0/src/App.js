import { getAuth, createUserWithEmailAndPassword } from "firebase/auth";
import "./App.css";
import { firebaseConfig, analytics, app } from "./firebase";
import NavBar from "./NavBar";

function App() {
  const auth = getAuth();
  createUserWithEmailAndPassword(auth, "go2ranuga@gmail.com", "password")
    .then((userCredentials) => {
      const user = userCredentials.user;
      console.log(user)
    })
    .catch((error) => {
      const errorCode = error.code;
      const errorMessage = error.message;
    });
  return <div className="App">
    <NavBar />
  </div>;
}

export default App;
