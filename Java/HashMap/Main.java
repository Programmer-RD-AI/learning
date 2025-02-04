import java.util.HashMap;

public class Main {
    public static void main(){
        //  HashMap implements a Map Interface (need import)
        //  HashMap is similar to ArrayList, but with key value pairs
        //  Stores objects, need to use wrapper class
        //  ex: (name, email), (usename, userID), (country, capital)

        HashMap<String, String> countries = new HashMap<>();
        // add a key and value
        countries.put("USA", "DC");
        countries.put("Sri Lanka", "Colombo");
        countries.put("India", "New Delhi");

        countries.remove("USA");

        countries.clear();

        countries.size();

        countries.replace("USA", "NYU");

        countries.containsKey("USA");

        for (String i: countries.keySet()){

        }
        return;
    }
}
