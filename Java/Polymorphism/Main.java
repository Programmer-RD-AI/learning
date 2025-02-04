public class Main {
    public static void main(String[] args){
        Car car = new Car();
        Bicycle bicycle = new Bicycle();
        Booat booat = new Booat();

        Vehicle[] racers = {car, bicycle, booat};

        for (Vehicle r: racers){
            r.go();
        }
        return;
    }
}
