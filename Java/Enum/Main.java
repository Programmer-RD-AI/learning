enum Planet{
    MERCURY(1),
    VENUS(2),
    EARTH(3),
    MARS(4),
    JUPITER(5),
    SATURN(6),
    URANUS(7),
    NEPTUNE(8),
    PLUTO(9);

    int number;
    Planet(int number){
        this.number = number;
    }
}

public class Main {
    public static void main(String[] args){
        // enum = enumerated (ordered listing of items in a collection) ex: days in a week and such
        Planet myPlanet = Planet.EARTH;
        System.out.println(myPlanet);
    }

    static void canILiveHere(Planet planet){
        switch (planet) {
            case Planet.EARTH:
                System.out.println("can");
                break;
            default:
                System.out.println("cant live here");
                break;
        }
    }
}
