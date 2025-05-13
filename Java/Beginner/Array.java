public class Array{
    public static void main(String[] args){
        String[] cars = {"Camaro", "Corvette", "Tesla"};
        cars[0] = "Mustang";
        System.out.println(cars[0]);
        // Another Method
        String[] cars_2 = new String[4];
        cars_2[0] = "Camaro";
        cars_2[1] = "Corvette";
        cars_2[2] = "Tesla";
        System.out.println(cars.length);
        return;
    }
}