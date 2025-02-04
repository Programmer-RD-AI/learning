public class Vehicle {
    double speed;
    String typeOfVehicle = "Vehicle";
    void go(){
        System.out.println("This " + this.typeOfVehicle + "is moving");
    }

    void stop(){
        System.out.println("The " + this.typeOfVehicle + " is stopped");
    }
}
