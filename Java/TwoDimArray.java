public class TwoDimArray{
    public static void main(String[] args){
        String[][] cars = new String[3][3];
        cars[0][0] = "Camaro";
        cars[0][1] = "Corvette";
        cars[0][2] = "Silverado";
        for (int i = 0; i<cars.length; i++){
            System.out.println();
            for (int j=0; j<cars[i].length; j++){
                System.out.print(cars[i][j] + " ");
            }
        }
        return;
    }
}