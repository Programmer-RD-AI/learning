public class Switch{
    public static void main(String[] args){
        // switch is a statement that allows a variable is tested for equality
        String day = "Friday";
        switch (day){
            case "Sunday": System.out.println("It is Sunday!"); break;
            case "Monday": System.out.println("It is Monday!"); break;
            case "Tuesday": System.out.println("It is Tuesday!"); break;
            case "Wednesday": System.out.println("It is Wednesday!"); break;
            case "Thursday": System.out.println("It is Thursday!"); break;
            default: System.out.println("That is not a day! :)");
        }
    }
}