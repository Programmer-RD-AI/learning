public class Main {
    public static void main(String[] args){
        // inner class = A class inside of another class. useful if a class should be limited in a scope.
        // usefully private but not necessaty help group classes that belong toghether
        // extremely useful for listerns for speicific proecursor for anonymous inner classes
        Outside out = new Outside();
        Outside.Inner inner = out.new Inner();
        return;
    }
}
