package composite;

public class ComputerTest {
    public static void main(String[] args){
        Component hd =  new Leaf(4000, "HDD");
        Component mouse = new Leaf(4000, "Mouse");
        Component monitor = new Leaf(4000, "Monitor");
        Component ram = new Leaf(4000, "RAM");
        Component cpu = new Leaf(4000, "CPU");

        Composite ph = new Composite("Peri");
        Composite cabinet = new Composite("Cabinet");
        Composite mb = new Composite("MB");
        Composite computer = new Composite("Computer");

        mb.addComponent(cpu);
        mb.addComponent(ram);
        ph.addComponent(mouse);
        ph.addComponent(monitor);
        cabinet.addComponent(hd);
        cabinet.addComponent(mb);
        computer.addComponent(ph);
        computer.addComponent(cabinet);

        computer.showPrice();
        return;
    }
}
