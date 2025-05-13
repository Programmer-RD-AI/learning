public class MyThread extends Thread{
    @Override
    public void run() {
        this.isDaemon();
        System.out.println("this threading is running");
    }


}
