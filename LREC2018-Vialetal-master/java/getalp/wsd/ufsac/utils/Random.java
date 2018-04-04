package getalp.wsd.ufsac.utils;

public class Random
{
    private static final java.util.Random random = new java.util.Random();

    public static double randomDoubleInRange(double min, double max)
    {
        return (random.nextDouble() * (max - min)) + min;
    }
    
    public static int randomInt(int max)
    {
        return random.nextInt(max);
    }

}
