package getalp.wsd.ufsac.utils;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;

public class File
{
    public static boolean exists(String filePath)
    {
        return Files.exists(Paths.get(filePath));
    }
    
    public static void moveFile(String from, String to)
    {
        try
        {
            Files.move(Paths.get(from), Paths.get(to), StandardCopyOption.REPLACE_EXISTING);
        }
        catch (Exception e)
        {
            throw new RuntimeException(e);
        }
    }

    public static void removeFile(String filepath)
    {
        try
        {
            Files.delete(Paths.get(filepath));
        }
        catch (Exception e)
        {
            throw new RuntimeException(e);
        }
    }
    
    public static String createTemporaryFileName()
    {
        try
        {
            return Files.createTempFile(null, null).toFile().getAbsolutePath();
        }
        catch (Exception e)
        {
            throw new RuntimeException(e);
        }
    }
    
    public static String createTemporaryFileName(String directoryPath)
    {
        try
        {
            return Files.createTempFile(Paths.get(directoryPath), null, null).toFile().getAbsolutePath();
        }
        catch (Exception e)
        {
            throw new RuntimeException(e);
        }
    }
}
