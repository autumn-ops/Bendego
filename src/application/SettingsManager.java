package application;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

public class SettingsManager {

    private static final String SETTINGS_FILE = "settings.txt";

    public static void saveSettings(String colorSpace, String excelFile) {
        try {
            FileWriter writer = new FileWriter(SETTINGS_FILE);
            writer.write("ColorSpace=" + colorSpace + Controller.newline);
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static String[] loadSettings() {
        String[] settings = new String[2];
        try {
            File settingsFile = new File(SETTINGS_FILE);
            if (settingsFile.exists()) {
                List<String> lines = Files.readAllLines(settingsFile.toPath());
                for (String line : lines) {
                    String[] parts = line.split("=");
                    if (parts[0].equals("ColorSpace")) {
                            settings[0] = parts[1];
                        
                    }
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return settings;
    }

    public static void saveColorSpaces(List<String> colorSpaces) {
        try {
            FileWriter writer = new FileWriter("color_spaces.txt");
            for (String colorSpace : colorSpaces) {
                writer.write(colorSpace + Controller.newline);
            }
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static List<String> loadColorSpaces() {
        List<String> colorSpaces = new ArrayList<>();
        try {
            File colorSpacesFile = new File("color_spaces.txt");
            if (colorSpacesFile.exists()) {
                colorSpaces = Files.readAllLines(colorSpacesFile.toPath());
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return colorSpaces;
    }

    public static void deleteUnusedFiles(String directoryPath) {
        try {
            Path directory = Paths.get(directoryPath);
            Files.walk(directory)
                .filter(Files::isRegularFile)
                .filter(path -> !isFileUsed(path))
                .forEach(path -> {
                    try {
                        Files.delete(path);
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                });
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static boolean isFileUsed(Path path) {
        try {
            List<String> lines = Files.readAllLines(Paths.get(SETTINGS_FILE));
            return lines.stream().anyMatch(line -> line.contains(path.getFileName().toString()));
        } catch (IOException e) {
            e.printStackTrace();
            return false;
        }
    }
}
