package application;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;

class moveFile {
	public static void run(String filePath, String directoryPath) throws IOException {
		Path source = Paths.get(filePath);
        Path dir = Paths.get(directoryPath);

        // ディレクトリがない場合は作成する
        if (!Files.exists(dir)) {
            Files.createDirectories(dir);
        }

        Path targetPath = dir.resolve(source.getFileName());

        // ファイルを移動し、既存のファイルを上書きする
        Files.move(source, targetPath, StandardCopyOption.REPLACE_EXISTING);
    }
}