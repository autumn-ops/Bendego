package application;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Paths;

import javafx.beans.property.SimpleBooleanProperty;

class Cut implements Runnable {

    private final String mode;
    private final File inPath;
    private final File outPath;
    private Process process;
    static SimpleBooleanProperty stop = new SimpleBooleanProperty(false);

    public Cut(String mode, File inPath, File outPath) {
        this.mode = mode;
        this.inPath = inPath;
        this.outPath = outPath;
    }

    @Override
    public void run() {
        Controller.workin_hard.set("Cut");

        stop.addListener((observableValue, oldValue, newValue) -> {
            System.out.println("作業の終了");
            stopProcess();
        });

        try {
            // プロジェクトのルートディレクトリからの相対パスを指定
            String pythonScriptPath = Paths.get("src", "application", "fast_seg.py").toString();

            // カレントディレクトリをプロジェクトのルートディレクトリに設定
            ProcessBuilder pb = new ProcessBuilder("python", pythonScriptPath, inPath.toString(), outPath.toString());

            pb.directory(new java.io.File(System.getProperty("user.dir"))); // プロジェクトのルートディレクトリ

            process = pb.start();

            // Pythonの標準出力を読み取る
            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8));
            String line;
            while ((line = reader.readLine()) != null) {
                Controller.scpane_text.set(line);
                System.out.println(line);
            }

            // Pythonのエラーを読み取る
            BufferedReader error = new BufferedReader(new InputStreamReader(process.getErrorStream(), StandardCharsets.UTF_8));
            while ((line = error.readLine()) != null) {
                Controller.scpane_text.set("ERROR: " + line);
                System.err.println("ERROR: " + line);
            }

            // プロセスの終了を待つ
            int exitCode = process.waitFor();
            Controller.scpane_text.set("Exited with code: " + exitCode);
            System.out.println("Exited with code: " + exitCode);

        } catch (Exception e) {
            Controller.scpane_text.set("Error: " + e.getMessage());
            e.printStackTrace();
        }
        Controller.workin_hard.set("END");
    }

    // プロセスを終了
    public void stopProcess() {
        if (process != null) {
            process.destroy();
        }
    }
}
