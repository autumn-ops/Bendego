package application;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Paths;

import javafx.beans.property.SimpleBooleanProperty;

class Training implements Runnable {

    private final String mode;
    private final File inPath;
    private final File outPath;
    private Process process;
    static SimpleBooleanProperty stop = new SimpleBooleanProperty(false);

    public Training(String mode, File inPath, File outPath) {
        this.mode = mode;
        this.inPath = inPath;
        this.outPath = outPath;
    }

    @Override
    public void run() {
        Controller.workin_hard.set("Training");

        stop.addListener((observableValue, oldValue, newValue) -> {
            System.out.println("トレーニングの終了");
            stopProcess();
        });
        
        File train_path  = outPath;
        if(Controller.train_path != null) {
        	train_path = Controller.train_path;
        }
        
        String batch_size = "4";
        if (isInteger(Controller.txtf)) {
        	batch_size = Controller.txtf;
        }

        try {
            // プロジェクトのルートディレクトリからの相対パスを指定
            String pythonScriptPath = Paths.get("src", "application", "make_dataset.py").toString();

            // カレントディレクトリをプロジェクトのルートディレクトリに設定
            ProcessBuilder pb = new ProcessBuilder("python", pythonScriptPath, inPath.toString(), train_path.toString(), batch_size);
            pb.directory(new java.io.File(System.getProperty("user.dir"))); // プロジェクトのルートディレクトリ

            process = pb.start();

            // Pythonスクリプトの出力をリアルタイムで読み取る
            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8));
            String line;
            while ((line = reader.readLine()) != null) {
            	Controller.scpane_text.set(line);
                System.out.println(line);
            }
            
            // Pythonスクリプトのエラーストリームをリアルタイムで読み取る
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
            e.printStackTrace();
        } finally {
            if (process != null) {
                process.destroy();
            }
            Controller.workin_hard.set("END");
        }
    }

    // プロセスを終了するメソッド
    public void stopProcess() {
        if (process != null) {
            process.destroy();
        }
    }
    
    public static boolean isInteger(String input) {
        try {
            Integer.parseInt(input);
            return true;
        } catch (NumberFormatException e) {
            return false;
        }
    }
}