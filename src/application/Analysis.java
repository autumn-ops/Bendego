package application;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;

import javafx.scene.control.Alert;
import javafx.scene.control.ButtonType;

class Analysis implements Runnable {
	
	private final String mode;
    private final File inPath;
    private final File outPath;

    public Analysis(String mode, File inPath, File outPath) {
        this.mode = mode;
        this.inPath = inPath;
        this.outPath = outPath;
    }
	
	ArrayList<File> excel_list = new ArrayList<File>();
	Analysis_data ad = new Analysis_data();
	int all_count = 0;

    private static final String RESTART_FILE = "Analysis_restart_point.txt";
    private int restartIndex;

	@Override
	public void run() {

		Controller.workin_hard.set("Analysis");

        restartIndex = loadRestartPoint();
        ArrayList<File> list = new ArrayList<File>();
        if(inPath.isDirectory()) {
        	DumpFile df = new DumpFile();
        	list = df.run(inPath);
        }else {
        	list.add(inPath);
        }
		
		//.xlsxファイルだけ取り出す
		for(File f: list) {if(f.toString().substring(f.toString().lastIndexOf(".")).equals(".xlsx")) {excel_list.add(f);}}
		
		if(excel_list.size() == 0) {Controller.scpane_error.set("ファイルが'.xlsx'か確認してください。");}

        //メインの処理
		for (int i = restartIndex; i < excel_list.size(); i++) {
        	Controller.indicator.set(i+1 + " / " + excel_list.size() + Controller.newline + " " + all_count + "行");
            File f = excel_list.get(i);
            try {
            	
            	Controller.scpane_text.set(f.getParentFile().getName() + " / " + f.getName());
            	all_count += ad.getLine(f);

            	Controller.indicator.set(i+1 + " / " + excel_list.size() + Controller.newline + " " + all_count + "行");
                
            } catch (Exception e) {
                int currentIndex = i;
                Controller.scpane_text.set("読み込めませんでした。\nファイル名の確認、シート名が原稿①になっているか確認してください\n");
                javafx.application.Platform.runLater(() -> showErrorDialog(currentIndex));
                break;
            }
        }

        // 処理が完了したら再開ポイントファイルを削除する
        try {
            Files.deleteIfExists(Paths.get(RESTART_FILE));
        } catch (IOException e) {
            System.err.println("Failed to delete restart point file");
        }
        Controller.workin_hard.set("END");
	}
	
    private void showErrorDialog(int index) {
        Alert alert = new Alert(javafx.scene.control.Alert.AlertType.ERROR, "Error processing file. Please fix the issue and press OK to continue.", ButtonType.OK);
        alert.showAndWait().ifPresent(response -> {
            if (response == ButtonType.OK) {
                saveRestartPoint(index);
                restartIndex = index;  // エラーしたファイルのインデックス
                run();
            }
        });
    }

    private void saveRestartPoint(int index) {
        try {
            Files.write(Paths.get(RESTART_FILE), Integer.toString(index).getBytes());
        } catch (IOException e) {
        	Controller.scpane_text.set("作業記録を保存に失敗しました。");
            System.err.println("Failed to save restart point");
        }
    }

    private int loadRestartPoint() {
        if (Files.exists(Paths.get(RESTART_FILE))) {
            try {
                String content = new String(Files.readAllBytes(Paths.get(RESTART_FILE)));
                return Integer.parseInt(content);
            } catch (IOException | NumberFormatException e) {
            	Controller.scpane_text.set("作業記録を読み込みに失敗しました。");
                System.err.println("Failed to load restart point, starting from the beginning");
            }
        }
        return 0;
    }
}