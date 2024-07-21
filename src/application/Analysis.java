package application;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;

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
                Controller.scpane_text.set(f.getParentFile().getName() + " / " + f.getName()+"が読み込めませんでした。\nファイル名の確認、シート名が原稿①になっているか確認してください\n");
                saveRestartPoint(currentIndex);
                break;
            }
        }
		try {
            Files.deleteIfExists(Paths.get(RESTART_FILE));
        } catch (IOException e) {
            System.err.println("Failed to delete restart point file");
        }
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