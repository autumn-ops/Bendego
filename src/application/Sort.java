package application;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

import javafx.scene.control.Alert;
import javafx.scene.control.ButtonType;

class Sort implements Runnable {

	private final String mode;
	private final File inPath;
	private final File outPath;

	public Sort(String mode, File inPath, File outPath) {
		this.mode = mode;
		this.inPath = inPath;
		this.outPath = outPath;
	}
	
    private static final String RESTART_FILE = "Sort_restart_point.txt";
    private int restartIndex;

	ArrayList<File> file_list = new ArrayList<File>();
	String scad;

	@Override
	public void run() {
		restartIndex = loadRestartPoint();
		
		ArrayList<File> list = new ArrayList<File>();

		DumpFile df = new DumpFile();
		list = df.run(inPath);

		String color = loadSettings();
   	 
		//名前の確認
		if(Controller.check_box == false){
			Controller.workin_hard.set("名前の確認");
			Check check = new Check();
			list = check.run(list);
		}

		Controller.workin_hard.set("Sort");

		//JPG以外を除外
		for(File f: list) {
			if(f.getName().substring(f.getName().lastIndexOf("."), f.getName().length()).equals(".jpg")) {
					if(f.getName().substring(0, f.getName().indexOf("_")+1).equals("04_")) {
						scad = "04";
					}else {
						scad = "01";
					}
					file_list.add(f);
					Controller.indicator.set(0 + " / " + file_list.size());

			}
		}

		file_list.sort(null);

		
		//Sortの処理
		Controller.workin_hard.set("Sort");
		
		File outDir = new File(inPath.toString() + Controller.separator + "Result");
		
		//outputフォルダの名前を探索
		if(Controller.txtf.equals("")) {
			File dir = new File(inPath.toString());
			int i = 1;
			while(true) {
				if (dir.exists()) {
					i++;
					dir = new File(inPath.toString() + i);
				}else {
					outDir = dir;
					break;
				}
			}
		}
		
		//画像の処理
		File savePath;

		for(int i=restartIndex;i<file_list.size();i++) {
			
            int currentIndex = i;

			Controller.indicator.set(i+1 + " / " + file_list.size());
			Controller.scpane_text.set(file_list.get(i).toString().replace(inPath.toString(), ""));
			
			String line = file_list.get(i).getName().substring(file_list.get(i).getName().indexOf("_")+1, file_list.get(i).getName().length());

			line = line.substring(0, line.indexOf("_"));

			//フォルダの作成
			savePath = new File(file_list.get(i).getParent().replace(inPath.toString(), outDir.toString()));
			
			//アイテムコードのパス
			if(!savePath.getName().equals(line) && !scad.equals("04")) {
				savePath = new File(savePath + Controller.separator + line);
			}
			
			try{
				Files.createDirectories(savePath.toPath());
			}catch(IOException e){
				Controller.scpane_error.set(savePath.toString() + "の作成に失敗しました。");
				System.out.println(e);
				javafx.application.Platform.runLater(() -> showErrorDialog(currentIndex));
			}

	   	 	InputStream is = null;
			try {
				is = getFileAsIOStream(color);
			} catch (FileNotFoundException e) {
		        System.err.println(color+"を読み込めませんでした。");
				e.printStackTrace();
			}
			
			//カラーチップの加工
			if(file_list.get(i).getName().substring(file_list.get(i).getName().lastIndexOf("_"), file_list.get(i).getName().length()).equals("_c.jpg")) {
				try {
					Resizecolor.BufferedImage(file_list.get(i), is, savePath.toString());
				} catch (IOException e) {
					Controller.scpane_error.set(file_list.get(i).toString().replace(inPath.getParent(), "") + "の作成に失敗しました。");
					e.printStackTrace();
					javafx.application.Platform.runLater(() -> showErrorDialog(currentIndex));
				}
			}
			//その他画像の加工
			else {
				try {
					Resizeetc.BufferedImage(file_list.get(i), is, savePath.toString());
				} catch (IOException e) {
					Controller.scpane_error.set(file_list.get(i).toString().replace(inPath.getParent(), "") + "の作成に失敗しました。");
					e.printStackTrace();
					javafx.application.Platform.runLater(() -> showErrorDialog(currentIndex));
				}
			}
			

	        // 処理が完了したら再開ポイントファイルを削除する
	        try {
	            Files.deleteIfExists(Paths.get(RESTART_FILE));
	        } catch (IOException e) {
	            System.err.println("Failed to delete restart point file");
	        }
		}
		if(Controller.thumb_box) {
			Thumb thumb = new Thumb("Thumb", outDir, outDir.getParentFile());
			thumb.run();
		}else {
			Controller.workin_hard.set("END");
		}
	}
	
	private InputStream getFileAsIOStream(String name) throws FileNotFoundException {
        File file = new File("res/"+name);
        InputStream ioStream = new FileInputStream(file);
        
        return ioStream;
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
    public static String loadSettings() {
        String[] settings = new String[2];
        try {
            File settingsFile = new File("settings.txt");
            if (settingsFile.exists()) {
                List<String> lines = Files.readAllLines(settingsFile.toPath());
                for (String line : lines) {
                    String[] parts = line.split("=");
                    if (parts.length == 2) {
                        if (parts[0].equals("ColorSpace")) {
                            settings[0] = parts[1];
                        } else {
                        	
                        }
                    }
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return settings[0].replace(" ", "");
    }
}