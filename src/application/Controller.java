package application;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.ResourceBundle;

import javafx.application.Platform;
import javafx.beans.property.SimpleStringProperty;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.control.CheckBox;
import javafx.scene.control.Label;
import javafx.scene.control.TextField;
import javafx.scene.input.DragEvent;
import javafx.scene.input.Dragboard;
import javafx.scene.input.TransferMode;
import javafx.scene.layout.AnchorPane;
import javafx.scene.layout.VBox;
import javafx.scene.text.Text;

public class Controller {

    @FXML
    private ResourceBundle resources;

    @FXML
    private URL location;

    @FXML
    private Button analysis_btn;  //行数計測ボタン

    @FXML
    private Button count_btn;  //JPGを数えるボタン

    @FXML
    private Button cut_btn;  //切り抜くボタン

    @FXML
    private Button pdf_btn;  //PDFに変換するボタン

    @FXML
    private Button thumb_btn;  //コンタクトシート作成ボタン

    @FXML
    private Button sort_btn;  //フォルダ分けボタン
    
    @FXML
    private Button train_btn;  //トレーニング用ボタン
    
    @FXML
    private Button Setting_btn;  //環境設定用ボタン

    @FXML
    private Label in_lbl;  //Input Panelのラベル

    @FXML
    private AnchorPane in_pane;  //Input Panel

    @FXML
    private Text indicator_lbl;  //進捗度を示すラベル 123/1000

    @FXML
    private CheckBox option1;  //sortの後Thumを行うかボタン  デフォルトは'ON'

    @FXML
    private CheckBox option2;  //sortの時、ファイル名のチェックをするかどうか  デフォルトは'OFF'

    @FXML
    private Label out_lbl;  //Output Panelのラベル

    @FXML
    private AnchorPane out_pane;  //Output Panel

    @FXML
    private VBox scrollpane;  //現在作業中のデータを表示するPanel  エラーもここに表示

    @FXML
    private TextField txt_field;

    @FXML
    private Label txt_title;

    @FXML
    private Text work_details;  //現在作業している内容  Sortの'Check'など

    static Thread mainThread;
    
    private String switch_light = "-fx-background-color: #239dda;" + "-fx-border-color:  #fdd23e;" + "-fx-background-radius: 10;" + "-fx-border-radius: 10;" + "-fx-text-fill: black;";
    
    private String switch_default = "-fx-background-color: #000000;" + "-fx-border-color:  #fdd23e;" + "-fx-background-radius: 10;" + "-fx-border-radius: 10;" + "-fx-text-fill: white;";
    
    static SimpleStringProperty scpane_text = new SimpleStringProperty(""); //scroll paneに出力するテキスト
    
    static SimpleStringProperty scpane_error = new SimpleStringProperty(""); //scroll paneに出力するエラー

    static SimpleStringProperty workin_hard = new SimpleStringProperty(""); //現在の作業を出力するテキスト
    
    static SimpleStringProperty indicator = new SimpleStringProperty(""); //作業状況の出力するテキスト
    
    private File in_path;

    private File out_path;

    static File train_path;
    
    private String mode = "";
    
    static String txtf = "";
    
    static boolean thumb_box;
    
    static boolean check_box;

    private static final String SETTINGS_FILE = "settings.txt";
    
	Select_Mode check_mode = new Select_Mode();  //なんのモードか確認

	static String separator = "\\";
	
	static String newline = "\n";
	
	static String colospace;
	
	static String font = "OsakaMono.ttf";

	void switch_hub(String awitch) {
		List<Button> list = new ArrayList<Button>();
		list.add(train_btn);
		list.add(analysis_btn);
		list.add(count_btn);
		list.add(cut_btn);
		list.add(pdf_btn);
		list.add(thumb_btn);
		list.add(sort_btn);
		
		for(int i=0; i<list.size();i++) {
			if(list.get(i).getText().equals(awitch)) {
				list.get(i).setStyle(switch_light);
			}else {
				list.get(i).setStyle(switch_default);
			}
		}
		
	}
	
	@FXML
    void Setting_acn(ActionEvent event) {
		Popup popup = new Popup();
        popup.showPopup();
    }

	@FXML
    void train_acn(ActionEvent event) {
		mode = "Train";
    	txt_title.setText("バッチサイズ");
    	txt_field.setPromptText("数値が低いほどメモリ使用量が低い。デフォルト'4'");
    	switch_hub(mode);
    }
	
    @FXML
    void analysis_acn(ActionEvent event) {  //PDFまたはEXCELファイルから行数を計測
    	mode = "Analysis";
    	txt_title.setText("シート名");
    	txt_field.setPromptText("読み込むシート名が'原稿①'ではない場合、入力");
    	switch_hub(mode);
    }

    @FXML
    void count_acn(ActionEvent event) {  //JPGの数を計測
    	mode = "Count";
    	txt_title.setText("出力名");
    	txt_field.setPromptText("フォルダ名を入力");
    	switch_hub(mode);
    }

    @FXML
    void cut_acn(ActionEvent event) {  //画像を切り抜くAI
    	mode = "Cut";
    	txt_title.setText("出力名");
    	txt_field.setPromptText("フォルダ名を入力");
    	switch_hub(mode);
    }

    @FXML
    void pdf_acn(ActionEvent event) {  // EXCELファイルをPDFに変換　行数も計測
    	mode = "PDF";
    	Path p1 = Paths.get("");
    	Path p2 = p1.toAbsolutePath();
    	txt_title.setText(p2.toString());
    	txt_field.setPromptText("読み込むシート名が'原稿①'ではない場合、入力");
    	switch_hub(mode);
    }

    @FXML
    void thumb_acn(ActionEvent event) {  //コンタクトシートを作成
    	mode = "Thumb";
    	txt_title.setText("出力名");
    	txt_field.setPromptText("フォルダ名を入力");
    	switch_hub(mode);
    }

    @FXML
    void sort_acn(ActionEvent event) {  //入力フォルダ内のデータを行ごとにフォルダ分け
    	mode = "Sort";
    	txt_title.setText("出力名");
    	txt_field.setPromptText("フォルダ名を入力");
    	switch_hub(mode);
    }

    @FXML
    void in_d_drop(DragEvent event) {  //入力フォルダのドロップ処理
		Dragboard board = event.getDragboard();
    	if (board.hasFiles()) {
    		in_pane.setStyle("-fx-border-color:  #fdd23e;"
    				+ "-fx-background-radius: 10;" 
    				+ "-fx-border-radius: 10;");
    		in_lbl.setText("OK");
            board.getFiles().forEach(file -> {
            	in_path = file;
            });
    	}
    }

    @FXML
    void in_d_over(DragEvent event) {  //入力フォルダの入力処理
    	Dragboard db = event.getDragboard();
		if(db.hasFiles()) {
			event.acceptTransferModes(TransferMode.COPY_OR_MOVE);
		event.consume();
		}
    }

    @FXML
    void out_d_drop(DragEvent event) {  //出力先フォルダのドロップ処理
    	Dragboard board = event.getDragboard();
    	if (board.hasFiles()) {
    		out_pane.setStyle("-fx-border-color:  #fdd23e;"
    				+ "-fx-background-radius: 10;" 
    				+ "-fx-border-radius: 10;");
    		out_lbl.setText("OK");
            board.getFiles().forEach(file -> {
            	out_path = file;
            	train_path = file;
            });
    	}
    }

    @FXML
    void out_d_over(DragEvent event) {  //出力先フォルダの入力処理
    	Dragboard db = event.getDragboard();
		if(db.hasFiles()) {
			event.acceptTransferModes(TransferMode.COPY_OR_MOVE);
		event.consume();
		}
    }

    @FXML
    void reset_btn(ActionEvent event) {  //UIを初期化
    	if (mainThread != null && mainThread.isAlive()) {
            System.out.println("終了操作の検知");
            mainThread.interrupt();
            try {
                mainThread.join();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    	in_pane.setStyle("-fx-border-color:  white;"
				+ "-fx-background-radius: 10;" 
				+ "-fx-border-radius: 10;"
				+ "-fx-border-width: 2");
		in_lbl.setText("Drop\nInput\nHere");
		out_pane.setStyle("-fx-border-color:  white;"
				+ "-fx-background-radius: 10;" 
				+ "-fx-border-radius: 10;"
				+ "-fx-border-width: 2");
		out_lbl.setText("Drop\nOutput\nHere");
		scrollpane.getChildren().clear();
		in_path = null;
		out_path = null;
		train_path = null;
		workin_hard.set("");
		indicator.set("");
    }

    @FXML
    void start_btn(ActionEvent event) {  //処理の開始
    	
    	//scroll panelの初期化
    	scpane_error.setValue("");
    	scpane_error.setValue("");
    	scrollpane.getChildren().clear();
    	
    	txtf = txt_field.getText();
    	
    	thumb_box = option1.isSelected();
    	check_box = option2.isSelected();
    	
    	//モードの選択確認
    	if(mode.equals("")) {
    		scpane_error.setValue("Modeが選択されていません。");
    		return;
    	}
    	
    	//フォルダパスの確認
    	if(in_path == null) {
    		scpane_error.setValue("処理したいフォルダが指定されていません。");
    		return;
    	}else {
        	if(out_path == null) {
        		out_path = in_path.getParentFile();
        		out_pane.setStyle("-fx-border-color:  #fdd23e;"
        				+ "-fx-background-radius: 10;" 
        				+ "-fx-border-radius: 10;");
        		out_lbl.setText("OK");
        	}
    	}
    	//作業の開始
    	check_mode.run(mode, in_path, out_path);
    	
    }

    @FXML
    void initialize() {
        assert analysis_btn != null : "fx:id=\"analysis_btn\" was not injected: check your FXML file 'Viewer.fxml'.";
        assert count_btn != null : "fx:id=\"count_btn\" was not injected: check your FXML file 'Viewer.fxml'.";
        assert cut_btn != null : "fx:id=\"cut_btn\" was not injected: check your FXML file 'Viewer.fxml'.";
        assert in_lbl != null : "fx:id=\"in_lbl\" was not injected: check your FXML file 'Viewer.fxml'.";
        assert in_pane != null : "fx:id=\"in_pane\" was not injected: check your FXML file 'Viewer.fxml'.";
        assert indicator_lbl != null : "fx:id=\"indicator_lbl\" was not injected: check your FXML file 'Viewer.fxml'.";
        assert option1 != null : "fx:id=\"option1\" was not injected: check your FXML file 'Viewer.fxml'.";
        assert option2 != null : "fx:id=\"option2\" was not injected: check your FXML file 'Viewer.fxml'.";
        assert out_lbl != null : "fx:id=\"out_lbl\" was not injected: check your FXML file 'Viewer.fxml'.";
        assert out_pane != null : "fx:id=\"out_pane\" was not injected: check your FXML file 'Viewer.fxml'.";
        assert pdf_btn != null : "fx:id=\"pdf_btn\" was not injected: check your FXML file 'Viewer.fxml'.";
        assert scrollpane != null : "fx:id=\"scrollpane\" was not injected: check your FXML file 'Viewer.fxml'.";
        assert sort_btn != null : "fx:id=\"sort_btn\" was not injected: check your FXML file 'Viewer.fxml'.";
        assert thumb_btn != null : "fx:id=\"thumb_btn\" was not injected: check your FXML file 'Viewer.fxml'.";
        assert train_btn != null : "fx:id=\"train_btn\" was not injected: check your FXML file 'Viewer.fxml'.";
        assert txt_field != null : "fx:id=\"txt_field\" was not injected: check your FXML file 'Viewer.fxml'.";
        assert txt_title != null : "fx:id=\"txt_title\" was not injected: check your FXML file 'Viewer.fxml'.";
        assert work_details != null : "fx:id=\"work_details\" was not injected: check your FXML file 'Viewer.fxml'.";
        
        if(Files.exists(Paths.get("Analysis_restart_point.txt"))) {
        	try {
	            Files.deleteIfExists(Paths.get("Analysis_restart_point.txt"));
	        } catch (IOException e) {
	            System.err.println("Failed to delete restart point file");
	        }
        }

        if(Files.exists(Paths.get("Sort_restart_point.txt"))) {
        	try {
	            Files.deleteIfExists(Paths.get("Sort_restart_point.txt"));
	        } catch (IOException e) {
	            System.err.println("Failed to delete restart point file");
	        }
        }

        in_lbl.setText("Drop"+newline+"Input"+newline+"Here");
        out_lbl.setText("Drop"+newline+"Output"+newline+"Here");
        
        colospace = loadSettings();
        
        scpane_text.addListener((observableValue, oldValue, newValue) -> {
        	Text newlbl = new Text(newValue);
        	newlbl.setStyle("-fx-fill: white;");
        	Platform.runLater(new Runnable() {
                @Override public void run() {
                	scrollpane.getChildren().add(0, newlbl);
                }
        	});
        });

        scpane_error.addListener((observableValue, oldValue, newValue) -> {
        	Text newlbl = new Text(newValue);
        	newlbl.setStyle("-fx-text-fill: red;");
        	Platform.runLater(new Runnable() {
                @Override public void run() {
                	scrollpane.getChildren().add(0, newlbl);
                }
        	});
        });
        
        workin_hard.addListener((observableValue, oldValue, newValue) -> {
        	Platform.runLater(new Runnable() {
                @Override public void run() {
                	work_details.setText(newValue);
                }
        	});
        });

        indicator.addListener((observableValue, oldValue, newValue) -> {
        	Platform.runLater(new Runnable() {
                @Override public void run() {
                	indicator_lbl.setText(newValue);
                }
        	});
        });

    }
    
    public static String loadSettings() {
        String[] settings = new String[2];
        try {
            File settingsFile = new File(SETTINGS_FILE);
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
        return settings[0];
    }
}
