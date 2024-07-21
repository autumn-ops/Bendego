package application;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.List;
import java.util.ResourceBundle;
import java.util.UUID;
import java.util.logging.Level;
import java.util.logging.Logger;

import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.control.ChoiceBox;
import javafx.scene.control.Label;
import javafx.scene.image.Image;
import javafx.scene.input.DragEvent;
import javafx.scene.input.TransferMode;
import javafx.stage.Modality;
import javafx.stage.Stage;

public class Popup {

    @FXML
    private ResourceBundle resources;

    @FXML
    private URL location;

    @FXML
    private Label back_lbl;

    @FXML
    private Button closeButton;

    @FXML
    private ChoiceBox<String> color_box;

    @FXML
    private Label excel_lbl;

    private static final Logger LOGGER = Logger.getLogger(Popup.class.getName());

    @FXML
    void back_do(DragEvent event) {
        if (event.getDragboard().hasFiles()) {
            event.acceptTransferModes(TransferMode.COPY);
        }
        event.consume();
    }

    @FXML
    void back_dd(DragEvent event) {
        if (event.getDragboard().hasFiles()) {
        	event.acceptTransferModes(TransferMode.COPY);
            List<File> files = event.getDragboard().getFiles();
            for (File file : files) {
                if (file.getName().endsWith(".jpg")) {
                    try {
                        Image image = new Image(file.toURI().toString());
                        if (image.getWidth() >= 7000 && image.getHeight() >= 7000) {
                            Path targetDir = Paths.get("screen");
                            if (!Files.exists(targetDir)) {
                                Files.createDirectory(targetDir);
                            }
                            Path targetPath = targetDir.resolve(file.getName());
                            while (Files.exists(targetPath)) {
                                String newFileName = UUID.randomUUID().toString() + ".jpg";
                                targetPath = targetDir.resolve(newFileName);
                            }
                            Files.copy(file.toPath(), targetPath, StandardCopyOption.REPLACE_EXISTING);
                            back_lbl.setText(file.getName());
                            break; // .jpgファイルを一つだけ処理する
                        }
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
            }
        }
        event.setDropCompleted(true);
        event.consume();
    }

    @FXML
    void excel_do(DragEvent event) {
        if (event.getDragboard().hasFiles()) {
            event.acceptTransferModes(TransferMode.COPY);
        }
        event.consume();
    }

    @FXML
    void excel_dd(DragEvent event) {
        if (event.getDragboard().hasFiles()) {
        	event.acceptTransferModes(TransferMode.COPY);
            List<File> files = event.getDragboard().getFiles();
            for (File file : files) {
                if (file.getName().endsWith(".xlsx")) {
                    try {
                        Path targetDir = Paths.get("res");
                        if (!Files.exists(targetDir)) {
                            Files.createDirectory(targetDir);
                        }
                        Path targetPath = targetDir.resolve(file.getName());
                        while (Files.exists(targetPath)) {
                            String newFileName = UUID.randomUUID().toString() + ".xlsx";
                            targetPath = targetDir.resolve(newFileName);
                        }
                        Files.copy(file.toPath(), targetPath, StandardCopyOption.REPLACE_EXISTING);
                        excel_lbl.setText(file.getName());
                        break; // .xlsxファイルを一つだけ処理する
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
            }
        }
        event.setDropCompleted(true);
        event.consume();
    }

    @FXML
    void close_action(ActionEvent event) {
        // 閉じるボタンがクリックされた時に設定を保存してウィンドウを閉じる
        saveSettings();
        Stage stage = (Stage) closeButton.getScene().getWindow();
        stage.close();
    }

    @FXML
    void initialize() {
        assert back_lbl != null : "fx:id=\"back_lbl\" was not injected: check your FXML file 'Popup.fxml'.";
        assert closeButton != null : "fx:id=\"closeButton\" was not injected: check your FXML file 'Popup.fxml'.";
        assert color_box != null : "fx:id=\"color_box\" was not injected: check your FXML file 'Popup.fxml'.";
        assert excel_lbl != null : "fx:id=\"excel_lbl\" was not injected: check your FXML file 'Popup.fxml'.";

        // カラースペースリストを設定ファイルから読み込む
        ObservableList<String> observableColorSpaces = FXCollections.observableArrayList(getTemplateFilesFromRes());
        color_box.setItems(observableColorSpaces);
        color_box.setStyle("-fx-control-inner-background: black;"
        		+ "-fx-text-fill : white;"
        		+ "-fx-background-color: black;"
        		+ "-fx-border-color: #fdd23e;");

        // 現在の設定を読み込む
        loadSettings();
    }

    
    private List<String> getTemplateFilesFromRes() {
        Path resPath = Paths.get("res").toAbsolutePath();  // プロジェクトフォルダ内のresフォルダを指定
        List<String> templateFiles = new ArrayList<>();
        try {
            File resDirectory = new File(resPath.toString());
            File[] files = resDirectory.listFiles((dir, name) -> name.toLowerCase().endsWith(".icc"));
            if (files != null) {
                for (File file : files) {
                    templateFiles.add(file.getName());  // ファイル名だけをリストに追加
                }
            }
        } catch (Exception e) {
            LOGGER.log(Level.SEVERE, "リソースディレクトリの読み込みに失敗しました。", e);
        }
        return templateFiles;
    }

    public void showPopup() {
        try {
            // Popup.fxmlを読み込む
            FXMLLoader loader = new FXMLLoader(getClass().getResource("Popup.fxml"));
            Parent root = loader.load();

            // ポップアップのステージを作成
            Stage popupStage = new Stage();
            popupStage.initModality(Modality.APPLICATION_MODAL);

            // ポップアップのシーンを作成
            Scene popupScene = new Scene(root);
            popupStage.setScene(popupScene);

            // ポップアップを表示
            popupStage.showAndWait();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void saveSettings() {
        SettingsManager.saveSettings(color_box.getValue(), excel_lbl.getText());
        SettingsManager.saveColorSpaces(color_box.getItems());
    }

    private void loadSettings() {
        String[] settings = SettingsManager.loadSettings();
        if (settings[0] != null) {
            if (!color_box.getItems().contains(settings[0])) {
                color_box.getItems().add(settings[0]);
            }
            color_box.setValue(settings[0]);
        }
    }
}
