package application;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

import com.aspose.cells.PageSetup;
import com.aspose.cells.PaperSizeType;
import com.aspose.cells.PdfSaveOptions;
import com.aspose.cells.SheetSet;
import com.aspose.cells.Workbook;
import com.aspose.cells.Worksheet;

class createPDF {

    private static final Logger LOGGER = Logger.getLogger(createPDF.class.getName());

    int run(File f, File inPath, File outPath) {
        int num = 0;
        String s = "原稿①";
        String name;

        try {
            // 行数の計測
            num = Line.getLine(f);

            Workbook demo = null;
            Worksheet demosheet = null;
            PageSetup demoSetup = null;

            // "res"フォルダからテンプレートファイルを取得
            List<String> templateFiles = getTemplateFilesFromRes();
            System.out.println(templateFiles);

            // テンプレートファイルを順次試す
            for (String templateFile : templateFiles) {
                System.out.println(templateFile);
                try {
                    InputStream is = getFileAsIOStream(templateFile);
                    demo = new Workbook(is);
                    demosheet = demo.getWorksheets().get("原稿①");
                    demoSetup = demosheet.getPageSetup();
                    break; // テンプレートのコピーが成功したらループを抜ける
                } catch (Exception e) {
                    LOGGER.log(Level.WARNING, templateFile + "の読み込みに失敗しました。次のテンプレートを試します。", e);
                }
            }

            if (demo == null || demosheet == null || demoSetup == null) {
                throw new RuntimeException("テンプレートファイルのコピーに失敗しました。");
            }

            Workbook workbook = new Workbook(f.toString());

            if (!Controller.txtf.equals("")) {
                s = Controller.txtf;
            }
            Worksheet sheet = workbook.getWorksheets().get(s);
            sheet.setPageBreakPreview(true);

            sheet.getHorizontalPageBreaks().clear();
            sheet.getVerticalPageBreaks().clear();

            PageSetup pageSetup = sheet.getPageSetup();
            if (pageSetup.getPrinterSettings() != null) {
                pageSetup.setPrinterSettings(null);
            }

            pageSetup.setPaperSize(PaperSizeType.PAPER_A_4);

            // シートのページ設定
            sheet.getHorizontalPageBreaks().add(51);
            sheet.getHorizontalPageBreaks().add(102);
            sheet.getHorizontalPageBreaks().add(153);
            sheet.getHorizontalPageBreaks().add(204);

            pageSetup.setFitToPagesTall(demoSetup.getFitToPagesTall());
            pageSetup.setFitToPagesWide(demoSetup.getFitToPagesWide());

            SheetSet sheetSet = new SheetSet(new int[]{sheet.getIndex()});
            PdfSaveOptions pdfSaveOptions = new PdfSaveOptions();
            pdfSaveOptions.setOnePagePerSheet(false);
            pdfSaveOptions.setSheetSet(sheetSet);

            makeList ml = new makeList();
            name = ml.makeName(f, inPath, outPath);
            workbook.save(new FileOutputStream(name), pdfSaveOptions);

        } catch (Exception e) {
            e.printStackTrace();
        }
        return num;
    }

    private InputStream getFileAsIOStream(String name) throws FileNotFoundException {
        File file = new File(name);
        InputStream ioStream = new FileInputStream(file);
        
        return ioStream;
    }

    private List<String> getTemplateFilesFromRes() {
        Path resPath = Paths.get("res").toAbsolutePath();  // プロジェクトフォルダ内のresフォルダを指定
        List<String> templateFiles = new ArrayList<>();
        try {
            File resDirectory = new File(resPath.toString());
            File[] files = resDirectory.listFiles((dir, name) -> name.toLowerCase().endsWith(".xlsx"));
            if (files != null) {
                for (File file : files) {
                    templateFiles.add(file.toString());  // ファイル名だけをリストに追加
                }
            }
        } catch (Exception e) {
            LOGGER.log(Level.SEVERE, "リソースディレクトリの読み込みに失敗しました。", e);
        }
        return templateFiles;
    }

}
