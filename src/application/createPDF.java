package application;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;

import com.aspose.cells.PageSetup;
import com.aspose.cells.PaperSizeType;
import com.aspose.cells.PdfSaveOptions;
import com.aspose.cells.SheetSet;
import com.aspose.cells.Workbook;
import com.aspose.cells.Worksheet;

class createPDF{
	
	int run(File f, File inPath, File outPath){
		int num = 0;
		String s = "原稿①";
		String name;
		try {
			
			//行数の計測
			num = Line.getLine(f);
			
			//EXCELファイルのテンプレートを取得
			InputStream is = getFileAsIOStream("temp.xlsx");
			Workbook demo = new Workbook(is);
			Worksheet demosheet = demo.getWorksheets().get("原稿①");
			PageSetup demoSetup = demosheet.getPageSetup();
			
			Workbook workbook = new Workbook(f.toString());
			
			if(!Controller.txtf.equals("")) {
				s = Controller.txtf;
			}
			Worksheet sheet = workbook.getWorksheets().get(s);
			sheet.setPageBreakPreview(true);
			
			sheet.getHorizontalPageBreaks().clear();
			sheet.getVerticalPageBreaks().clear();
			
			PageSetup pageSetup = sheet.getPageSetup();
			if(pageSetup.getPrinterSettings() != null)
			{
				pageSetup.setPrinterSettings(null);
			}
			
			pageSetup.setPaperSize(PaperSizeType.PAPER_A_4);
			
			//シートのページ設定
			sheet.getHorizontalPageBreaks().add(51);
			sheet.getHorizontalPageBreaks().add(102);
			sheet.getHorizontalPageBreaks().add(153);
			sheet.getHorizontalPageBreaks().add(204);
			
			pageSetup.setFitToPagesTall(demoSetup.getFitToPagesTall());
			pageSetup.setFitToPagesWide(demoSetup.getFitToPagesWide());
			
			SheetSet sheetSet = new SheetSet(new int[] { sheet.getIndex() });
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
	
	private InputStream getFileAsIOStream(final String fileName) 
    {
        InputStream ioStream = this.getClass()
            .getClassLoader()
            .getResourceAsStream(fileName);
        
        if (ioStream == null) {
            throw new IllegalArgumentException(fileName + " is not found");
        }
        return ioStream;
    }
}