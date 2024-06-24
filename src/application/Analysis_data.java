package application;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.PushbackInputStream;
import java.util.Arrays;

import org.apache.poi.EncryptedDocumentException;
import org.apache.poi.hssf.usermodel.HSSFWorkbook;
import org.apache.poi.poifs.filesystem.FileMagic;
import org.apache.poi.ss.usermodel.Cell;
import org.apache.poi.ss.usermodel.CellType;
import org.apache.poi.ss.usermodel.Row;
import org.apache.poi.ss.usermodel.Sheet;
import org.apache.poi.ss.usermodel.Workbook;
import org.apache.poi.ss.usermodel.WorkbookFactory;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;

class Analysis_data {
	private static Workbook excel;

    private static final int MAX_PATTERN_LENGTH = 44;
    
	int getLine(File f) {
		
		String s = "原稿①";
		if(!Controller.txtf.equals("")) {
			s = Controller.txtf;
		}

	    int startline = 11;
	    
	    int column = 9;
	    
	    int num = 0;

		try {
			FileInputStream inst = new FileInputStream(f);
			
			PushbackInputStream inp = new PushbackInputStream(inst, MAX_PATTERN_LENGTH);
	        byte[] data = new byte[MAX_PATTERN_LENGTH];
	        inp.read(data, 0, MAX_PATTERN_LENGTH);
	        inp.unread(data);
	        FileMagic fm = FileMagic.valueOf(data);
	        if (FileMagic.OOXML==fm){
	        	excel = new XSSFWorkbook(inp);
	        }else if(FileMagic.OLE2==fm){
	        	excel = new HSSFWorkbook(inp);
	        }else {
	        	excel = WorkbookFactory.create(inp);
	        }
	        
        // シート名がわかっている場合
        Sheet sheet = excel.getSheet(s);
        int[] page = sheet.getRowBreaks();
        
        int skipnum = 0;
        
        int score;
        
        for(int i=startline; i<=page[page.length-1]; i++) {

        	//アイテムコードの行を取得
        	if(skipnum == 0) {
        		score = 0;
        		//n行目 0~
        		Row row = sheet.getRow(i);//複数行使用している場合は、一番上の行
        		
				String item = getValue(row.getCell(column-4)); //商品名
				String goods = getValue(row.getCell(column-1));//商品記号
				String code = getValue(row.getCell(column));   //アイテムコード
				

    	        if(item != "") {
    	        	score += 2;
    	        }
    	        if(goods != "") {
    	        	score += 1;
    	        }
    	        if(code != "") {
    	        	score += 1;
	        	}
    	        if(score > 1) {
    	        	num++;
    	        }
    	    }
        	
        	//原稿の最後
        	if(i == page[page.length-1]) {
        	}
        	//ページの終端
        	else if(contains(page, i)) {
        		i += startline;
        	}
        	//４行ごとにリセット
        	if(skipnum == 3) {skipnum = 0;}
        	else {skipnum++;}
        }
        excel.close();
		} catch (EncryptedDocumentException | IOException e) {
			Controller.scpane_text.set("Excelデータの読み込みに失敗しました。");
			e.printStackTrace();
		}
		return num;
	}
	
	public static boolean contains(final int[] page, final int key) {
        return Arrays.stream(page).anyMatch(i -> i == key);
    }
	
	static String getValue(Cell cell) {

		CellType cType = cell.getCellType();
		switch(cType) {
			case STRING:
				return cell.getStringCellValue().replace(Controller.newline, "").replace(" ", "");
	      	case NUMERIC:
	      		return String.valueOf(cell.getNumericCellValue());
	        default:
	        	return "";
	      }
	}
}